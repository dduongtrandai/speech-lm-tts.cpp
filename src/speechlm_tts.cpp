#include "speechlm_tts.h"
#include "llm/llama_backend.h"
#include "codecs/neucodec_onnx.h"
#include "profiles/neutts_air.h"
#include "profiles/vieneu.h"
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <regex>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <cmath>

// Thread-local error reporting
thread_local std::string g_last_error = "";

static void set_last_error(const std::string& err) {
    g_last_error = err;
}

static std::string escape_regex(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() * 2);
    for (char ch : text) {
        switch (ch) {
            case '\\': case '.': case '^': case '$': case '|': case '?':
            case '*': case '+': case '(': case ')': case '[': case ']':
            case '{': case '}':
                escaped.push_back('\\');
                break;
            default:
                break;
        }
        escaped.push_back(ch);
    }
    return escaped;
}

static bool parse_json_string_field(const std::string& json, const std::string& key, std::string& out_value) {
    const std::regex re("\"" + escape_regex(key) + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (!std::regex_search(json, m, re)) {
        return false;
    }
    out_value = m[1].str();
    return true;
}

static bool parse_numeric_array_after_key(const std::string& json, size_t object_pos, std::vector<float>& out_values) {
    const size_t codes_pos = json.find("\"codes\"", object_pos);
    const size_t embedding_pos = json.find("\"embedding\"", object_pos);
    size_t field_pos = std::string::npos;
    if (codes_pos != std::string::npos && embedding_pos != std::string::npos) {
        field_pos = std::min(codes_pos, embedding_pos);
    } else {
        field_pos = codes_pos != std::string::npos ? codes_pos : embedding_pos;
    }
    if (field_pos == std::string::npos) {
        return false;
    }

    const size_t arr_start = json.find('[', field_pos);
    const size_t arr_end = json.find(']', arr_start);
    if (arr_start == std::string::npos || arr_end == std::string::npos) {
        return false;
    }

    out_values.clear();
    std::stringstream ss(json.substr(arr_start + 1, arr_end - arr_start - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            out_values.push_back(std::stof(token));
        } catch (...) {
            return false;
        }
    }
    return !out_values.empty();
}

static bool parse_voice_embedding_from_json(const std::string& voices_json, const std::string& voice_id, std::vector<float>& out_embedding) {
    out_embedding.clear();

    const std::string quoted_id = "\"" + voice_id + "\"";
    const size_t object_pos = voices_json.find(quoted_id);
    if (object_pos == std::string::npos) {
        return false;
    }

    if (!parse_numeric_array_after_key(voices_json, object_pos, out_embedding)) {
        return false;
    }

    return out_embedding.size() == 128;
}

static bool set_first_available_voice(struct slm_context * slm);

enum class SlmProfile {
    NEUTTS_AIR_V1,
    VIENEU_V2_TURBO
};

struct slm_context {
    SlmProfile profile = SlmProfile::NEUTTS_AIR_V1;
    std::string current_voice_id = "";
    std::vector<float> current_voice_embedding;

    std::unique_ptr<LlamaBackend> llama;
    std::unique_ptr<NeuCodecOnnx> codec_decoder;
    std::unique_ptr<NeuCodecOnnx> codec_encoder;

    std::string voices_json = "";
};

static void normalize_output_level(std::vector<float>& audio) {
    float peak = 0.0f;
    for (float sample : audio) {
        if (std::isfinite(sample)) {
            peak = std::max(peak, std::abs(sample));
        }
    }

    if (peak < 1.0e-5f || peak >= 0.80f) {
        return;
    }

    const float gain = std::min(0.80f / peak, 12.0f);
    for (float& sample : audio) {
        if (!std::isfinite(sample)) {
            sample = 0.0f;
        } else {
            sample = std::clamp(sample * gain, -1.0f, 1.0f);
        }
    }
}

SLM_API const char * slm_version(void) {
    return "0.1.0";
}

SLM_API const char * slm_last_error(void) {
    return g_last_error.c_str();
}

SLM_API void slm_audio_free(struct slm_audio * a) {
    if (a) {
        if (a->samples) {
            free(a->samples);
            a->samples = nullptr;
        }
        a->n_samples = 0;
    }
}

SLM_API void slm_init_default_params(struct slm_init_params * p) {
    if (p) {
        p->abi_version = 1;
        p->model_path = nullptr;
        p->encoder_path = nullptr;
        p->decoder_path = nullptr;
        p->voices_json_path = nullptr;
        p->n_ctx = 2048;
        p->n_threads = 4;
        p->n_gpu_layers = 0;
        p->flash_attn = false;
        p->mlock = true;
    }
}

SLM_API struct slm_context * slm_init(const struct slm_init_params * params) {
    if (!params || !params->model_path || !params->decoder_path) {
        set_last_error("Invalid initialization parameters: model_path and decoder_path are required.");
        return nullptr;
    }

    auto ctx = std::make_unique<slm_context>();

    // Profile Selection heuristic
    std::string model_path_str(params->model_path);
    std::transform(model_path_str.begin(), model_path_str.end(), model_path_str.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (model_path_str.find("vieneu") != std::string::npos) {
        ctx->profile = SlmProfile::VIENEU_V2_TURBO;
    } else {
        ctx->profile = SlmProfile::NEUTTS_AIR_V1;
    }

    // Initialize llama backend
    ctx->llama = std::make_unique<LlamaBackend>();
    LlamaBackendParams llama_params;
    llama_params.model_path = params->model_path;
    llama_params.n_ctx = params->n_ctx > 0 ? params->n_ctx : 2048;
    llama_params.n_threads = params->n_threads > 0 ? params->n_threads : 4;
    llama_params.n_gpu_layers = params->n_gpu_layers;
    llama_params.flash_attn = params->flash_attn;
    llama_params.mlock = params->mlock;

    if (!ctx->llama->initialize(llama_params)) {
        set_last_error("Failed to initialize llama.cpp model context.");
        return nullptr;
    }

    // Initialize ONNX decoder
    ctx->codec_decoder = std::make_unique<NeuCodecOnnx>();
    if (!ctx->codec_decoder->initialize(params->decoder_path, llama_params.n_threads)) {
        set_last_error("Failed to initialize ONNX decoder.");
        return nullptr;
    }

    // Initialize optional ONNX encoder
    if (params->encoder_path && strlen(params->encoder_path) > 0) {
        ctx->codec_encoder = std::make_unique<NeuCodecOnnx>();
        if (!ctx->codec_encoder->initialize(params->encoder_path, llama_params.n_threads)) {
            set_last_error("Failed to initialize ONNX encoder.");
            return nullptr;
        }
    }

    // Load preset voices if voices_json_path is provided
    if (params->voices_json_path && strlen(params->voices_json_path) > 0) {
        std::ifstream fs(params->voices_json_path);
        if (fs.is_open()) {
            std::stringstream buffer;
            buffer << fs.rdbuf();
            ctx->voices_json = buffer.str();

            std::string default_voice;
            if (parse_json_string_field(ctx->voices_json, "default_voice", default_voice)) {
                std::vector<float> emb;
                if (parse_voice_embedding_from_json(ctx->voices_json, default_voice, emb)) {
                    ctx->current_voice_id = default_voice;
                    ctx->current_voice_embedding = std::move(emb);
                }
            }
            if (ctx->current_voice_embedding.empty()) {
                set_first_available_voice(ctx.get());
            }
        }
    }

    return ctx.release();
}

SLM_API void slm_free(struct slm_context * slm) {
    if (slm) {
        delete slm;
    }
}

SLM_API void slm_tts_default_params(struct slm_tts_params * p) {
    if (p) {
        p->abi_version = 1;
        p->text = nullptr;
        p->voice_id = nullptr;
        p->voice_embedding = nullptr;
        p->temperature = 0.4f;
        p->top_k = 50;
        p->max_chars = 256;
        p->max_tokens = 2048;
        p->skip_normalize = false;
        p->skip_phonemize = false;
        p->apply_watermark = true;
    }
}

static int slm_synthesize_impl(struct slm_context * slm, const struct slm_tts_params * params, struct slm_audio * out) {
    if (!slm || !params || !params->text || !out) {
        set_last_error("Invalid synthesize arguments: context, params, text, and out are required.");
        return -1;
    }

    std::vector<float> voice_emb(128, 0.0f);

    if (params->voice_id && params->voice_id[0] != '\0') {
        if (slm_set_preset_voice(slm, params->voice_id) != 0) {
            return -1;
        }
    }

    if (params->voice_embedding) {
        std::copy(params->voice_embedding, params->voice_embedding + 128, voice_emb.begin());
    } else if (!slm->current_voice_embedding.empty()) {
        voice_emb = slm->current_voice_embedding;
    }

    std::vector<float> out_audio;
    bool success = false;

    if (slm->profile == SlmProfile::NEUTTS_AIR_V1) {
        success = NeuTtsAirProfile::synthesize(
            *slm->llama,
            *slm->codec_decoder,
            params->text,
            params->temperature,
            params->top_k,
            params->max_tokens,
            params->skip_phonemize,
            out_audio
        );
    } else if (slm->profile == SlmProfile::VIENEU_V2_TURBO) {
        success = VieneuProfile::synthesize(
            *slm->llama,
            *slm->codec_decoder,
            params->text,
            voice_emb,
            params->temperature,
            params->top_k,
            params->max_tokens,
            params->skip_phonemize,
            out_audio
        );
    }

    if (!success || out_audio.empty()) {
        set_last_error("Synthesis pipeline failed or generated empty audio.");
        return -1;
    }

    normalize_output_level(out_audio);

    // Allocate samples on the heap (using malloc to match standard C free)
    out->samples = (float*)malloc(out_audio.size() * sizeof(float));
    if (!out->samples) {
        set_last_error("Memory allocation failed for audio output buffer.");
        return -1;
    }

    std::copy(out_audio.begin(), out_audio.end(), out->samples);
    out->n_samples = (int)out_audio.size();
    out->sample_rate = 24000;
    out->channels = 1;

    return 0;
}

static void clear_audio_output(struct slm_audio * out) {
    if (!out) {
        return;
    }
    if (out->samples) {
        free(out->samples);
    }
    out->samples = nullptr;
    out->n_samples = 0;
    out->sample_rate = 0;
    out->channels = 0;
}

static int slm_synthesize_cpp_guard(struct slm_context * slm, const struct slm_tts_params * params, struct slm_audio * out) {
    try {
        return slm_synthesize_impl(slm, params, out);
    } catch (const std::exception& e) {
        clear_audio_output(out);
        set_last_error(std::string("Unhandled C++ exception during synthesis: ") + e.what());
        return -1;
    } catch (...) {
        clear_audio_output(out);
        set_last_error("Unhandled unknown C++ exception during synthesis.");
        return -1;
    }
}

SLM_API int slm_synthesize(struct slm_context * slm, const struct slm_tts_params * params, struct slm_audio * out) {
    return slm_synthesize_cpp_guard(slm, params, out);
}

SLM_API int slm_encode_reference(struct slm_context * slm, const char * ref_audio_path, float * out_embedding_128) {
    if (!slm || !ref_audio_path || !out_embedding_128) {
        set_last_error("Invalid encode_reference arguments.");
        return -1;
    }

    if (!slm->codec_encoder) {
        set_last_error("Speaker encoder was not loaded at initialization.");
        return -1;
    }

    std::vector<float> waveform;
    if (!read_wav_file_24k_mono(ref_audio_path, waveform) || waveform.empty()) {
        set_last_error("Failed to read reference WAV file or it was empty.");
        return -1;
    }

    std::vector<float> emb;
    if (!slm->codec_encoder->encode_speaker(waveform, emb)) {
        set_last_error("Failed to extract speaker embedding using ONNX encoder.");
        return -1;
    }
    if (emb.size() != 128) {
        set_last_error("Speaker encoder returned invalid embedding size (expected 128).");
        return -1;
    }

    std::copy(emb.begin(), emb.end(), out_embedding_128);
    return 0;
}

SLM_API int slm_list_preset_voices(struct slm_context * slm, char * out_json, int max_len) {
    if (!slm || !out_json || max_len <= 0) {
        set_last_error("Invalid list_preset_voices arguments.");
        return -1;
    }

    std::string result = "[]";
    if (!slm->voices_json.empty()) {
        result = slm->voices_json;
    }

    if ((int)result.size() >= max_len) {
        set_last_error("Output buffer size is too small to fit the voices JSON string.");
        return -1;
    }

    strncpy(out_json, result.c_str(), max_len - 1);
    out_json[max_len - 1] = '\0';
    return 0;
}

SLM_API int slm_set_preset_voice(struct slm_context * slm, const char * voice_id) {
    if (!slm || !voice_id) {
        set_last_error("Invalid set_preset_voice arguments.");
        return -1;
    }

    if (slm->voices_json.empty()) {
        set_last_error("No voices.json loaded to look up the voice.");
        return -1;
    }

    std::vector<float> emb;
    if (parse_voice_embedding_from_json(slm->voices_json, voice_id, emb)) {
        slm->current_voice_id = voice_id;
        slm->current_voice_embedding = std::move(emb);
        return 0;
    }

    set_last_error("Failed to find voice embedding/codes[128] for voice ID: " + std::string(voice_id));
    return -1;
}

static bool set_first_available_voice(struct slm_context * slm) {
    if (!slm || slm->voices_json.empty()) {
        return false;
    }

    const std::regex preset_re("\"([^\"]+)\"\\s*:\\s*\\{[^\\}]*\"(?:embedding|codes)\"\\s*:");
    auto begin = std::sregex_iterator(slm->voices_json.begin(), slm->voices_json.end(), preset_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        std::string id = (*it)[1].str();
        if (id == "meta" || id == "presets") {
            continue;
        }
        std::vector<float> emb;
        if (parse_voice_embedding_from_json(slm->voices_json, id, emb)) {
            slm->current_voice_id = id;
            slm->current_voice_embedding = std::move(emb);
            return true;
        }
    }
    return false;
}
