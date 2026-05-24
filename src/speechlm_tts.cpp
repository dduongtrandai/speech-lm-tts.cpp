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

// Thread-local error reporting
thread_local std::string g_last_error = "";

static void set_last_error(const std::string& err) {
    g_last_error = err;
}

static bool parse_voice_embedding_from_json(const std::string& voices_json, const std::string& voice_id, std::vector<float>& out_embedding) {
    out_embedding.clear();

    const std::string voice_pattern = "\"" + voice_id + "\"[^\\}]*\"(embedding|codes)\"\\s*:\\s*\\[([^\\]]*)\\]";
    const std::regex re(voice_pattern);
    std::smatch m;
    if (!std::regex_search(voices_json, m, re)) {
        return false;
    }

    const std::string arr = m[2].str();
    std::stringstream ss(arr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            out_embedding.push_back(std::stof(token));
        } catch (...) {
            return false;
        }
    }

    return out_embedding.size() == 128;
}

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

SLM_API int slm_synthesize(struct slm_context * slm, const struct slm_tts_params * params, struct slm_audio * out) {
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
