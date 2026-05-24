#include "speechlm_tts.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>

static void print_usage(char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n\n"
              << "Options:\n"
              << "  -m, --model PATH       Path to GGUF model (required)\n"
              << "  -d, --decoder PATH     Path to ONNX decoder (required)\n"
              << "  -e, --encoder PATH     Path to ONNX encoder (optional)\n"
              << "  -t, --text TEXT        Text to synthesize (default: 'Xin chào các bạn.')\n"
              << "  -v, --voice ID         Preset voice ID (default: none)\n"
              << "  -o, --output PATH      Path to output WAV file (default: 'output.wav')\n"
              << "  --voices-json PATH     Path to voices.json file (optional)\n"
              << "  -h, --help             Show this help message\n";
}

static bool write_wav_file(const std::string& path, const float* samples, int n_samples, int sample_rate) {
    std::ofstream fs(path, std::ios::binary);
    if (!fs.is_open()) return false;

    struct WavHeader {
        char riff[4] = {'R', 'I', 'F', 'F'};
        int32_t overall_size;
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt_chunk_marker[4] = {'f', 'm', 't', ' '};
        int32_t length_of_fmt = 16;
        int16_t format_type = 1;
        int16_t channels = 1;
        int32_t sample_rate;
        int32_t byterate;
        int16_t block_align;
        int16_t bits_per_sample = 16;
        char data_chunk_header[4] = {'d', 'a', 't', 'a'};
        int32_t data_size;
    } header;

    int32_t data_size = n_samples * sizeof(int16_t);
    header.overall_size = 36 + data_size;
    header.sample_rate = sample_rate;
    header.byterate = sample_rate * sizeof(int16_t);
    header.block_align = sizeof(int16_t);
    header.data_size = data_size;

    fs.write((const char*)&header, sizeof(header));

    for (int i = 0; i < n_samples; ++i) {
        float sample = samples[i];
        if (sample < -1.0f) sample = -1.0f;
        if (sample > 1.0f) sample = 1.0f;
        int16_t pcm = (int16_t)(sample * 32767.0f);
        fs.write((const char*)&pcm, sizeof(pcm));
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::string model_path = "";
    std::string decoder_path = "";
    std::string encoder_path = "";
    std::string text = "Xin chào các bạn.";
    std::string voice_id = "";
    std::string output_path = "output.wav";
    std::string voices_json_path = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (arg == "-d" || arg == "--decoder") {
            if (i + 1 < argc) decoder_path = argv[++i];
        } else if (arg == "-e" || arg == "--encoder") {
            if (i + 1 < argc) encoder_path = argv[++i];
        } else if (arg == "-t" || arg == "--text") {
            if (i + 1 < argc) text = argv[++i];
        } else if (arg == "-v" || arg == "--voice") {
            if (i + 1 < argc) voice_id = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "--voices-json") {
            if (i + 1 < argc) voices_json_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model option is required.\n";
        print_usage(argv[0]);
        return 1;
    }
    if (decoder_path.empty()) {
        std::cerr << "Error: --decoder option is required.\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "[CLI] Initializing SpeechLM TTS context...\n"
              << "  Model: " << model_path << "\n"
              << "  Decoder: " << decoder_path << "\n";
    if (!encoder_path.empty()) {
        std::cout << "  Encoder: " << encoder_path << "\n";
    }

    slm_init_params params;
    slm_init_default_params(&params);
    params.model_path = model_path.c_str();
    params.decoder_path = decoder_path.c_str();
    if (!encoder_path.empty()) {
        params.encoder_path = encoder_path.c_str();
    }
    if (!voices_json_path.empty()) {
        params.voices_json_path = voices_json_path.c_str();
    }

    slm_context* ctx = slm_init(&params);
    if (!ctx) {
        std::cerr << "[CLI] Initialization failed: " << slm_last_error() << "\n";
        return 1;
    }
    std::cout << "[CLI] Initialization successful.\n";

    if (!voice_id.empty() && !voices_json_path.empty()) {
        std::cout << "[CLI] Setting preset voice ID: " << voice_id << "\n";
        if (slm_set_preset_voice(ctx, voice_id.c_str()) != 0) {
            std::cerr << "[CLI] Failed to set voice: " << slm_last_error() << "\n";
            slm_free(ctx);
            return 1;
        }
    }

    std::cout << "[CLI] Synthesizing text: \"" << text << "\"\n";
    slm_tts_params tts_params;
    slm_tts_default_params(&tts_params);
    tts_params.text = text.c_str();

    slm_audio audio;
    memset(&audio, 0, sizeof(audio));

    if (slm_synthesize(ctx, &tts_params, &audio) != 0) {
        std::cerr << "[CLI] Synthesis failed: " << slm_last_error() << "\n";
        slm_free(ctx);
        return 1;
    }

    std::cout << "[CLI] Synthesis successful. n_samples=" << audio.n_samples 
              << ", sample_rate=" << audio.sample_rate << "Hz\n";

    std::cout << "[CLI] Writing output file: " << output_path << "\n";
    if (!write_wav_file(output_path, audio.samples, audio.n_samples, audio.sample_rate)) {
        std::cerr << "[CLI] Failed to write WAV file.\n";
        slm_audio_free(&audio);
        slm_free(ctx);
        return 1;
    }

    std::cout << "[CLI] Done.\n";
    slm_audio_free(&audio);
    slm_free(ctx);
    return 0;
}
