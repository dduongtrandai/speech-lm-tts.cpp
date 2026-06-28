#include "vieneu_v3_onnx.h"

#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

std::string VieneuV3OnnxEngine::join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') {
        return dir + name;
    }
#ifdef _WIN32
    return dir + "\\" + name;
#else
    return dir + "/" + name;
#endif
}

bool VieneuV3OnnxEngine::file_exists(const std::string& path) {
    if (path.empty()) {
        return false;
    }
#ifdef _WIN32
    const DWORD attrs = GetFileAttributesA(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

bool VieneuV3OnnxEngine::read_text_file(const std::string& path, std::string& out) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs.is_open()) {
        return false;
    }
    std::stringstream buffer;
    buffer << fs.rdbuf();
    out = buffer.str();
    return true;
}

bool VieneuV3OnnxEngine::validate_assets(const VieneuV3OnnxInit& init, std::string& error) {
    const std::string onnx_dir = init.onnx_dir.empty() ? init.model_dir : init.onnx_dir;
    const std::string codec_dir = init.codec_dir;
    const std::string config_path = init.config_path.empty() ? join_path(init.model_dir, "config.json") : init.config_path;
    const std::string tokenizer_path = init.tokenizer_path.empty() ? join_path(init.model_dir, "tokenizer.json") : init.tokenizer_path;

    const std::vector<std::string> required = {
        join_path(onnx_dir, "vieneu_prefill.onnx"),
        join_path(onnx_dir, "vieneu_decode_step.onnx"),
        join_path(onnx_dir, "vieneu_acoustic_cached.onnx"),
        join_path(onnx_dir, "vieneu_v3_heads.npz"),
        config_path,
        tokenizer_path,
        join_path(codec_dir, "moss_audio_tokenizer_decode_full.onnx"),
        join_path(codec_dir, "moss_audio_tokenizer_encode.onnx"),
    };

    for (const std::string& path : required) {
        if (!file_exists(path)) {
            error = "Missing required VieNeu v3 ONNX asset: " + path;
            return false;
        }
    }
    codec_encode_path_ = join_path(codec_dir, "moss_audio_tokenizer_encode.onnx");
    return true;
}

bool VieneuV3OnnxEngine::load_session(const std::string& path, std::unique_ptr<Ort::Session>& session, std::string& error) {
    try {
#ifdef _WIN32
        std::wstring w_path(path.begin(), path.end());
        session = std::make_unique<Ort::Session>(*env_, w_path.c_str(), *session_options_);
#else
        session = std::make_unique<Ort::Session>(*env_, path.c_str(), *session_options_);
#endif
        return true;
    } catch (const std::exception& e) {
        error = "Failed to load ONNX session " + path + ": " + e.what();
        return false;
    }
}

bool VieneuV3OnnxEngine::load_voices(const std::string& voices_path, std::string& error) {
    voices_json_.clear();
    if (voices_path.empty()) {
        return true;
    }
    if (!read_text_file(voices_path, voices_json_)) {
        error = "Failed to read VieNeu v3 voices JSON: " + voices_path;
        return false;
    }
    return true;
}

bool VieneuV3OnnxEngine::initialize(const VieneuV3OnnxInit& init, std::string& error) {
    initialized_ = false;
    env_.reset();
    prefill_session_.reset();
    decode_session_.reset();
    acoustic_session_.reset();
    codec_decode_session_.reset();
    session_options_.reset();
    codec_encode_path_.clear();

    if (init.model_dir.empty() && init.onnx_dir.empty()) {
        error = "VieNeu v3 requires model_dir or onnx_dir.";
        return false;
    }
    if (init.codec_dir.empty()) {
        error = "VieNeu v3 requires codec_dir with MOSS ONNX codec files.";
        return false;
    }
    if (!validate_assets(init, error)) {
        return false;
    }

    env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VieneuV3Onnx");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(init.n_threads > 0 ? init.n_threads : 4);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const std::string onnx_dir = init.onnx_dir.empty() ? init.model_dir : init.onnx_dir;
    if (!load_session(join_path(onnx_dir, "vieneu_prefill.onnx"), prefill_session_, error) ||
        !load_session(join_path(onnx_dir, "vieneu_decode_step.onnx"), decode_session_, error) ||
        !load_session(join_path(onnx_dir, "vieneu_acoustic_cached.onnx"), acoustic_session_, error) ||
        !load_session(join_path(init.codec_dir, "moss_audio_tokenizer_decode_full.onnx"), codec_decode_session_, error)) {
        return false;
    }

    if (!load_voices(init.voices_json_path, error)) {
        return false;
    }

    initialized_ = true;
    return true;
}

bool VieneuV3OnnxEngine::synthesize(const VieneuV3OnnxParams& params, std::vector<float>& out_audio, std::string& error) {
    out_audio.clear();
    if (!initialized_) {
        error = "VieNeu v3 ONNX engine is not initialized.";
        return false;
    }
    if (params.text.empty()) {
        error = "VieNeu v3 synthesis requires non-empty text.";
        return false;
    }

    error = "VieNeu v3 native ONNX asset loading is available, but tokenization/NPZ embedding decode and autoregressive generation are not implemented in this runtime build yet.";
    return false;
}
