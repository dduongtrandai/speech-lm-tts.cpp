#ifndef VIENEU_V3_ONNX_H
#define VIENEU_V3_ONNX_H

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

struct VieneuV3OnnxInit {
    std::string model_dir;
    std::string onnx_dir;
    std::string codec_dir;
    std::string config_path;
    std::string tokenizer_path;
    std::string voices_json_path;
    int n_threads = 4;
};

struct VieneuV3OnnxParams {
    std::string text;
    std::string voice_id;
    std::string ref_audio_path;
    float temperature = 0.8f;
    int top_k = 25;
    float top_p = 0.95f;
    int max_new_frames = 300;
    float repetition_penalty = 1.2f;
    int max_chars = 384;
    bool apply_watermark = true;
};

class VieneuV3OnnxEngine {
public:
    bool initialize(const VieneuV3OnnxInit& init, std::string& error);
    bool synthesize(const VieneuV3OnnxParams& params, std::vector<float>& out_audio, std::string& error);

    const std::string& voices_json() const { return voices_json_; }
    int sample_rate() const { return 48000; }

private:
    static std::string join_path(const std::string& dir, const std::string& name);
    static bool file_exists(const std::string& path);
    static bool read_text_file(const std::string& path, std::string& out);

    bool load_session(const std::string& path, std::unique_ptr<Ort::Session>& session, std::string& error);
    bool validate_assets(const VieneuV3OnnxInit& init, std::string& error);
    bool load_voices(const std::string& voices_path, std::string& error);

    std::shared_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> prefill_session_;
    std::unique_ptr<Ort::Session> decode_session_;
    std::unique_ptr<Ort::Session> acoustic_session_;
    std::unique_ptr<Ort::Session> codec_decode_session_;
    std::string codec_encode_path_;
    std::string voices_json_;
    bool initialized_ = false;
};

#endif // VIENEU_V3_ONNX_H
