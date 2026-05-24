#ifndef NEUTTS_AIR_H
#define NEUTTS_AIR_H

#include <string>
#include <vector>
#include "llm/llama_backend.h"
#include "codecs/neucodec_onnx.h"

class NeuTtsAirProfile {
public:
    static std::string format_prompt(const std::string& phonemes);
    static std::vector<int32_t> extract_speech_ids(const std::string& generated_text);

    static bool synthesize(
        LlamaBackend& llama,
        NeuCodecOnnx& decoder,
        const std::string& text,
        float temperature,
        int top_k,
        int max_tokens,
        bool skip_phonemize,
        std::vector<float>& out_audio
    );
};

#endif // NEUTTS_AIR_H
