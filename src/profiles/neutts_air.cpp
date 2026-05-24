#include "neutts_air.h"
#include <regex>
#include <iostream>

std::string NeuTtsAirProfile::format_prompt(const std::string& phonemes) {
    return "<|TEXT_PROMPT_START|>" + phonemes + "<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>";
}

std::vector<int32_t> NeuTtsAirProfile::extract_speech_ids(const std::string& generated_text) {
    std::vector<int32_t> ids;
    std::regex re("<\\|(?:speech|s)_(\\d+)\\|>");
    auto start = std::sregex_iterator(generated_text.begin(), generated_text.end(), re);
    auto end = std::sregex_iterator();
    for (auto i = start; i != end; ++i) {
        ids.push_back(std::stoi((*i)[1].str()));
    }
    return ids;
}

bool NeuTtsAirProfile::synthesize(
    LlamaBackend& llama,
    NeuCodecOnnx& decoder,
    const std::string& text,
    float temperature,
    int top_k,
    int max_tokens,
    bool skip_phonemize,
    std::vector<float>& out_audio)
{
    out_audio.clear();
    
    // In this profile, if skip_phonemize is false, we print a warning and treat text as phonemes.
    // Real deployment should use pre-phonemized input for maximum quality on CPU.
    std::string phonemes = text;
    if (!skip_phonemize) {
        std::cerr << "[NeuTTS Air] Warning: C++ runtime G2P for English is not fully implemented. Treating input text as phonemes." << std::endl;
    }

    std::string prompt = format_prompt(phonemes);
    auto prompt_tokens = llama.tokenize(prompt, true);

    if (!llama.decode(prompt_tokens, 0, true)) {
        return false;
    }

    std::string generated_text = "";
    llama_token curr_token = 0;
    int n_tokens = 0;

    while (n_tokens < max_tokens) {
        curr_token = llama.sample(temperature, top_k);
        if (llama.is_eog(curr_token)) {
            break;
        }

        std::string piece = llama.token_to_piece(curr_token);
        generated_text += piece;

        if (generated_text.find("<|SPEECH_GENERATION_END|>") != std::string::npos) {
            break;
        }

        std::vector<llama_token> next_tokens = { curr_token };
        if (!llama.decode(next_tokens, 0, false)) {
            break;
        }

        n_tokens++;
    }

    auto speech_ids = extract_speech_ids(generated_text);
    if (speech_ids.empty()) {
        std::cerr << "[NeuTTS Air] No speech tokens generated!" << std::endl;
        return false;
    }

    std::vector<float> chunk_audio;
    if (!decoder.decode_neucodec(speech_ids, chunk_audio)) {
        return false;
    }

    out_audio = std::move(chunk_audio);
    return true;
}
