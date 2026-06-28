#ifndef SPEECHLM_TTS_H
#define SPEECHLM_TTS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef _WIN32
#   if defined(SLM_STATIC)
#       define SLM_API
#   elif defined(SLM_BUILD_DLL)
#       define SLM_API __declspec(dllexport)
#   else
#       define SLM_API __declspec(dllimport)
#   endif
#else
#   define SLM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct slm_init_params {
    int          abi_version;
    const char * model_path;
    const char * encoder_path;
    const char * decoder_path;
    const char * voices_json_path;
    int          n_ctx;
    int          n_threads;
    int          n_gpu_layers;
    bool         flash_attn;
    bool         mlock;
};

struct slm_audio {
    float * samples;
    int     n_samples;
    int     sample_rate;
    int     channels;
};

struct slm_context;

struct slm_tts_params {
    int           abi_version;
    const char *  text;
    const char *  voice_id;
    const float * voice_embedding;
    float         temperature;
    int           top_k;
    int           max_chars;
    int           max_tokens;
    bool          skip_normalize;
    bool          skip_phonemize;
    bool          apply_watermark;
};

struct slm_init_params_v2 {
    int          abi_version;
    const char * profile;
    const char * model_dir;
    const char * onnx_dir;
    const char * codec_dir;
    const char * config_path;
    const char * tokenizer_path;
    const char * voices_json_path;
    int          n_threads;
};

struct slm_tts_params_v2 {
    int          abi_version;
    const char * text;
    const char * voice_id;
    const char * ref_audio_path;
    float        temperature;
    int          top_k;
    float        top_p;
    int          max_new_frames;
    float        repetition_penalty;
    int          max_chars;
    bool         apply_watermark;
};

SLM_API const char * slm_version(void);
SLM_API const char * slm_last_error(void);

SLM_API void slm_audio_free(struct slm_audio * a);

SLM_API void slm_init_default_params(struct slm_init_params * p);
SLM_API struct slm_context * slm_init(const struct slm_init_params * params);
SLM_API void slm_free(struct slm_context * slm);

SLM_API void slm_tts_default_params(struct slm_tts_params * p);
SLM_API int slm_synthesize(struct slm_context * slm, const struct slm_tts_params * params, struct slm_audio * out);

SLM_API void slm_init_v2_default_params(struct slm_init_params_v2 * p);
SLM_API struct slm_context * slm_init_v2(const struct slm_init_params_v2 * params);
SLM_API void slm_tts_v2_default_params(struct slm_tts_params_v2 * p);
SLM_API int slm_synthesize_v2(struct slm_context * slm, const struct slm_tts_params_v2 * params, struct slm_audio * out);

SLM_API int slm_encode_reference(struct slm_context * slm, const char * ref_audio_path, float * out_embedding_128);
SLM_API int slm_list_preset_voices(struct slm_context * slm, char * out_json, int max_len);
SLM_API int slm_set_preset_voice(struct slm_context * slm, const char * voice_id);

#ifdef __cplusplus
}
#endif

#endif // SPEECHLM_TTS_H
