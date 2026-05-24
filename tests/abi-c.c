#include "speechlm_tts.h"
#include <stdio.h>

int main(void) {
    printf("[ABI Test] SpeechLM TTS Public ABI Contract: OK\n");
    printf("[ABI Test] Runtime version: %s\n", slm_version());
    
    // Smoke check structures and functions exist
    struct slm_init_params params;
    slm_init_default_params(&params);
    
    struct slm_tts_params tts_params;
    slm_tts_default_params(&tts_params);
    
    return 0;
}
