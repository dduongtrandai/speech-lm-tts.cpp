#include "profiles/vieneu.h"

#include <iostream>
#include <string>

static int require_contains(const std::string& value, const std::string& expected, const char* message) {
    if (value.find(expected) != std::string::npos) {
        return 0;
    }
    std::cerr << message << "\nExpected fragment: " << expected << "\nActual: " << value << "\n";
    return 1;
}

static int require_not_contains(const std::string& value, const std::string& rejected, const char* message) {
    if (value.find(rejected) == std::string::npos) {
        return 0;
    }
    std::cerr << message << "\nRejected fragment: " << rejected << "\nActual: " << value << "\n";
    return 1;
}

int main() {
    const std::string sentence = VieneuProfile::phonemize(
        u8"Tôi thích chơi cờ vua vào thời gian rảnh rỗi");

    int failed = 0;
    failed += require_contains(sentence, u8"vˈuə", "ASCII Vietnamese syllable 'vua' bypassed Vietnamese G2P.");
    failed += require_not_contains(sentence, u8"ˈvua", "ASCII Vietnamese syllable was incorrectly retained as English text.");

    const std::string codas = VieneuProfile::phonemize(u8"miếng loan");
    failed += require_contains(codas, u8"iɛɜŋ", "UTF-8 rime 'iếng' lost its coda.");
    failed += require_contains(codas, u8"waːn", "ASCII rime 'oan' lost its coda.");

    return failed == 0 ? 0 : 1;
}
