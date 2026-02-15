#include "string_utils.h"
#include <stdlib.h>
#include <string.h>

char* safe_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* new_s = malloc(len);
    if (new_s) {
        memcpy(new_s, s, len);
    }
    return new_s;
}

int str_starts_with(const char *str, const char *pre) {
    size_t lenpre = strlen(pre);
    size_t lenstr = strlen(str);
    return lenstr < lenpre ? 0 : memcmp(pre, str, lenpre) == 0;
}
