#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <stddef.h>

/**
 * @brief Duplicates a string safely.
 * 
 * @param s The string to duplicate.
 * @return char* A pointer to the newly allocated string, or NULL if allocation fails.
 */
char* safe_strdup(const char* s);

/**
 * @brief Checks if a string starts with a given prefix.
 * 
 * @param str The string to check.
 * @param pre The prefix to look for.
 * @return int 1 if str starts with pre, 0 otherwise.
 */
int str_starts_with(const char *str, const char *pre);

#endif
