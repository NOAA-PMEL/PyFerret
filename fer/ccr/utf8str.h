#include <Python.h> /* make sure Python.h is first */

void create_utf8_str_(const int *codepoint, char *utf8str, int *utf8strlen);
void text_to_utf8_(const char *text, const int *textlen, char *utf8str, int *utf8strlen);

