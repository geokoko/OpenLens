#ifndef AUDIO_H
#define AUDIO_H

#include <stddef.h>

int init_audio(const char *device_name, unsigned int rate);
int capture_audio(void *buffer, size_t frames);
void close_audio(void);

#endif
