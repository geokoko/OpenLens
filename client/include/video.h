#ifndef VIDEO_H
#define VIDEO_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    void *start;
    size_t length;
} buffer_t;

int init_camera(const char *device_name, int width, int height);
int capture_frame(void **data, size_t *length);
void release_frame(void);
void close_camera(void);

#endif
