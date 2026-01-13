#ifndef NETWORK_H
#define NETWORK_H

#include <stddef.h>
#include <stdint.h>

int connect_to_server(const char *host, int port, const char *path);
int send_video_frame(const void *data, size_t len);
int send_audio_chunk(const void *data, size_t len);
void close_connection(void);

#endif
