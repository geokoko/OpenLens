#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>

static int sock = -1;

// Helper to create a random WebSocket key
static void generate_ws_key(char *key) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (int i = 0; i < 22; i++) {
        key[i] = charset[rand() % (sizeof(charset) - 1)];
    }
    key[22] = '=';
    key[23] = '=';
    key[24] = '\0';
}

int connect_to_server(const char *host, int port, const char *path) {
    struct sockaddr_in server_addr;
    struct hostent *he;
    
    // 1. Resolve Host
    if ((he = gethostbyname(host)) == NULL) {
        perror("gethostbyname");
        return -1;
    }

    // 2. Create Socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        return -1;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr = *((struct in_addr *)he->h_addr);
    memset(&(server_addr.sin_zero), 0, 8);

    // 3. Connect
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(struct sockaddr)) == -1) {
        perror("connect");
        return -1;
    }

    printf("Connected to %s:%d\n", host, port);

    // 4. WebSocket Handshake
    char key[25];
    generate_ws_key(key);
    
    char request[1024];
    snprintf(request, sizeof(request), 
             "GET %s HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Upgrade: websocket\r\n"
             "Connection: Upgrade\r\n"
             "Sec-WebSocket-Key: %s\r\n"
             "Sec-WebSocket-Version: 13\r\n\r\n",
             path, host, port, key);

    if (send(sock, request, strlen(request), 0) == -1) {
        perror("send handshake");
        return -1;
    }

    // 5. Wait for Response (Simple Check)
    char buffer[1024];
    int n = recv(sock, buffer, sizeof(buffer) - 1, 0);
    if (n <= 0) {
        perror("recv handshake");
        return -1;
    }
    buffer[n] = '\0';

    if (strstr(buffer, "101 Switching Protocols") == NULL) {
        fprintf(stderr, "WebSocket Handshake failed:\n%s\n", buffer);
        return -1;
    }

    printf("WebSocket Handshake Successful!\n");
    return 0;
}

// Very simple WebSocket Frame Encoder (Client -> Server, Masking Required)
static int send_frame(const void *data, size_t len, int opcode) {
    uint8_t header[14]; // Max header size
    size_t header_len = 0;

    // Byte 0: FIN + Opcode
    header[0] = 0x80 | (opcode & 0x0F);

    // Byte 1: Mask bit (Must be set for client) + Payload len
    if (len <= 125) {
        header[1] = 0x80 | len;
        header_len = 2;
    } else if (len <= 65535) {
        header[1] = 0x80 | 126;
        header[2] = (len >> 8) & 0xFF;
        header[3] = len & 0xFF;
        header_len = 4;
    } else {
        header[1] = 0x80 | 127;
        // 64-bit length... assuming we don't send > 4GB frames
        // Just filling lower 32 bits for simplicity here
        header[2] = 0; header[3] = 0; header[4] = 0; header[5] = 0;
        header[6] = (len >> 24) & 0xFF;
        header[7] = (len >> 16) & 0xFF;
        header[8] = (len >> 8) & 0xFF;
        header[9] = len & 0xFF;
        header_len = 10;
    }

    // Masking Key (4 bytes)
    uint8_t mask[4];
    for(int i=0; i<4; i++) mask[i] = rand() % 255;
    
    memcpy(header + header_len, mask, 4);
    header_len += 4;

    // Send Header
    if (send(sock, header, header_len, 0) == -1) return -1;

    // Send Masked Data
    // Optimization: Create a static scratch buffer or loop in 4KB chunks.
    
    uint8_t chunk[4096];
    size_t sent = 0;
    const uint8_t *p = (const uint8_t*)data;
    
    while (sent < len) {
        size_t to_send = (len - sent) > sizeof(chunk) ? sizeof(chunk) : (len - sent);
        for (size_t i = 0; i < to_send; i++) {
            chunk[i] = p[sent + i] ^ mask[(sent + i) % 4];
        }
        if (send(sock, chunk, to_send, 0) == -1) return -1;
        sent += to_send;
    }

    return 0;
}

int send_video_frame(const void *data, size_t len) {
    // Opcode 2 = Binary
    // We might want to prepend a "type" byte (e.g. 0x01 for video, 0x02 for audio)

    return send_frame(data, len, 0x02); 
}

int send_audio_chunk(const void *data, size_t len) {
    // For now, let's assume we just stream Video.

    return 0; 
}

void close_connection(void) {
    if (sock != -1) close(sock);
}
