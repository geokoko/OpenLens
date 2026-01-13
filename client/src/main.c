#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include "video.h"
#include "network.h"

static volatile int running = 1;

void handle_sigint(int sig) {
    running = 0;
}

int main(int argc, char **argv) {
    const char *host = "127.0.0.1";
    int port = 8000;
    const char *path = "/ws/generate-response";

    if (argc > 1) host = argv[1];
    if (argc > 2) port = atoi(argv[2]);

    signal(SIGINT, handle_sigint);

    printf("AR Glasses Client Starting...\n");
    printf("Target: %s:%d%s\n", host, port, path);

    // 1. Connect Network
    if (connect_to_server(host, port, path) != 0) {
        return 1;
    }

    // 2. Init Camera (640x480 MJPEG)
    // Pi Camera usually supports this well.
    if (init_camera("/dev/video0", 640, 480) != 0) {
        close_connection();
        return 1;
    }

    printf("Streaming started. Press Ctrl+C to stop.\n");

    void *frame_data;
    size_t frame_len;
    int frames = 0;

    while (running) {
        // 3. Capture
        if (capture_frame(&frame_data, &frame_len) == 0) {
            // 4. Send
            if (send_video_frame(frame_data, frame_len) != 0) {
                fprintf(stderr, "Failed to send frame\n");
                break;
            }
            
            // 5. Release
            release_frame();
            
            frames++;
            if (frames % 30 == 0) {
                printf("Sent 30 frames...\n");
            }
        } else {
            // Capture failed (timeout?), retry
            usleep(10000);
        }
    }

    printf("\nShutting down...\n");
    close_camera();
    close_connection();
    
    return 0;
}
