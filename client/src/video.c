#include "video.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#define CLEAR(x) memset(&(x), 0, sizeof(x))

static int fd = -1;
static buffer_t *buffers;
static unsigned int n_buffers;
static struct v4l2_buffer buf; // Current buffer info

static int xioctl(int fh, int request, void *arg) {
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

int init_camera(const char *device_name, int width, int height) {
    struct v4l2_capability cap;
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;

    // 1. Open Device
    fd = open(device_name, O_RDWR | O_NONBLOCK, 0);
    if (fd == -1) {
        perror("Opening video device");
        return -1;
    }

    // 2. Check Capabilities
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        perror("Querying Capabilities");
        return -1;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is no video capture device\n", device_name);
        return -1;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s does not support streaming i/o\n", device_name);
        return -1;
    }

    // 3. Set Format (MJPEG for efficiency)
    CLEAR(fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG; // Compress on device!
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("Setting Pixel Format");
        return -1;
    }

    // 4. Request Buffers
    CLEAR(req);
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        perror("Requesting Buffers");
        return -1;
    }

    buffers = calloc(req.count, sizeof(*buffers));

    // 5. Map Buffers
    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buffer;
        CLEAR(buffer);

        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = n_buffers;

        if (xioctl(fd, VIDIOC_QUERYBUF, &buffer) == -1) {
            perror("Querying Buffer");
            return -1;
        }

        buffers[n_buffers].length = buffer.length;
        buffers[n_buffers].start = mmap(NULL, buffer.length,
                                      PROT_READ | PROT_WRITE, MAP_SHARED,
                                      fd, buffer.m.offset);

        if (MAP_FAILED == buffers[n_buffers].start) {
            perror("mmap");
            return -1;
        }
    }

    // 6. Queue Buffers
    for (unsigned int i = 0; i < n_buffers; ++i) {
        struct v4l2_buffer buffer;
        CLEAR(buffer);
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = i;

        if (xioctl(fd, VIDIOC_QBUF, &buffer) == -1) {
            perror("Queue Buffer");
            return -1;
        }
    }

    // 7. Start Streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("Start Streaming");
        return -1;
    }

    return 0;
}

int capture_frame(void **data, size_t *length) {
    fd_set fds;
    struct timeval tv;
    int r;

    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    tv.tv_sec = 2;
    tv.tv_usec = 0;

    r = select(fd + 1, &fds, NULL, NULL, &tv);
    if (-1 == r) {
        perror("select");
        return -1;
    }
    if (0 == r) {
        fprintf(stderr, "select timeout\n");
        return -1;
    }

    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        perror("Retrieving Frame (DQBUF)");
        return -1;
    }

    *data = buffers[buf.index].start;
    *length = buf.bytesused;

    return 0;
}

void release_frame(void) {
    if (xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
        perror("Releasing Frame (QBUF)");
    }
}

void close_camera(void) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd, VIDIOC_STREAMOFF, &type);

    for (unsigned int i = 0; i < n_buffers; ++i)
        munmap(buffers[i].start, buffers[i].length);
    
    close(fd);
    free(buffers);
}
