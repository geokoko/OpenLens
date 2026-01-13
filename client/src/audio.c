#include "audio.h"
#include <alsa/asoundlib.h>
#include <stdio.h>

static snd_pcm_t *pcm_handle;
static const char *device = "default";

int init_audio(const char *device_name, unsigned int rate) {
    int err;
    snd_pcm_hw_params_t *params;

    if (device_name) device = device_name;

    if ((err = snd_pcm_open(&pcm_handle, device, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "Cannot open audio device %s (%s)\n", device, snd_strerror(err));
        return -1;
    }

    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(pcm_handle, params);

    // Interleaved mode
    snd_pcm_hw_params_set_access(pcm_handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    // 16-bit Little Endian
    snd_pcm_hw_params_set_format(pcm_handle, params, SND_PCM_FORMAT_S16_LE);
    // Mono
    snd_pcm_hw_params_set_channels(pcm_handle, params, 1);
    // Sampling rate (approx)
    snd_pcm_hw_params_set_rate_near(pcm_handle, params, &rate, 0);

    if ((err = snd_pcm_hw_params(pcm_handle, params)) < 0) {
        fprintf(stderr, "Cannot set audio parameters (%s)\n", snd_strerror(err));
        return -1;
    }

    if ((err = snd_pcm_prepare(pcm_handle)) < 0) {
        fprintf(stderr, "Cannot prepare audio interface for use (%s)\n", snd_strerror(err));
        return -1;
    }

    return 0;
}

int capture_audio(void *buffer, size_t frames) {
    int err;
    if ((err = snd_pcm_readi(pcm_handle, buffer, frames)) != frames) {
        fprintf(stderr, "Read from audio interface failed (%s)\n", snd_strerror(err));
        if (err == -EPIPE) {
            // Overrun
            snd_pcm_prepare(pcm_handle);
        }
        return -1;
    }
    return 0;
}

void close_audio(void) {
    snd_pcm_close(pcm_handle);
}
