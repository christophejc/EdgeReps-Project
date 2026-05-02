/* Edge Impulse Arduino examples
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// These sketches are tested with 2.0.4 ESP32 Arduino Core
// https://github.com/espressif/arduino-esp32/releases/tag/2.0.4

/* Includes ---------------------------------------------------------------- */
#include <e6908project_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
#include "esp_camera.h"

// ── CAMERA MODEL ──────────────────────────────────────────────────────────────
#define CAMERA_MODEL_XIAO_ESP32S3
#if defined(CAMERA_MODEL_XIAO_ESP32S3)
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    10
#define SIOD_GPIO_NUM    40
#define SIOC_GPIO_NUM    39
#define Y9_GPIO_NUM      48
#define Y8_GPIO_NUM      11
#define Y7_GPIO_NUM      12
#define Y6_GPIO_NUM      14
#define Y5_GPIO_NUM      16
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM      17
#define Y2_GPIO_NUM      15
#define VSYNC_GPIO_NUM   38
#define HREF_GPIO_NUM    47
#define PCLK_GPIO_NUM    13
#endif
// ─────────────────────────────────────────────────────────────────────────────

#define EI_CAMERA_RAW_FRAME_BUFFER_COLS  320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS  240
#define EI_CAMERA_FRAME_BYTE_SIZE        3

// ── TRIGGER SETTINGS ──────────────────────────────────────────────────────────
#define CONFIDENCE_THRESHOLD  0.85
#define CONFIRM_THRESHOLD     5
// ─────────────────────────────────────────────────────────────────────────────

static bool debug_nn       = false;
static bool is_initialised = false;
uint8_t *snapshot_buf;

// ── DEBOUNCE STATE ────────────────────────────────────────────────────────────
String last_label      = "neither";
int    confirm_count   = 0;
String current_trigger = "neither";
// ─────────────────────────────────────────────────────────────────────────────

// ── METRICS STATE ─────────────────────────────────────────────────────────────
unsigned long last_fps_time  = 0;
int           frame_count    = 0;
float         avg_dsp        = 0;
float         avg_class      = 0;
int           metric_samples = 0;
// ─────────────────────────────────────────────────────────────────────────────

static camera_config_t camera_config = {
    .pin_pwdn      = PWDN_GPIO_NUM,
    .pin_reset     = RESET_GPIO_NUM,
    .pin_xclk      = XCLK_GPIO_NUM,
    .pin_sscb_sda  = SIOD_GPIO_NUM,
    .pin_sscb_scl  = SIOC_GPIO_NUM,
    .pin_d7        = Y9_GPIO_NUM,
    .pin_d6        = Y8_GPIO_NUM,
    .pin_d5        = Y7_GPIO_NUM,
    .pin_d4        = Y6_GPIO_NUM,
    .pin_d3        = Y5_GPIO_NUM,
    .pin_d2        = Y4_GPIO_NUM,
    .pin_d1        = Y3_GPIO_NUM,
    .pin_d0        = Y2_GPIO_NUM,
    .pin_vsync     = VSYNC_GPIO_NUM,
    .pin_href      = HREF_GPIO_NUM,
    .pin_pclk      = PCLK_GPIO_NUM,
    .xclk_freq_hz  = 20000000,
    .ledc_timer    = LEDC_TIMER_0,
    .ledc_channel  = LEDC_CHANNEL_0,
    .pixel_format  = PIXFORMAT_JPEG,
    .frame_size    = FRAMESIZE_QVGA,
    .jpeg_quality  = 12,
    .fb_count      = 1,
    .fb_location   = CAMERA_FB_IN_PSRAM,
    .grab_mode     = CAMERA_GRAB_WHEN_EMPTY,
};

bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);

void setup() {
    Serial.begin(115200);
    while (!Serial);

    Serial.println("=== Edge Impulse Inferencing Demo ===");
    Serial.println("Model: fitness-ai-nolunge");
    Serial.printf("Confidence threshold: %.2f\n", CONFIDENCE_THRESHOLD);
    Serial.printf("Confirm frames: %d\n", CONFIRM_THRESHOLD);

    // print memory on boot
    Serial.printf("Free heap on boot:  %d bytes\n", ESP.getFreeHeap());
    Serial.printf("Free PSRAM on boot: %d bytes\n", ESP.getFreePsram());

    if (ei_camera_init() == false) {
        ei_printf("Failed to initialize Camera!\r\n");
    } else {
        ei_printf("Camera initialized\r\n");
    }

    // print memory after camera init
    Serial.printf("Free heap after camera init:  %d bytes\n", ESP.getFreeHeap());
    Serial.printf("Free PSRAM after camera init: %d bytes\n", ESP.getFreePsram());

    last_fps_time = millis();

    ei_printf("\nStarting continuous inference in 2 seconds...\n");
    ei_sleep(2000);
}

void loop() {
    if (ei_sleep(5) != EI_IMPULSE_OK) {
        return;
    }

    snapshot_buf = (uint8_t*)malloc(
        EI_CAMERA_RAW_FRAME_BUFFER_COLS *
        EI_CAMERA_RAW_FRAME_BUFFER_ROWS *
        EI_CAMERA_FRAME_BYTE_SIZE
    );

    if (snapshot_buf == nullptr) {
        ei_printf("ERR: Failed to allocate snapshot buffer!\n");
        return;
    }

    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data     = &ei_camera_get_data;

    if (ei_camera_capture(
            (size_t)EI_CLASSIFIER_INPUT_WIDTH,
            (size_t)EI_CLASSIFIER_INPUT_HEIGHT,
            snapshot_buf) == false) {
        ei_printf("Failed to capture image\r\n");
        free(snapshot_buf);
        return;
    }

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        free(snapshot_buf);
        return;
    }

    // ── LATENCY METRICS ───────────────────────────────────────────────────────
    int total_ms = result.timing.dsp + result.timing.classification;
    metric_samples++;
    avg_dsp   = avg_dsp   + (result.timing.dsp   - avg_dsp)   / metric_samples;
    avg_class = avg_class + (result.timing.classification - avg_class) / metric_samples;

    ei_printf("Latency - DSP: %d ms | Classification: %d ms | Total: %d ms\n",
        result.timing.dsp,
        result.timing.classification,
        total_ms);
    // ─────────────────────────────────────────────────────────────────────────

    // ── FPS COUNTER ───────────────────────────────────────────────────────────
    frame_count++;
    unsigned long now = millis();
    if (now - last_fps_time >= 5000) {
        float fps = frame_count / 5.0;
        ei_printf("\n========== METRICS REPORT ==========\n");
        ei_printf("FPS:                  %.2f\n", fps);
        ei_printf("Avg DSP latency:      %.1f ms\n", avg_dsp);
        ei_printf("Avg classify latency: %.1f ms\n", avg_class);
        ei_printf("Avg total latency:    %.1f ms\n", avg_dsp + avg_class);
        ei_printf("Free heap:            %d bytes\n", ESP.getFreeHeap());
        ei_printf("Free PSRAM:           %d bytes\n", ESP.getFreePsram());
        ei_printf("Estimated power:      ~300 mW\n");
        ei_printf("=====================================\n\n");
        frame_count    = 0;
        last_fps_time  = now;
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ── PREDICTIONS ───────────────────────────────────────────────────────────
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: %.5f\r\n",
            ei_classifier_inferencing_categories[i],
            result.classification[i].value);
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ── FIND HIGHEST CONFIDENCE LABEL ────────────────────────────────────────
    float  max_val   = 0;
    String max_label = "neither";

    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > max_val) {
            max_val   = result.classification[i].value;
            max_label = String(ei_classifier_inferencing_categories[i]);
        }
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ── DEBOUNCE + CONSISTENCY CHECK ─────────────────────────────────────────
    if (max_val >= CONFIDENCE_THRESHOLD && max_label != "neither") {

        if (max_label == last_label) {
            confirm_count++;
        } else {
            confirm_count = 1;
            last_label    = max_label;
        }

        if (confirm_count >= CONFIRM_THRESHOLD) {
            if (max_label != current_trigger) {
                current_trigger = max_label;

                if (max_label == "squat_position") {
                    Serial.println("TRIGGER:SQUAT");
                } else if (max_label == "pushup_position") {
                    Serial.println("TRIGGER:PUSHUP");
                }

                ei_printf(">> Trigger sent: %s\n", max_label.c_str());
            }
            confirm_count = 0;
        }

    } else {
        if (current_trigger != "neither") {
            current_trigger = "neither";
            Serial.println("TRIGGER:NONE");
            ei_printf(">> Trigger sent: NONE\n");
        }
        last_label    = "neither";
        confirm_count = 0;
    }
    // ─────────────────────────────────────────────────────────────────────────

    free(snapshot_buf);
}

bool ei_camera_init(void) {
    if (is_initialised) return true;

    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }

    sensor_t* s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, 0);
    }

    is_initialised = true;
    return true;
}

void ei_camera_deinit(void) {
    esp_err_t err = esp_camera_deinit();
    if (err != ESP_OK) {
        ei_printf("Camera deinit failed\n");
        return;
    }
    is_initialised = false;
}

bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
    bool do_resize = false;

    if (!is_initialised) {
        ei_printf("ERR: Camera is not initialized\r\n");
        return false;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ei_printf("Camera capture failed\n");
        return false;
    }

    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);
    esp_camera_fb_return(fb);

    if (!converted) {
        ei_printf("Conversion failed\n");
        return false;
    }

    if ((img_width  != EI_CAMERA_RAW_FRAME_BUFFER_COLS) ||
        (img_height != EI_CAMERA_RAW_FRAME_BUFFER_ROWS)) {
        do_resize = true;
    }

    if (do_resize) {
        ei::image::processing::crop_and_interpolate_rgb888(
            out_buf,
            EI_CAMERA_RAW_FRAME_BUFFER_COLS,
            EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
            out_buf,
            img_width,
            img_height);
    }

    return true;
}

static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
    size_t pixel_ix    = offset * 3;
    size_t pixels_left = length;
    size_t out_ptr_ix  = 0;

    while (pixels_left != 0) {
        out_ptr[out_ptr_ix] = (snapshot_buf[pixel_ix + 2] << 16) +
                               (snapshot_buf[pixel_ix + 1] << 8) +
                               snapshot_buf[pixel_ix];
        out_ptr_ix++;
        pixel_ix += 3;
        pixels_left--;
    }
    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor"
#endif