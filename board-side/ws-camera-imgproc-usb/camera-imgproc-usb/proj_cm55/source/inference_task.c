/*****************************************************************************
* \file inference_task.c
*
* \brief CM55 inference task: orchestrates camera frame capture, optional
*        on-device image-processing stage, neural-network inference, and
*        publication of detection / FPS / preview events to the host.
*****************************************************************************/
#include "inference_task.h"
#include "lcd_task.h"
#include "model.h"
#include "imgproc/imgproc.h"
#include "comm/comm_task.h"
#include "comm/comm_proto.h"
#include "pipeline_config.h"
#include "ifx_time_utils.h"

#include "FreeRTOS.h"
#include "task.h"
#include "cyabs_rtos.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Externals from main.c / lcd_task.c */
extern cy_semaphore_t model_semaphore;
extern cy_semaphore_t usb_semaphore;
extern uint8_t        bgr888_uint8[];
extern uint8_t        _device_connected;

/* Storage shared with lcd_task.c (it expects these symbols). */
prediction_od_t  prediction;
volatile float   inference_time;

/* Detection threshold from the original CE design. */
#define AI_CONF_THRESHOLD       (0.30f)

/* Class index translation: model 0=Scissors,1=Paper,2=Rock -> dataset 0=Rock,1=Paper,2=Scissors */
static const uint8_t  k_model_to_dataset[NUM_CLASSES] = { 2, 1, 0 };
static const char    *k_class_names[NUM_CLASSES]      = { "Rock", "Paper", "Scissors" };

/* Model output buffer: 8 attributes x 5 detections (column-major). */
#define MODEL_DET_FIELDS    8
static float s_model_out[MODEL_DET_FIELDS * MAX_PREDICTIONS];

/* 80x60 grayscale preview scratch in SoCMEM. */
static __attribute__((section(".cy_socmem_data"), aligned(16)))
       uint8_t s_preview[COMM_PREVIEW_W * COMM_PREVIEW_H];

/*-------------------------------------------------------------------
 * Postprocessing: parse model output (column-major 8x5) into top-N.
 *------------------------------------------------------------------*/
static void parse_model_output(const float *out, prediction_od_t *p)
{
    p->count = 0;
    for (int d = 0; d < MAX_PREDICTIONS; d++) {
        float flag = out[7 * MAX_PREDICTIONS + d];
        if (flag <= 0.0f) continue;

        /* find best class score among indices 4,5,6 */
        float scores[3] = {
            out[4 * MAX_PREDICTIONS + d],
            out[5 * MAX_PREDICTIONS + d],
            out[6 * MAX_PREDICTIONS + d],
        };
        int   best = 0;
        float best_s = scores[0];
        if (scores[1] > best_s) { best = 1; best_s = scores[1]; }
        if (scores[2] > best_s) { best = 2; best_s = scores[2]; }
        if (best_s < AI_CONF_THRESHOLD) continue;

        float cx = out[0 * MAX_PREDICTIONS + d];
        float cy = out[1 * MAX_PREDICTIONS + d];
        float w  = out[2 * MAX_PREDICTIONS + d];
        float h  = out[3 * MAX_PREDICTIONS + d];

        int idx = p->count;
        p->bbox_int16[idx * 4 + 0] = (int16_t)(cx - HALF(w) + RND_F2I_FACTOR);
        p->bbox_int16[idx * 4 + 1] = (int16_t)(cy - HALF(h) + RND_F2I_FACTOR);
        p->bbox_int16[idx * 4 + 2] = (int16_t)(cx + HALF(w) + RND_F2I_FACTOR);
        p->bbox_int16[idx * 4 + 3] = (int16_t)(cy + HALF(h) + RND_F2I_FACTOR);
        p->conf[idx]               = best_s;
        p->class_id[idx]           = k_model_to_dataset[best];
        snprintf(p->class_string[idx], MAX_CLASS_LEN, "%s",
                 k_class_names[ k_model_to_dataset[best] ]);
        p->count++;
        if (p->count >= MAX_PREDICTIONS) break;
    }
}

int8_t get_best_class(const float *cls, size_t size, float *max_cls_val)
{
    int8_t best = -1;
    float  best_v = 0.0f;
    for (size_t i = 0; i < size; i++) {
        if (cls[i] > best_v) { best_v = cls[i]; best = (int8_t)i; }
    }
    if (max_cls_val) *max_cls_val = best_v;
    return best;
}

/*-------------------------------------------------------------------
 * Build 80x60 grayscale preview from the 320x240 RGB888 model input
 * (top-left 320x240 region; the rest of the 320x320 buffer is padding).
 *------------------------------------------------------------------*/
static void build_preview(const uint8_t *rgb)
{
    for (int py = 0; py < COMM_PREVIEW_H; py++) {
        int sy = (py * CAMERA_HEIGHT) / COMM_PREVIEW_H;
        for (int px = 0; px < COMM_PREVIEW_W; px++) {
            int sx = (px * CAMERA_WIDTH) / COMM_PREVIEW_W;
            const uint8_t *p = &rgb[(sy * IMAGE_WIDTH + sx) * 3];
            s_preview[py * COMM_PREVIEW_W + px] =
                (uint8_t)((p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8);
        }
    }
}

/*-------------------------------------------------------------------
 * Bench-mode helper: copy host-uploaded image into model input buffer.
 *------------------------------------------------------------------*/
static bool load_bench_image(uint8_t *dst)
{
    uint16_t w, h;
    const uint8_t *img = comm_bench_image(&w, &h);
    if (!img) return false;
    if (w != CAMERA_WIDTH || h != CAMERA_HEIGHT) return false;
    /* Top-aligned: image goes into rows 0..239, rest stays zero. */
    memcpy(dst, img, (size_t)w * h * 3);
    memset(dst + (size_t)w * h * 3, 0,
           (size_t)w * (IMAGE_HEIGHT - h) * 3);
    return true;
}

/*-------------------------------------------------------------------
 * Synthetic test-pattern generator: when no USB host camera is plugged
 * into the board, the LCD task never signals usb_semaphore, so the
 * inference loop would otherwise wedge forever.  We fall back to a
 * generated animated pattern so the host GUI still gets a live preview
 * and the user can visually verify their selected algorithm (sobel,
 * kirsch, etc.) is running on the device.
 *
 * Pattern: animated diagonal gradient + checkerboard + a moving bright
 * bar -- gives every algorithm something interesting to operate on.
 *------------------------------------------------------------------*/
static void synth_fill_frame(uint8_t *rgb, uint32_t tick)
{
    /* Top CAMERA_WIDTH x CAMERA_HEIGHT region holds the actual frame; the
     * rest of the IMAGE_HEIGHT - CAMERA_HEIGHT rows are zero-padded. */
    uint32_t bar_x = (tick * 4u) % CAMERA_WIDTH;
    for (int y = 0; y < CAMERA_HEIGHT; y++) {
        uint8_t *row = &rgb[(size_t)y * IMAGE_WIDTH * 3];
        for (int x = 0; x < CAMERA_WIDTH; x++) {
            uint8_t r = (uint8_t)((x + tick) & 0xFF);
            uint8_t g = (uint8_t)((y + (tick >> 1)) & 0xFF);
            uint8_t b = (uint8_t)((x + y) & 0xFF);
            /* checkerboard accent every 32 px so edge filters catch it */
            if (((x >> 5) ^ (y >> 5)) & 1u) {
                r = (uint8_t)(r ^ 0x40u);
                g = (uint8_t)(g ^ 0x40u);
                b = (uint8_t)(b ^ 0x40u);
            }
            /* moving white bar */
            int dx = (int)x - (int)bar_x;
            if (dx < 0) dx = -dx;
            if (dx < 6) { r = g = b = 0xFF; }
            row[x * 3 + 0] = r;
            row[x * 3 + 1] = g;
            row[x * 3 + 2] = b;
        }
    }
    /* Zero the bottom padding so subsequent algos don't see stale data. */
    if (IMAGE_HEIGHT > CAMERA_HEIGHT) {
        memset(rgb + (size_t)CAMERA_HEIGHT * IMAGE_WIDTH * 3, 0,
               (size_t)(IMAGE_HEIGHT - CAMERA_HEIGHT) * IMAGE_WIDTH * 3);
    }
}

/*-------------------------------------------------------------------
 * Inference task entry. Spawned from main.c.
 *------------------------------------------------------------------*/
void cm55_inference_task(void *arg)
{
    (void)arg;

    if (IMAI_init() != 0) {
        printf("\r\nIMAI_init() failed\r\n");
        CY_ASSERT(0);
    }
    printf("\r\nInference task started\r\n");

    uint32_t synth_tick = 0u;

    for (;;) {
        bool bench = pipeline_get_bench();

        /* Acquire input frame */
        uint8_t *frame;
        if (bench) {
            /* wait for a fresh bench image */
            if (!pipeline_get_bench_ready()) {
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }
            if (!load_bench_image(bgr888_uint8)) {
                pipeline_set_bench_ready(false);
                continue;
            }
            frame = bgr888_uint8;
        } else {
            /* Normal camera path with a fallback for "no USB camera
             * attached".  Wait briefly for a real frame; if none arrives
             * within the timeout, synthesize one so the algo + preview
             * pipeline still produces output.  This is what makes the
             * GUI show live video on a board without a webcam. */
            cy_rslt_t r = cy_rtos_semaphore_get(&usb_semaphore,
                                                pdMS_TO_TICKS(80));
            if (r == CY_RSLT_SUCCESS && _device_connected) {
                frame = draw();
            } else {
                /* Synthesize a frame.  Throttle to ~15 fps so we don't
                 * saturate the UART when streaming is on. */
                synth_fill_frame(bgr888_uint8, synth_tick);
                synth_tick += 4u;
                frame = bgr888_uint8;
                vTaskDelay(pdMS_TO_TICKS(10));
            }
        }

        /* On-device image-processing stage. */
        uint8_t algo = pipeline_get_algo();
        uint32_t algo_t0 = (uint32_t)(ifx_time_get_ms_f() * 1000.0f);
        imgproc_apply(algo, frame, IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_HEIGHT);
        uint32_t algo_t1 = (uint32_t)(ifx_time_get_ms_f() * 1000.0f);

        /* Inference */
        uint32_t inf_t0 = algo_t1;
        IMAI_compute(frame, s_model_out);
        uint32_t inf_t1 = (uint32_t)(ifx_time_get_ms_f() * 1000.0f);
        inference_time = (float)(inf_t1 - inf_t0) / 1000.0f;

        parse_model_output(s_model_out, &prediction);

        /* Publish telemetry */
        uint32_t algo_us  = algo_t1 - algo_t0;
        uint32_t infer_us = inf_t1  - inf_t0;
        pipeline_record_timing(algo_us, infer_us);

        /* Detection event */
        if (prediction.count > 0) {
            uint8_t  cids[MAX_PREDICTIONS];
            uint8_t  conf[MAX_PREDICTIONS];
            int16_t  bbox[MAX_PREDICTIONS * 4];
            for (int i = 0; i < prediction.count; i++) {
                cids[i] = prediction.class_id[i];
                int c = (int)(prediction.conf[i] * 100.0f);
                if (c > 255) c = 255;
                if (c < 0) c = 0;
                conf[i] = (uint8_t)c;
                int16_t x = prediction.bbox_int16[i*4+0];
                int16_t y = prediction.bbox_int16[i*4+1];
                int16_t x2= prediction.bbox_int16[i*4+2];
                int16_t y2= prediction.bbox_int16[i*4+3];
                bbox[i*4+0] = x;
                bbox[i*4+1] = y;
                bbox[i*4+2] = (int16_t)(x2 - x);
                bbox[i*4+3] = (int16_t)(y2 - y);
            }
            comm_send_detection((uint8_t)prediction.count, cids, conf, bbox);
        } else {
            comm_send_detection(0, NULL, NULL, NULL);
        }

        /* FPS event (cheap, send every frame) */
        comm_send_fps(pipeline_get_fps_x100(), algo_us, infer_us);

        /* Optional preview stream.  IMPORTANT: emit BEFORE bench_result so
         * the host receives the full preview frame while it's still
         * waiting on the BENCH_RESULT ACK.  If we sent the preview after
         * the bench result, the host would proceed to the next
         * BENCH_BEGIN and the in-flight preview chunks would race with
         * the new image upload, producing a corrupted preview frame. */
        if (pipeline_get_stream()) {
            build_preview(frame);
            comm_send_preview_gray80x60(s_preview);
        }

        /* Bench-result event (best class for the whole frame) */
        if (bench) {
            uint8_t best_cls = 0xFF;
            uint8_t best_conf = 0;
            float bv = 0.0f;
            for (int i = 0; i < prediction.count; i++) {
                if (prediction.conf[i] > bv) {
                    bv = prediction.conf[i];
                    best_cls = prediction.class_id[i];
                    int c = (int)(bv * 100.0f);
                    if (c > 255) c = 255;
                    best_conf = (uint8_t)c;
                }
            }
            comm_send_bench_result(best_cls, best_conf, algo_us, infer_us);
            pipeline_set_bench_ready(false);
        }

        /* Notify LCD task only when LCD rendering is enabled. */
        if (pipeline_get_lcd() && !bench) {
            cy_rtos_semaphore_set(&model_semaphore);
        }
    }
}
