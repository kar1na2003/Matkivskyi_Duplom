/*****************************************************************************
* \file pipeline_config.h
*
* \brief Runtime pipeline configuration shared across tasks. All values are
*        read/written through helpers below to keep updates atomic.
*****************************************************************************/
#ifndef _PIPELINE_CONFIG_H_
#define _PIPELINE_CONFIG_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum image-processing algorithm ID (mirrors imgproc.h). */
#define PIPELINE_MAX_ALGO_ID    63u

typedef struct {
    uint8_t  selected_algo;     /* IMGPROC_ALGO_* enum value */
    bool     lcd_enabled;       /* render to on-board LCD */
    bool     stream_enabled;    /* push downsampled preview frames over UART */
    bool     bench_mode;        /* host pushes images instead of camera input */
    bool     bench_image_ready; /* host has uploaded a fresh bench image */
    uint32_t fps_x100;          /* current inference FPS * 100 */
    uint32_t algo_us;           /* last algo time in microseconds */
    uint32_t infer_us;          /* last inference time in microseconds */
} pipeline_cfg_t;

void     pipeline_cfg_init(void);
uint8_t  pipeline_get_algo(void);
void     pipeline_set_algo(uint8_t algo);
bool     pipeline_get_lcd(void);
void     pipeline_set_lcd(bool en);
bool     pipeline_get_stream(void);
void     pipeline_set_stream(bool en);
bool     pipeline_get_bench(void);
void     pipeline_set_bench(bool en);
bool     pipeline_get_bench_ready(void);
void     pipeline_set_bench_ready(bool en);
void     pipeline_record_timing(uint32_t algo_us, uint32_t infer_us);
uint32_t pipeline_get_fps_x100(void);
uint32_t pipeline_get_algo_us(void);
uint32_t pipeline_get_infer_us(void);

#ifdef __cplusplus
}
#endif

#endif /* _PIPELINE_CONFIG_H_ */
