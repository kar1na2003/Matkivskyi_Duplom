/*****************************************************************************
* \file pipeline_config.c
* \brief Implementation of the runtime pipeline configuration store.
*
* All getters/setters use a single FreeRTOS mutex. The fields are simple
* scalars so this is overkill on a 32-bit core, but keeps semantics
* obviously correct and allows future fields (strings, structs) to slot in
* without changing call sites.
*****************************************************************************/
#include "pipeline_config.h"
#include "FreeRTOS.h"
#include "semphr.h"

static pipeline_cfg_t s_cfg;
static SemaphoreHandle_t s_lock;
static StaticSemaphore_t s_lock_buf;

static inline void lock(void)   { xSemaphoreTake(s_lock, portMAX_DELAY); }
static inline void unlock(void) { xSemaphoreGive(s_lock); }

void pipeline_cfg_init(void)
{
    s_lock = xSemaphoreCreateMutexStatic(&s_lock_buf);
    s_cfg.selected_algo     = 0u;     /* passthrough by default */
    s_cfg.lcd_enabled       = true;
    s_cfg.stream_enabled    = false;
    s_cfg.bench_mode        = false;
    s_cfg.bench_image_ready = false;
    s_cfg.fps_x100          = 0u;
    s_cfg.algo_us           = 0u;
    s_cfg.infer_us          = 0u;
}

uint8_t pipeline_get_algo(void)
{
    lock();
    uint8_t v = s_cfg.selected_algo;
    unlock();
    return v;
}

void pipeline_set_algo(uint8_t algo)
{
    if (algo > PIPELINE_MAX_ALGO_ID) {
        return;
    }
    lock();
    s_cfg.selected_algo = algo;
    unlock();
}

bool pipeline_get_lcd(void)   { lock(); bool v = s_cfg.lcd_enabled;       unlock(); return v; }
void pipeline_set_lcd(bool e) { lock(); s_cfg.lcd_enabled = e;            unlock(); }
bool pipeline_get_stream(void){ lock(); bool v = s_cfg.stream_enabled;    unlock(); return v; }
void pipeline_set_stream(bool e){lock(); s_cfg.stream_enabled = e;        unlock(); }
bool pipeline_get_bench(void) { lock(); bool v = s_cfg.bench_mode;        unlock(); return v; }
void pipeline_set_bench(bool e){lock(); s_cfg.bench_mode = e;             unlock(); }
bool pipeline_get_bench_ready(void){lock(); bool v=s_cfg.bench_image_ready; unlock(); return v;}
void pipeline_set_bench_ready(bool e){lock(); s_cfg.bench_image_ready = e; unlock();}

void pipeline_record_timing(uint32_t algo_us, uint32_t infer_us)
{
    lock();
    s_cfg.algo_us  = algo_us;
    s_cfg.infer_us = infer_us;
    /* fps from total frame time, low-pass filtered */
    uint32_t total = algo_us + infer_us;
    if (total > 0u) {
        uint32_t inst = (uint32_t)(100000000ull / (uint64_t)total); /* 1e6/total *100 */
        s_cfg.fps_x100 = (s_cfg.fps_x100 * 3u + inst) / 4u;
    }
    unlock();
}

uint32_t pipeline_get_fps_x100(void){lock(); uint32_t v=s_cfg.fps_x100; unlock(); return v;}
uint32_t pipeline_get_algo_us(void) {lock(); uint32_t v=s_cfg.algo_us;  unlock(); return v;}
uint32_t pipeline_get_infer_us(void){lock(); uint32_t v=s_cfg.infer_us; unlock(); return v;}
