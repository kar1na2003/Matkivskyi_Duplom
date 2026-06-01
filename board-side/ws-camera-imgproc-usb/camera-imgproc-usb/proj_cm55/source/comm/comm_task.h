/*****************************************************************************
* \file comm/comm_task.h
*****************************************************************************/
#ifndef _COMM_TASK_H_
#define _COMM_TASK_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COMM_TASK_NAME          "CM55 Comm Task"
#define COMM_TASK_STACK_SIZE    (8U * 1024U)
#define COMM_TASK_PRIORITY      (1U)   /* low priority - cooperative */

/* Spawn the task and ready the protocol layer. */
void comm_task_start(void);

/* Send an event frame. Thread-safe, blocks on UART TX. */
bool comm_send(uint8_t evt_id, const uint8_t *payload, uint16_t len);

/* Send an event frame with caller-chosen SEQ byte (used by ACKs which echo
 * the command's SEQ). For normal telemetry events use comm_send() which
 * passes SEQ=0. */
bool comm_send_seq(uint8_t evt_id, uint8_t seq, const uint8_t *payload, uint16_t len);

/* Convenience wrappers used by inference task. */
void comm_send_fps(uint32_t fps_x100, uint32_t algo_us, uint32_t infer_us);
void comm_send_detection(uint8_t count, const uint8_t *class_ids,
                         const uint8_t *conf_x100, const int16_t *bbox_xywh);
void comm_send_bench_result(uint8_t class_id, uint8_t conf_x100,
                            uint32_t algo_us, uint32_t infer_us);

/* Push a downsampled gray preview (80x60). Sent only if streaming enabled. */
void comm_send_preview_gray80x60(const uint8_t *gray);

/* Bench image buffer access for the inference task. Returns NULL until a
 * complete image has been received. The pointer is owned by the comm
 * subsystem and stays valid until pipeline_set_bench_ready(false) is called. */
const uint8_t *comm_bench_image(uint16_t *w_out, uint16_t *h_out);

#ifdef __cplusplus
}
#endif
#endif
