/*****************************************************************************
* \file comm/comm_task.c
*
* \brief UART command/event protocol task. Shares the BSP retarget UART
*        (CYBSP_DEBUG_UART_HW) - reads bytes via PDL polling; writes through
*        a TX mutex that also guards stdio.
*****************************************************************************/
#include "comm_task.h"
#include "comm_proto.h"
#include "../pipeline_config.h"
#include "../imgproc/imgproc.h"
#include "../retarget_io_init.h"

#include "cybsp.h"
#include "cy_pdl.h"
#include "mtb_hal.h"
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include <string.h>

/* Pending baud-rate change: applied after the current command's ACK has
 * been physically transmitted at the old baud.  0 means "no change". */
static volatile uint32_t s_pending_baud = 0u;

/* ---------------- CRC16-CCITT (poly 0x1021, init 0xFFFF) ---------------- */
static uint16_t crc16_ccitt(const uint8_t *p, size_t n)
{
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0; i < n; i++) {
        crc ^= (uint16_t)p[i] << 8;
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

/* ---------------- TX path ---------------- */
static SemaphoreHandle_t   s_tx_mutex;
static StaticSemaphore_t   s_tx_mutex_buf;

static void uart_put_blocking(const uint8_t *data, uint32_t len)
{
    Cy_SCB_UART_PutArrayBlocking(CYBSP_DEBUG_UART_HW, (void *)data, len);
    /* drain TX FIFO so subsequent frames don't interleave with printf */
    while (!Cy_SCB_UART_IsTxComplete(CYBSP_DEBUG_UART_HW)) { /* spin */ }
}

bool comm_send(uint8_t evt_id, const uint8_t *payload, uint16_t len)
{
    return comm_send_seq(evt_id, 0u, payload, len);
}

/* Internal: send a frame with caller-chosen SEQ (used by send_ack). */
bool comm_send_seq(uint8_t evt_id, uint8_t seq, const uint8_t *payload, uint16_t len)
{
    if (len > COMM_MAX_PAYLOAD) return false;

    /* Build the CRC input buffer: TYPE | SEQ | LEN_LO | LEN_HI | PAYLOAD */
    static uint8_t crc_in[4 + COMM_MAX_PAYLOAD];
    crc_in[0] = evt_id;
    crc_in[1] = seq;
    crc_in[2] = (uint8_t)(len & 0xFFu);
    crc_in[3] = (uint8_t)((len >> 8) & 0xFFu);
    if (len) memcpy(&crc_in[4], payload, len);
    uint16_t crc = crc16_ccitt(crc_in, (size_t)(4 + len));

    uint8_t sof = COMM_SOF;
    uint8_t crc_buf[2] = { (uint8_t)(crc & 0xFF), (uint8_t)(crc >> 8) };

    if (xSemaphoreTake(s_tx_mutex, pdMS_TO_TICKS(200)) != pdTRUE) return false;
    uart_put_blocking(&sof, 1);
    uart_put_blocking(crc_in, (uint32_t)(4 + len));   /* type + seq + len + payload */
    uart_put_blocking(crc_buf, 2);
    xSemaphoreGive(s_tx_mutex);
    return true;
}

void comm_send_fps(uint32_t fps_x100, uint32_t algo_us, uint32_t infer_us)
{
    uint8_t buf[12];
    memcpy(&buf[0], &fps_x100, 4);
    memcpy(&buf[4], &algo_us, 4);
    memcpy(&buf[8], &infer_us, 4);
    comm_send(COMM_EVT_FPS, buf, sizeof buf);
}

void comm_send_detection(uint8_t count, const uint8_t *class_ids,
                         const uint8_t *conf_x100, const int16_t *bbox_xywh)
{
    if (count > 5) count = 5;
    uint8_t buf[1 + 5 * (1 + 1 + 8)];
    buf[0] = count;
    int o = 1;
    for (int i = 0; i < count; i++) {
        buf[o++] = class_ids[i];
        buf[o++] = conf_x100[i];
        memcpy(&buf[o], &bbox_xywh[i * 4], 8);
        o += 8;
    }
    comm_send(COMM_EVT_DETECTION, buf, (uint16_t)o);
}

void comm_send_bench_result(uint8_t class_id, uint8_t conf_x100,
                            uint32_t algo_us, uint32_t infer_us)
{
    uint8_t buf[10];
    buf[0] = class_id;
    buf[1] = conf_x100;
    memcpy(&buf[2], &algo_us, 4);
    memcpy(&buf[6], &infer_us, 4);
    comm_send(COMM_EVT_BENCH_RESULT, buf, sizeof buf);
}

void comm_send_preview_gray80x60(const uint8_t *gray)
{
    /* BEGIN */
    uint8_t hdr[5];
    uint16_t w = COMM_PREVIEW_W, h = COMM_PREVIEW_H;
    memcpy(&hdr[0], &w, 2);
    memcpy(&hdr[2], &h, 2);
    hdr[4] = COMM_FRAME_GRAY8;
    comm_send(COMM_EVT_FRAME_BEGIN, hdr, sizeof hdr);

    /* CHUNKs of 240 bytes (80x60 = 4800 bytes -> 20 chunks) */
    const uint16_t CHUNK = 240;
    uint16_t total = (uint16_t)(COMM_PREVIEW_W * COMM_PREVIEW_H);
    uint16_t off = 0;
    while (off < total) {
        uint16_t n = (uint16_t)((total - off) > CHUNK ? CHUNK : (total - off));
        comm_send(COMM_EVT_FRAME_CHUNK, gray + off, n);
        off = (uint16_t)(off + n);
    }
    comm_send(COMM_EVT_FRAME_END, NULL, 0);
}

/* ---------------- RX path / framer ---------------- */
typedef enum {
    RX_SOF, RX_TYPE, RX_SEQ, RX_LEN_LO, RX_LEN_HI,
    RX_PAYLOAD, RX_CRC_LO, RX_CRC_HI
} rx_state_t;

static rx_state_t s_rx_state;
static uint8_t    s_rx_type;
static uint8_t    s_rx_seq;
static uint16_t   s_rx_len;
static uint16_t   s_rx_idx;
static uint8_t    s_rx_payload[COMM_MAX_PAYLOAD];
static uint8_t    s_rx_crc_lo;

/* ---- duplicate-suppression window (TCP-style idempotent retransmit) ---- */
typedef struct { uint8_t type; uint8_t seq; uint8_t status; uint8_t valid; } dedup_entry_t;
static dedup_entry_t s_dedup[COMM_DEDUP_WINDOW];
static uint8_t       s_dedup_head;

static bool dedup_lookup(uint8_t type, uint8_t seq, uint8_t *status_out)
{
    for (uint8_t i = 0; i < COMM_DEDUP_WINDOW; i++) {
        if (s_dedup[i].valid && s_dedup[i].type == type && s_dedup[i].seq == seq) {
            *status_out = s_dedup[i].status;
            return true;
        }
    }
    return false;
}

static void dedup_store(uint8_t type, uint8_t seq, uint8_t status)
{
    s_dedup[s_dedup_head].type = type;
    s_dedup[s_dedup_head].seq = seq;
    s_dedup[s_dedup_head].status = status;
    s_dedup[s_dedup_head].valid = 1u;
    s_dedup_head = (uint8_t)((s_dedup_head + 1u) % COMM_DEDUP_WINDOW);
}

/* Bench image buffer (320x240 RGB888). Lives in SoCMEM. */
static __attribute__((section(".cy_socmem_data"), aligned(16)))
       uint8_t s_bench_image[320 * 240 * 3];
static uint32_t s_bench_off;
static uint16_t s_bench_w, s_bench_h;

const uint8_t *comm_bench_image(uint16_t *w_out, uint16_t *h_out)
{
    if (!pipeline_get_bench_ready()) return NULL;
    if (w_out) *w_out = s_bench_w;
    if (h_out) *h_out = s_bench_h;
    return s_bench_image;
}

/* ACK = [cmd_id, cmd_seq, status]. Frame-level SEQ of EVT_ACK is 0. */
static void send_ack(uint8_t cmd, uint8_t seq, uint8_t status)
{
    uint8_t b[3] = { cmd, seq, status };
    comm_send_seq(COMM_EVT_ACK, 0u, b, 3);
}

static void send_info(void)
{
    uint8_t buf[COMM_MAX_PAYLOAD];
    int o = 0;
    buf[o++] = (uint8_t)IMGPROC_ALGO_COUNT;
    for (uint8_t i = 0; i < IMGPROC_ALGO_COUNT; i++) {
        const imgproc_info_t *info = imgproc_get_info(i);
        if (!info) continue;
        uint8_t nlen = (uint8_t)strlen(info->name);
        if (o + 3 + nlen > (int)sizeof buf) break;
        buf[o++] = i;
        buf[o++] = info->family;
        buf[o++] = nlen;
        memcpy(&buf[o], info->name, nlen);
        o += nlen;
    }
    comm_send(COMM_EVT_INFO, buf, (uint16_t)o);
}

static void process_command(void)
{
    /* duplicate? re-send the cached ACK without re-running the command. */
    uint8_t cached_status;
    if (dedup_lookup(s_rx_type, s_rx_seq, &cached_status)) {
        send_ack(s_rx_type, s_rx_seq, cached_status);
        return;
    }

    uint8_t status = COMM_ACK_OK;
    switch (s_rx_type) {
    case COMM_CMD_PING:
        break;
    case COMM_CMD_GET_INFO:
        send_info();
        break;
    case COMM_CMD_SET_ALGO:
        if (s_rx_len >= 1) pipeline_set_algo(s_rx_payload[0]);
        else status = COMM_ACK_BAD_LEN;
        break;
    case COMM_CMD_SET_LCD:
        if (s_rx_len >= 1) pipeline_set_lcd(s_rx_payload[0] != 0);
        else status = COMM_ACK_BAD_LEN;
        break;
    case COMM_CMD_SET_STREAM:
        if (s_rx_len >= 1) pipeline_set_stream(s_rx_payload[0] != 0);
        else status = COMM_ACK_BAD_LEN;
        break;
    case COMM_CMD_SET_BENCH:
        if (s_rx_len >= 1) {
            pipeline_set_bench(s_rx_payload[0] != 0);
            if (s_rx_payload[0] == 0) pipeline_set_bench_ready(false);
        } else status = COMM_ACK_BAD_LEN;
        break;
    case COMM_CMD_GET_FPS:
        comm_send_fps(pipeline_get_fps_x100(),
                      pipeline_get_algo_us(),
                      pipeline_get_infer_us());
        break;
    case COMM_CMD_SET_BAUDRATE:
        if (s_rx_len == 4) {
            uint32_t b;
            memcpy(&b, s_rx_payload, 4);
            /* Sanity-limit: KitProg3 VCOM is reliable up to ~3 Mbaud.  We
             * accept 9600..3000000 and defer the actual switch until after
             * the ACK has been sent at the current baud. */
            if (b >= 9600u && b <= 3000000u) {
                s_pending_baud = b;
            } else {
                status = COMM_ACK_BAD_PARAM;
            }
        } else {
            status = COMM_ACK_BAD_LEN;
        }
        break;
    case COMM_CMD_BENCH_BEGIN:
        if (s_rx_len >= 4) {
            memcpy(&s_bench_w, &s_rx_payload[0], 2);
            memcpy(&s_bench_h, &s_rx_payload[2], 2);
            if ((uint32_t)s_bench_w * s_bench_h * 3 > sizeof s_bench_image) {
                status = COMM_ACK_BAD_PARAM;
            } else {
                s_bench_off = 0;
                pipeline_set_bench_ready(false);
            }
        } else status = COMM_ACK_BAD_LEN;
        break;
    case COMM_CMD_BENCH_CHUNK:
        if (s_bench_off + s_rx_len <= sizeof s_bench_image) {
            memcpy(&s_bench_image[s_bench_off], s_rx_payload, s_rx_len);
            s_bench_off += s_rx_len;
        } else status = COMM_ACK_BAD_LEN;
        break;
    case COMM_CMD_BENCH_END:
        if (s_bench_off == (uint32_t)s_bench_w * s_bench_h * 3) {
            pipeline_set_bench_ready(true);
        } else status = COMM_ACK_BAD_PARAM;
        break;
    default:
        status = COMM_ACK_UNKNOWN;
        break;
    }

    dedup_store(s_rx_type, s_rx_seq, status);
    send_ack(s_rx_type, s_rx_seq, status);

    /* Deferred baud-rate change: the ACK above used blocking PutArray +
     * IsTxComplete spin, so by the time we reach this point the ACK bytes
     * have physically left the wire at the old baud.  Safe to switch now. */
    if (s_pending_baud != 0u && status == COMM_ACK_OK) {
        uint32_t target = s_pending_baud;
        s_pending_baud = 0u;
        /* Take tx mutex so no concurrent printf() corrupts the reconfig. */
        if (xSemaphoreTake(s_tx_mutex, pdMS_TO_TICKS(200)) == pdTRUE) {
            uint32_t actual = 0u;
            (void)mtb_hal_uart_set_baud(debug_uart_hal(), target, &actual);
            xSemaphoreGive(s_tx_mutex);
        }
    }
}

static void feed_byte(uint8_t b)
{
    switch (s_rx_state) {
    case RX_SOF:
        if (b == COMM_SOF) s_rx_state = RX_TYPE;
        break;
    case RX_TYPE:
        s_rx_type = b; s_rx_state = RX_SEQ; break;
    case RX_SEQ:
        s_rx_seq = b; s_rx_state = RX_LEN_LO; break;
    case RX_LEN_LO:
        s_rx_len = b; s_rx_state = RX_LEN_HI; break;
    case RX_LEN_HI:
        s_rx_len |= (uint16_t)b << 8;
        if (s_rx_len > COMM_MAX_PAYLOAD) { s_rx_state = RX_SOF; break; }
        s_rx_idx = 0;
        s_rx_state = (s_rx_len == 0) ? RX_CRC_LO : RX_PAYLOAD;
        break;
    case RX_PAYLOAD:
        s_rx_payload[s_rx_idx++] = b;
        if (s_rx_idx >= s_rx_len) s_rx_state = RX_CRC_LO;
        break;
    case RX_CRC_LO:
        s_rx_crc_lo = b; s_rx_state = RX_CRC_HI; break;
    case RX_CRC_HI: {
        uint16_t got = (uint16_t)((b << 8) | s_rx_crc_lo);
        uint8_t  buf[4 + COMM_MAX_PAYLOAD];
        buf[0] = s_rx_type;
        buf[1] = s_rx_seq;
        buf[2] = (uint8_t)(s_rx_len & 0xFF);
        buf[3] = (uint8_t)(s_rx_len >> 8);
        if (s_rx_len) memcpy(&buf[4], s_rx_payload, s_rx_len);
        uint16_t expect = crc16_ccitt(buf, (size_t)(4 + s_rx_len));
        if (expect == got) {
            process_command();
        } else {
            /* Drop silently: a re-tx with correct CRC will get through.
             * Don't ACK with bad-CRC because the host won't know the seq. */
        }
        s_rx_state = RX_SOF;
        break;
    }
    }
}

static void comm_task_main(void *arg)
{
    (void)arg;
    /* Read in batches of up to 64 bytes per syscall to keep up with the
     * UART RX FIFO at high baud (3 Mbaud = 300 KB/s = a 240-byte chunk
     * every 0.8 ms).  Per-byte polling at 1+ Mbaud was causing FIFO
     * overruns -> dropped bytes -> CRC fail -> BENCH_END BAD_PARAM. */
    uint8_t buf[64];
    for (;;) {
        uint32_t got = Cy_SCB_UART_GetArray(CYBSP_DEBUG_UART_HW,
                                            buf, sizeof buf);
        if (got) {
            for (uint32_t i = 0; i < got; i++) feed_byte(buf[i]);
            /* Stay in the loop: if the FIFO refilled while we processed
             * this batch, drain it before yielding.  Only yield when the
             * FIFO is genuinely empty. */
        } else {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }
}

void comm_task_start(void)
{
    s_tx_mutex = xSemaphoreCreateMutexStatic(&s_tx_mutex_buf);
    s_rx_state = RX_SOF;
    xTaskCreate(comm_task_main, COMM_TASK_NAME,
                COMM_TASK_STACK_SIZE / sizeof(StackType_t),
                NULL, COMM_TASK_PRIORITY, NULL);
}
