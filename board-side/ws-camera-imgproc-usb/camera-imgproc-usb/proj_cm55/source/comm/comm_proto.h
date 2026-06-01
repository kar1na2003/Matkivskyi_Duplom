/*****************************************************************************
* \file comm/comm_proto.h
*
* \brief ModusMate UART wire protocol. Single source of truth shared with the
*        Python host (host/modusmate_host/protocol.py mirrors these constants).
*
* Frame layout (little endian):
*   [SOF=0xA5][TYPE u8][SEQ u8][LEN u16][PAYLOAD * LEN][CRC16-CCITT u16]
*   CRC is computed over TYPE..PAYLOAD (does NOT include SOF or CRC itself).
*
* Reliability:
*   - Every host -> board command carries a sequence number (SEQ).
*   - The board echoes the SEQ in its EVT_ACK payload so the host can match
*     the right ACK and discard stale duplicates.
*   - The board de-duplicates incoming commands by (TYPE, SEQ) within an
*     8-slot LRU window; a re-sent command after a lost ACK is re-ACKed but
*     not re-executed (idempotent retransmission, like TCP).
*   - The host retransmits a command up to COMM_MAX_RETRIES times if no
*     matching ACK arrives within COMM_ACK_TIMEOUT_MS.
*   - Board -> host events (EVT_*) carry SEQ=0 and are not retransmitted;
*     they are fire-and-forget telemetry. EVT_ACK uses the SEQ of the
*     command it acknowledges (in payload), the frame-level SEQ is 0.
*
* All numeric payload fields are little-endian.
*****************************************************************************/
#ifndef _COMM_PROTO_H_
#define _COMM_PROTO_H_

#include <stdint.h>

#define COMM_PROTOCOL_VERSION    2u

#define COMM_SOF                 0xA5u
#define COMM_MAX_PAYLOAD         1280u  /* INFO packet for 51+ algos plus headroom */
/* Baud rate matches the BSP retarget UART (KitProg3 virtual COM, default
 * 115200 8N1). The protocol is byte-stream only - the host scans for SOF
 * and tolerates printf debug text appearing between frames. */
#define COMM_BAUDRATE            115200u

/* Reliability tunables (host-side; firmware only stores the dedup window). */
#define COMM_ACK_TIMEOUT_MS      300u
#define COMM_MAX_RETRIES         3u
#define COMM_DEDUP_WINDOW        8u   /* board remembers last N (type,seq) */

/* ----- HOST -> BOARD command IDs ----- */
#define COMM_CMD_PING            0x01u
#define COMM_CMD_GET_INFO        0x02u  /* returns EVT_INFO with algo list */
#define COMM_CMD_SET_ALGO        0x10u  /* payload: u8 algo_id */
#define COMM_CMD_SET_LCD         0x11u  /* payload: u8 enable */
#define COMM_CMD_SET_STREAM      0x12u  /* payload: u8 enable */
#define COMM_CMD_SET_BENCH       0x13u  /* payload: u8 enable */
#define COMM_CMD_GET_FPS         0x20u
#define COMM_CMD_SET_BAUDRATE    0x21u  /* payload: u32 baudrate (LE). Board
                                         * ACKs at the current baud, then
                                         * reconfigures the SCB UART to the
                                         * new rate. Host must switch its
                                         * serial port right after the ACK. */
/* Bench image upload: BEGIN { u16 width, u16 height } CHUNK { u8 data... } END {} */
#define COMM_CMD_BENCH_BEGIN     0x30u
#define COMM_CMD_BENCH_CHUNK     0x31u
#define COMM_CMD_BENCH_END       0x32u

/* ----- BOARD -> HOST event IDs ----- */
#define COMM_EVT_ACK             0x80u  /* payload: u8 cmd_id, u8 cmd_seq, u8 status (0=ok) */
#define COMM_EVT_LOG             0x81u  /* payload: ASCII message */
#define COMM_EVT_INFO            0x82u  /* payload: u8 num_algos, then for each:
                                                    u8 id, u8 family, u8 name_len, name */
#define COMM_EVT_FPS             0x90u  /* payload: u32 fps_x100, u32 algo_us, u32 infer_us */
#define COMM_EVT_DETECTION       0x91u  /* payload: u8 count, then per detection:
                                                    u8 class_id, u8 conf_x100,
                                                    i16 x, i16 y, i16 w, i16 h */
#define COMM_EVT_BENCH_RESULT    0x92u  /* payload: u8 class_id, u8 conf_x100,
                                                    u32 algo_us, u32 infer_us */
#define COMM_EVT_FRAME_BEGIN     0xA0u  /* payload: u16 width, u16 height, u8 format */
#define COMM_EVT_FRAME_CHUNK     0xA1u  /* payload: bytes */
#define COMM_EVT_FRAME_END       0xA2u  /* payload: empty */

/* ACK status codes */
#define COMM_ACK_OK              0x00u
#define COMM_ACK_BAD_LEN         0x01u
#define COMM_ACK_BAD_PARAM       0x02u
#define COMM_ACK_UNKNOWN         0xFFu
#define COMM_ACK_BAD_CRC         0xFEu

/* preview format codes */
#define COMM_FRAME_GRAY8         0x01u

/* compact max sizes */
#define COMM_PREVIEW_W           80
#define COMM_PREVIEW_H           60

#endif /* _COMM_PROTO_H_ */
