"""ModusMate host-side controller package."""

from .protocol import (
    SOF, MAX_PAYLOAD,
    CMD_PING, CMD_GET_INFO, CMD_SET_ALGO, CMD_SET_LCD, CMD_SET_STREAM,
    CMD_SET_BENCH, CMD_GET_FPS, CMD_BENCH_BEGIN, CMD_BENCH_CHUNK, CMD_BENCH_END,
    EVT_ACK, EVT_LOG, EVT_INFO, EVT_FPS, EVT_DETECTION, EVT_BENCH_RESULT,
    EVT_FRAME_BEGIN, EVT_FRAME_CHUNK, EVT_FRAME_END,
    crc16_ccitt, encode_frame, FrameDecoder,
)
from .link import BoardLink
from .algos import ALGO_NAMES, ALGO_FAMILIES, family_name

__all__ = [
    "SOF", "MAX_PAYLOAD",
    "CMD_PING", "CMD_GET_INFO", "CMD_SET_ALGO", "CMD_SET_LCD", "CMD_SET_STREAM",
    "CMD_SET_BENCH", "CMD_GET_FPS", "CMD_BENCH_BEGIN", "CMD_BENCH_CHUNK", "CMD_BENCH_END",
    "EVT_ACK", "EVT_LOG", "EVT_INFO", "EVT_FPS", "EVT_DETECTION", "EVT_BENCH_RESULT",
    "EVT_FRAME_BEGIN", "EVT_FRAME_CHUNK", "EVT_FRAME_END",
    "crc16_ccitt", "encode_frame", "FrameDecoder",
    "BoardLink", "ALGO_NAMES", "ALGO_FAMILIES", "family_name",
]
