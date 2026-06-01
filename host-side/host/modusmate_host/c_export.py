"""Export trained per-algorithm classifiers as IMAI-compatible C models.

The firmware in ``camera-imgproc-usb`` consumes models via the
Imagimob ABI: ``IMAI_compute(const uint8_t *in, float *out)`` taking a
``uint8[3, 320, 320]`` RGB tensor and writing 40 floats interpreted as
a column-major ``float[8, 5]`` detection grid (8 attributes x 5
detection slots).  Attribute layout:

* row 0..3 : bbox cx, cy, w, h
* row 4..6 : per-class score (3 classes - Scissors/Paper/Rock for
  the existing camera-imgproc-usb firmware build)
* row 7   : detection-active flag (>0 means the slot is valid)

This module takes a trained sklearn :class:`MLPClassifier` (single
hidden layer is what we use here) plus the dataset's class labels and
generates a drop-in ``model.h`` / ``model.c`` / ``manifest.json`` triple
under ``models/<algo>_<dataset>_<date>/``.  The generated
``IMAI_compute`` does:

1. **Area-average downsample** the 320x320x3 input to ``side x side x 3``
   (where ``side`` matches the size the model was trained at, default
   32).  Pure C, no malloc, no library calls.
2. Normalise to ``float32`` in [0, 1] and run a single hidden-layer MLP
   with ReLU activation and softmax output.
3. Write the top-class probability into the appropriate row-4/5/6 lane,
   with the detection flag set on slot 0 and a centred placeholder bbox
   so the firmware's existing detection parser is happy.

Class-count constraint: the existing firmware decodes exactly 3 class
score lanes.  Models with more or fewer classes can still be exported,
but only the first ``min(n_classes, 3)`` get emitted; the remainder
are clamped.  For best on-board fidelity, train on a 3-class problem
that lines up with the firmware's
``model_to_dataset = {2, 1, 0}`` mapping.

Generated weights are stored as ``static const float`` arrays in the
``IM_ML_MODEL_MEM`` section so they live alongside the real models in
SoCMEM rather than blowing up internal SRAM.
"""
from __future__ import annotations

import datetime as _dt
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


# Camera-side input tensor that the firmware feeds the model.  Must
# match ``IMAI_DATAIN_SHAPE`` of the existing real model so the firmware
# never branches.
FIRMWARE_INPUT_HW = 320
FIRMWARE_INPUT_C = 3
FIRMWARE_OUTPUT_DIMS = (8, 5)  # rows x dets, column-major

#: Maximum number of class-score lanes the firmware decodes.  Anything
#: beyond this gets dropped during export.
FIRMWARE_MAX_CLASSES = 3


@dataclass
class ExportedMLP:
    """Plain description of a one-hidden-layer MLP ready for C codegen.

    The MLP input is a ``(model_h, model_w, 3)`` uint8 image. For square
    models ``model_w == model_h == side``; for rectangular models (e.g.
    full-frame 320x240) ``side`` is set to ``max(w, h)`` for legacy
    callers but the C codegen uses width/height independently.
    """

    #: input side length the MLP was trained on (kept for legacy callers;
    #: equals ``max(model_w, model_h)`` for non-square models)
    side: int
    #: hidden layer size
    hidden: int
    #: output size (== number of classes)
    n_classes: int
    #: shape (input, hidden)
    W1: np.ndarray
    b1: np.ndarray
    #: shape (hidden, n_classes)
    W2: np.ndarray
    b2: np.ndarray
    classes: List[str]
    algo_name: str
    dataset: str
    #: model input width (defaults to ``side`` for square models)
    model_w: int = 0
    #: model input height (defaults to ``side`` for square models)
    model_h: int = 0

    def __post_init__(self) -> None:
        if self.model_w == 0:
            self.model_w = int(self.side)
        if self.model_h == 0:
            self.model_h = int(self.side)

    @classmethod
    def from_sklearn(cls, clf, side: int, classes: Sequence[str],
                     algo_name: str, dataset: str,
                     model_w: Optional[int] = None,
                     model_h: Optional[int] = None) -> "ExportedMLP":
        """Build from a trained ``sklearn.neural_network.MLPClassifier``.

        Only single-hidden-layer MLPs are supported. ``model_w`` /
        ``model_h`` default to ``side`` for square models.
        """
        if len(clf.coefs_) != 2:
            raise ValueError(
                "exporter expects exactly 1 hidden layer "
                f"(got {len(clf.coefs_) - 1})")
        W1, W2 = clf.coefs_
        b1, b2 = clf.intercepts_
        return cls(
            side=int(side),
            hidden=int(W1.shape[1]),
            n_classes=int(W2.shape[1]),
            W1=np.asarray(W1, dtype=np.float32),
            b1=np.asarray(b1, dtype=np.float32),
            W2=np.asarray(W2, dtype=np.float32),
            b2=np.asarray(b2, dtype=np.float32),
            classes=list(classes),
            algo_name=algo_name,
            dataset=dataset,
            model_w=int(model_w) if model_w is not None else int(side),
            model_h=int(model_h) if model_h is not None else int(side),
        )


def _c_array(name: str, arr: np.ndarray) -> str:
    """Render a flat ``static const float NAME[N] = { ... };`` literal."""
    flat = arr.astype(np.float32).ravel()
    body = ", ".join(f"{v:.7g}f" for v in flat)
    # break into lines of 8 floats for readability without going wild
    chunks = body.split(", ")
    lines = []
    for i in range(0, len(chunks), 8):
        lines.append("    " + ", ".join(chunks[i:i + 8]) + ",")
    if lines:
        # strip trailing comma on last value
        lines[-1] = lines[-1].rstrip(",")
    return (
        f"static const float {name}[{flat.size}] IM_ML_MODEL_MEM = {{\n"
        + "\n".join(lines)
        + "\n};\n"
    )


def _bytes_estimate(mlp: ExportedMLP) -> int:
    n_w = mlp.W1.size + mlp.b1.size + mlp.W2.size + mlp.b2.size
    return n_w * 4  # float32


# ----------------------------------------------------------------- header

_HEADER_TPL = """\
/*
 * ModusMate per-algo MLP model — generated by
 * modusmate_host.c_export.write_imai_model().
 *
 *   preprocessing algo : {algo}
 *   trained on dataset : {dataset}
 *   input (host train) : {model_w}x{model_h}x3 uint8
 *   input (firmware)   : 320x320x3 uint8 (downsampled inside IMAI_compute)
 *   hidden layer       : {hidden} units, ReLU
 *   output classes     : {n_classes}  ({classes_pretty})
 *
 * The firmware ABI expects float[8, 5] column-major output.  This
 * model writes its top-{export_classes} class probabilities into the
 * class-score lanes of slot 0 and sets the detection flag.  Class
 * indices beyond {export_classes} are dropped because the firmware's
 * detection parser reads only those score lanes.
 */
#ifndef MODUSMATE_PERALGO_MODEL_H
#define MODUSMATE_PERALGO_MODEL_H

#include <stdbool.h>
#include <stdint.h>
#include "mtb_ml_model.h"

#define IMAI_API_FUNCTION

typedef int8_t  q7_t;
typedef int16_t q15_t;
typedef int32_t q31_t;
typedef int64_t q63_t;

#define IMAI_MODEL_ID {model_id}

#define IMAGINET_TYPES_NONE    (0x0)
#define IMAGINET_TYPES_FLOAT32 (0x14)
#define IMAGINET_TYPES_UINT8   (0x71)

/* datain [320,320,3] (307200 bytes) - matches the real RPS model. */
#define IMAI_DATAIN_RANK    (3)
#define IMAI_DATAIN_SHAPE   (((int[]){{3, 320, 320}})
#define IMAI_DATAIN_COUNT   (307200)
#define IMAI_DATAIN_TYPE    uint8_t
#define IMAI_DATAIN_TYPE_ID IMAGINET_TYPES_UINT8
#define IMAI_DATAIN_SHIFT   0
#define IMAI_DATAIN_OFFSET  0
#define IMAI_DATAIN_SCALE   1
#define IMAI_DATAIN_SYMBOLS {{ }}

/* dataout [8,5] (160 bytes), column-major, matches the real RPS model. */
#define IMAI_DATAOUT_RANK    (2)
#define IMAI_DATAOUT_SHAPE   (((int[]){{5, 8}})
#define IMAI_DATAOUT_COUNT   (40)
#define IMAI_DATAOUT_TYPE    float
#define IMAI_DATAOUT_TYPE_ID IMAGINET_TYPES_FLOAT32
#define IMAI_DATAOUT_SHIFT   0
#define IMAI_DATAOUT_OFFSET  0
#define IMAI_DATAOUT_SCALE   1
#define IMAI_DATAOUT_SYMBOLS {{ }}

#define IMAI_KEY_MAX (8)

#define IMAI_RET_SUCCESS    0
#define IMAI_RET_NODATA    -1
#define IMAI_RET_ERROR     -2
#define IMAI_RET_STREAMEND -3

#define IPWIN_RET_SUCCESS    0
#define IPWIN_RET_NODATA    -1
#define IPWIN_RET_ERROR     -2
#define IPWIN_RET_STREAMEND -3

void IMAI_compute(const uint8_t *restrict datain, float *restrict dataout);
void IMAI_finalize(void);
int  IMAI_init(void);

void IMAI_mtb_models_profile_log(void);
void IMAI_mtb_models_print_info(void);
#define IMAI_MAX_MTB_MODELS 4
extern int32_t          IMAI_mtb_models_count;
extern mtb_ml_model_t  *IMAI_mtb_models[IMAI_MAX_MTB_MODELS];

#define IMAI_REGIONS_COUNT 0
#define IMAI_REGIONS_NAMES {{}}
typedef enum {{ IMAI_REGIONS_EMPTY }} IMAI_Region_t;

typedef enum {{
    IMAI_PARAM_UNDEFINED = 0,
    IMAI_PARAM_INPUT     = 1,
    IMAI_PARAM_OUTPUT    = 2,
    IMAI_PARAM_REFERENCE = 3,
    IMAI_PARAM_HANDLE    = 7,
}} IMAI_param_attrib;

#endif /* MODUSMATE_PERALGO_MODEL_H */
"""


def render_header(mlp: ExportedMLP) -> str:
    rng = np.random.default_rng(abs(hash((mlp.algo_name, mlp.dataset))) % (2 ** 32))
    mid = ", ".join(f"0x{b:02x}" for b in rng.integers(0, 256, 16, dtype=np.uint8))
    classes_pretty = ", ".join(mlp.classes)
    return _HEADER_TPL.format(
        algo=mlp.algo_name, dataset=mlp.dataset,
        model_w=mlp.model_w, model_h=mlp.model_h,
        hidden=mlp.hidden, n_classes=mlp.n_classes,
        classes_pretty=classes_pretty,
        export_classes=min(mlp.n_classes, FIRMWARE_MAX_CLASSES),
        model_id="{" + mid + "}",
    )


# ----------------------------------------------------------------- source

_SOURCE_TPL = """\
/*
 * ModusMate per-algo MLP model — implementation.
 * See model.h for ABI documentation.
 */
#ifndef COMPONENT_ML_TFLM
#error "Symbol COMPONENT_ML_TFLM is not defined. The exported model uses the\
 same build flags as the real Imagimob model — keep NN_MODEL_NAME / NN_TYPE\
 wiring intact in the firmware Makefile."
#endif

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "cy_utils.h"
#include "mtb_ml_model.h"

#include "model.h"

/* The firmware Makefile defines CY_ML_MODEL_MEM=.cy_socmem_data, which
 * forces weights into the SoCMEM region (~4.25 MB on PSE84 EPC2). For
 * full-frame inputs (320x240x3) the weight tensors easily exceed that
 * region. We deliberately ignore the firmware's CY_ML_MODEL_MEM
 * directive and put weights in plain `.rodata`, so they live in the
 * external SMIF flash (m55_nvm region, ~5.75 MB) and execute XIP. The
 * trade-off is slower per-weight access (XIP from QSPI flash vs
 * on-chip SRAM); the win is fitting weights that wouldn't otherwise
 * link. Smaller models still link the same way -- there's no
 * correctness difference, only a latency one. */
#define IM_ML_MODEL_MEM

/* No scratch buffers needed: IMAI_compute_fused below walks every
 * downsample output pixel exactly once and accumulates its
 * contribution into the HIDDEN-dimensional pre-activation vector on
 * the fly. Total static state is HIDDEN + 2*N_CLASSES floats
 * (well under 1 KB even at 320x240 inputs). */

#define MODEL_W      {model_w}
#define MODEL_H      {model_h}
#define HIDDEN       {hidden}
#define N_CLASSES    {n_classes}
#define EXPORT_CLASSES {export_classes}
#define INPUT_FEATS  (MODEL_W * MODEL_H * 3)

int32_t         IMAI_mtb_models_count = 0;
mtb_ml_model_t *IMAI_mtb_models[IMAI_MAX_MTB_MODELS];

{w1_decl}
{b1_decl}
{w2_decl}
{b2_decl}

#define MAX_DET 5
#define OUT_IDX(row, det) ((row) * MAX_DET + (det))

/* Fused downsample + MLP first layer.
 *
 *   y_j = b1[j] + sum over output pixels (oy,ox), channels c of
 *           pixel(oy,ox,c) * W1[((oy*W+ox)*3 + c) * HIDDEN + j]
 *
 * Materialising the (MODEL_W*MODEL_H*3) downsampled tensor would cost
 * ~900 KB of RAM at 320x240; on PSE84 EPC2 the internal data region
 * is 256 KB and putting the buffer in SoCMEM via .cy_socmem_data
 * doubles its flash cost (initialised data has an LMA in m55_nvm).
 * Instead we walk every output pixel once and accumulate its
 * contribution to every hidden unit on the fly. Static state is just
 * h[HIDDEN] + logits/probs[N_CLASSES] (well under 1 KB). */
static void IMAI_compute_fused(const uint8_t *datain, float *dataout)
{{
    static float h[HIDDEN];
    static float logits[N_CLASSES];
    static float probs[N_CLASSES];

    /* Initialise h with biases; the loop below adds W1 contributions. */
    for (int j = 0; j < HIDDEN; j++) h[j] = MLP_B1[j];

    for (int oy = 0; oy < MODEL_H; oy++) {{
        int y0 = (oy * 320) / MODEL_H;
        int y1 = ((oy + 1) * 320) / MODEL_H;
        if (y1 <= y0) y1 = y0 + 1;
        for (int ox = 0; ox < MODEL_W; ox++) {{
            int x0 = (ox * 320) / MODEL_W;
            int x1 = ((ox + 1) * 320) / MODEL_W;
            if (x1 <= x0) x1 = x0 + 1;
            uint32_t sum_r = 0, sum_g = 0, sum_b = 0;
            const int npx = (y1 - y0) * (x1 - x0);
            for (int iy = y0; iy < y1; iy++) {{
                const uint8_t *row = datain + (iy * 320 + x0) * 3;
                for (int ix = 0; ix < (x1 - x0); ix++) {{
                    sum_r += row[ix * 3 + 0];
                    sum_g += row[ix * 3 + 1];
                    sum_b += row[ix * 3 + 2];
                }}
            }}
            const float scale = 1.0f / ((float)npx * 255.0f);
            const float r = (float)sum_r * scale;
            const float g = (float)sum_g * scale;
            const float b = (float)sum_b * scale;
            const int feat = (oy * MODEL_W + ox) * 3;
            const float *w_r = MLP_W1 + (feat + 0) * HIDDEN;
            const float *w_g = MLP_W1 + (feat + 1) * HIDDEN;
            const float *w_b = MLP_W1 + (feat + 2) * HIDDEN;
            for (int j = 0; j < HIDDEN; j++) {{
                h[j] += r * w_r[j] + g * w_g[j] + b * w_b[j];
            }}
        }}
    }}

    /* ReLU */
    for (int j = 0; j < HIDDEN; j++) if (h[j] < 0.0f) h[j] = 0.0f;

    /* Output layer: logits = h @ W2 + b2.
     * W2 is laid out (HIDDEN, N_CLASSES) row-major. */
    for (int k = 0; k < N_CLASSES; k++) {{
        float a = MLP_B2[k];
        const float *col = MLP_W2 + k;
        for (int j = 0; j < HIDDEN; j++) {{
            a += h[j] * col[j * N_CLASSES];
        }}
        logits[k] = a;
    }}

    /* Stable softmax. */
    float m = logits[0];
    for (int k = 1; k < N_CLASSES; k++) if (logits[k] > m) m = logits[k];
    float s = 0.0f;
    for (int k = 0; k < N_CLASSES; k++) {{
        probs[k] = expf(logits[k] - m);
        s += probs[k];
    }}
    if (s <= 0.0f) s = 1.0f;
    for (int k = 0; k < N_CLASSES; k++) probs[k] /= s;

    /* Write result into firmware float[8,5] column-major layout. */
    memset(dataout, 0, sizeof(float) * 40);
    dataout[OUT_IDX(0, 0)] = 160.0f;
    dataout[OUT_IDX(1, 0)] = 120.0f;
    dataout[OUT_IDX(2, 0)] = 100.0f;
    dataout[OUT_IDX(3, 0)] = 100.0f;
    const int ncls = N_CLASSES < EXPORT_CLASSES ? N_CLASSES : EXPORT_CLASSES;
    for (int k = 0; k < ncls; k++) {{
        dataout[OUT_IDX(4 + k, 0)] = probs[k];
    }}
    dataout[OUT_IDX(7, 0)] = 1.0f;  /* slot active */
}}

void IMAI_compute(const uint8_t *restrict datain, float *restrict dataout)
{{
    IMAI_compute_fused(datain, dataout);
}}

void IMAI_finalize(void)  {{ IMAI_mtb_models_count = 0; }}
int  IMAI_init(void)      {{ IMAI_mtb_models_count = 0; return IMAI_RET_SUCCESS; }}
void IMAI_mtb_models_print_info(void) {{ }}
void IMAI_mtb_models_profile_log(void)  {{ }}
"""


def render_source(mlp: ExportedMLP) -> str:
    return _SOURCE_TPL.format(
        model_w=mlp.model_w, model_h=mlp.model_h,
        hidden=mlp.hidden,
        n_classes=mlp.n_classes,
        export_classes=min(mlp.n_classes, FIRMWARE_MAX_CLASSES),
        w1_decl=_c_array("MLP_W1", mlp.W1),
        b1_decl=_c_array("MLP_B1", mlp.b1),
        w2_decl=_c_array("MLP_W2", mlp.W2),
        b2_decl=_c_array("MLP_B2", mlp.b2),
    )


# ---------------------------------------------------------------- manifest


def render_manifest(mlp: ExportedMLP, *, test_acc: Optional[float] = None,
                    prep_us: Optional[float] = None) -> dict:
    return {
        "name": f"{mlp.algo_name}_{mlp.dataset}",
        "description": (
            f"Single-hidden-layer MLP (input={mlp.side}x{mlp.side}x3, "
            f"hidden={mlp.hidden}, classes={mlp.n_classes}) trained on "
            f"the '{mlp.dataset}' dataset with the '{mlp.algo_name}' "
            f"firmware preprocessing applied.  IMAI ABI-compatible with "
            f"the existing camera-imgproc-usb model slot."
        ),
        "framework": "modusmate_mlp_v1",
        "imai_api": True,
        "input_shape": [FIRMWARE_INPUT_HW, FIRMWARE_INPUT_HW, FIRMWARE_INPUT_C],
        "input_dtype": "uint8",
        "output_shape": [FIRMWARE_OUTPUT_DIMS[0], FIRMWARE_OUTPUT_DIMS[1]],
        "output_dtype": "float32",
        "output_layout": "column_major",
        "classes": mlp.classes,
        "flash_bytes": _bytes_estimate(mlp),
        "ram_bytes": (mlp.model_w * mlp.model_h * 3 + mlp.hidden + mlp.n_classes) * 4,
        "prep_algo": mlp.algo_name,
        "training": {
            "dataset": mlp.dataset,
            "side": mlp.side,
            "model_w": mlp.model_w,
            "model_h": mlp.model_h,
            "hidden": mlp.hidden,
            "test_acc": test_acc,
            "host_prep_us": prep_us,
            "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "files": ["model.h", "model.c"],
    }


# ---------------------------------------------------------------- write


def write_imai_model(mlp: ExportedMLP, out_dir: Path,
                     *, test_acc: Optional[float] = None,
                     prep_us: Optional[float] = None) -> Path:
    """Write ``model.h``, ``model.c`` and ``manifest.json`` under ``out_dir``.

    Returns the directory written to.  Caller is responsible for picking
    a unique directory name; the convention used by ``algo_train`` is
    ``models/<algo>_<dataset>_<YYYY-MM-DD>/``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.h").write_text(render_header(mlp))
    (out_dir / "model.c").write_text(render_source(mlp))
    manifest = render_manifest(mlp, test_acc=test_acc, prep_us=prep_us)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return out_dir
