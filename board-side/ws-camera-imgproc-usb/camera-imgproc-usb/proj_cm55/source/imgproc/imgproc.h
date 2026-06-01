/*****************************************************************************
* \file imgproc/imgproc.h
*
* \brief On-device image-processing library: ~32 selectable algorithms,
*        all with the same signature so they can be dispatched at runtime.
*        Operates in-place on the model input buffer (320x320 RGB888 with
*        the camera image at the top, padding underneath).
*
* The dispatch table is the single source of truth for the algorithm IDs;
* the host-side Python algos.py mirrors it.
*****************************************************************************/
#ifndef _IMGPROC_H_
#define _IMGPROC_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Algorithm IDs - keep in sync with host/modusmate_host/algos.py */
typedef enum {
    IMGPROC_PASSTHROUGH = 0,
    IMGPROC_GRAYSCALE,
    IMGPROC_INVERT,
    IMGPROC_HIST_EQ,
    IMGPROC_GAUSSIAN_3,
    IMGPROC_GAUSSIAN_5,
    IMGPROC_MEAN_3,
    IMGPROC_MEDIAN_3,
    IMGPROC_BILATERAL,        /* lightweight 3x3 bilateral */
    IMGPROC_SOBEL,
    IMGPROC_ROBERTS,
    IMGPROC_PREWITT,
    IMGPROC_SCHARR,
    IMGPROC_KIRSCH,
    IMGPROC_FREI_CHEN,
    IMGPROC_CANNY,
    IMGPROC_MARR_HILDRETH,    /* zero-crossings of LoG */
    IMGPROC_LAPLACIAN,
    IMGPROC_DOG,
    IMGPROC_LOG,
    IMGPROC_DOH,              /* determinant of Hessian */
    IMGPROC_HARRIS,
    IMGPROC_SHI_TOMASI,
    IMGPROC_FAST9,
    IMGPROC_OTSU,
    IMGPROC_ADAPTIVE_MEAN,
    IMGPROC_ADAPTIVE_GAUSSIAN,
    IMGPROC_TRIANGLE,
    IMGPROC_NIBLACK,
    IMGPROC_SAUVOLA,
    IMGPROC_GABOR,
    IMGPROC_LBP,
    IMGPROC_LAWS_ENERGY,
    IMGPROC_FRANGI,
    IMGPROC_HESSIAN_RIDGE,
    IMGPROC_HOG_VIS,          /* HoG cell magnitudes rendered as image */
    IMGPROC_ERODE,
    IMGPROC_DILATE,
    IMGPROC_OPEN,
    IMGPROC_CLOSE,
    IMGPROC_MORPH_GRADIENT,
    IMGPROC_REGION_GROW,
    IMGPROC_WATERSHED,
    IMGPROC_SHARPEN,
    IMGPROC_EMBOSS,
    IMGPROC_MSER,
    IMGPROC_AGAST,
    IMGPROC_BRIEF,
    IMGPROC_AKAZE,
    IMGPROC_BLOB_LOG_MULTISCALE,
    IMGPROC_FAST12,
    IMGPROC_ALGO_COUNT
} imgproc_algo_t;

/* Algorithm metadata used by the comm layer to advertise capability. */
typedef struct {
    const char *name;       /* short ASCII identifier */
    uint8_t     family;     /* 0 basics, 1 edges, 2 blobs, 3 keypoints,
                               4 thresh, 5 texture, 6 ridge, 7 morphology */
} imgproc_info_t;

const imgproc_info_t *imgproc_get_info(uint8_t id);

/* Apply the selected algorithm in-place on a 320x320 RGB888 buffer. The
 * function clears the right-of-camera pad region after processing so the
 * model never sees stale data.
 *
 * Returns elapsed time in microseconds (best-effort, may be 0 if no timer).
 */
uint32_t imgproc_apply(uint8_t algo_id,
                       uint8_t *rgb888,
                       int width,
                       int height,
                       int active_height);

#ifdef __cplusplus
}
#endif
#endif /* _IMGPROC_H_ */
