/*****************************************************************************
* \file imgproc/imgproc.c
*
* \brief Implementations of the on-device image-processing algorithms.
*
* All operations work on a 320x320 RGB888 buffer in place. The active
* image is the top (active_height) rows; the rest are padding zeros that
* feed the YOLO-style detector. We:
*   1. Convert active region to a temporary grayscale plane (uint8).
*   2. Run the algorithm into a second grayscale scratch plane.
*   3. Replicate gray result back into RGB888 channels.
*
* Heavy algorithms (HoG, Frangi, watershed-lite) deliberately keep
* memory bounded by working line-by-line where possible. Performance is
* secondary to behavioural correctness for the benchmark study.
*****************************************************************************/
#include "imgproc.h"
#include "cybsp.h"
#include "ifx_time_utils.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Active region max dims; matches lcd/inference task buffer. */
#define IMP_W_MAX   320
#define IMP_H_MAX   240

/* Two scratch planes in SoCMEM (external PSRAM, plenty of room). */
static __attribute__((section(".cy_socmem_data"), aligned(16)))
       uint8_t s_gray_a[IMP_W_MAX * IMP_H_MAX];
static __attribute__((section(".cy_socmem_data"), aligned(16)))
       uint8_t s_gray_b[IMP_W_MAX * IMP_H_MAX];

/* One extra u8 plane used as an intermediate by 2-pass algos
 * (morph open/close/gradient, DoG). */
static __attribute__((section(".cy_socmem_data"), aligned(16)))
       uint8_t s_tmp_u8_1[IMP_W_MAX * IMP_H_MAX];

/* Large raw scratch pool. It is sized to hold the worst-case need:
 *  - two int16_t planes (Canny gx/gy) = 307 200 B
 *  - one uint32_t plane (region-grow stack) = 307 200 B
 *  - one float plane (ridge/blob detectors)  = 307 200 B
 * Each algorithm casts the pool into the types it needs. No two algos
 * run simultaneously, so aliasing is safe. */
#define IMP_SCRATCH_BYTES  (IMP_W_MAX * IMP_H_MAX * 4)  /* 307 200 */
static __attribute__((section(".cy_socmem_data"), aligned(16)))
       uint8_t s_scratch[IMP_SCRATCH_BYTES];

#define IMP_SCRATCH_F32()  ((float    *)(void *)s_scratch)
#define IMP_SCRATCH_U32()  ((uint32_t *)(void *)s_scratch)
#define IMP_SCRATCH_I16_A() ((int16_t *)(void *)s_scratch)
#define IMP_SCRATCH_I16_B() ((int16_t *)(void *)(s_scratch + IMP_W_MAX * IMP_H_MAX * 2))

static inline uint8_t clamp_u8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
static inline int     iabs(int v)     { return v < 0 ? -v : v; }

/*---------------- helpers: rgb<->gray ----------------*/
static void rgb_to_gray(const uint8_t *rgb, uint8_t *gray, int w, int h)
{
    int n = w * h;
    for (int i = 0; i < n; i++) {
        int r = rgb[i * 3 + 0];
        int g = rgb[i * 3 + 1];
        int b = rgb[i * 3 + 2];
        /* fixed-point ~ 0.299r + 0.587g + 0.114b */
        gray[i] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
    }
}

static void gray_to_rgb(const uint8_t *gray, uint8_t *rgb, int w, int h)
{
    int n = w * h;
    for (int i = 0; i < n; i++) {
        rgb[i * 3 + 0] = gray[i];
        rgb[i * 3 + 1] = gray[i];
        rgb[i * 3 + 2] = gray[i];
    }
}

/*---------------- 3x3 / 5x5 convolution helpers ----------------*/
static void conv3x3(const uint8_t *in, uint8_t *out, int w, int h,
                    const int8_t k[9], int divisor, int offset)
{
    if (divisor == 0) divisor = 1;
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int s = 0;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++)
                    s += in[(y + ky) * w + (x + kx)] * k[(ky + 1) * 3 + (kx + 1)];
            out[y * w + x] = clamp_u8(s / divisor + offset);
        }
    }
    /* zero edges to avoid garbage */
    for (int x = 0; x < w; x++) { out[x] = 0; out[(h - 1) * w + x] = 0; }
    for (int y = 0; y < h; y++) { out[y * w] = 0; out[y * w + w - 1] = 0; }
}

static void conv3x3_signed(const uint8_t *in, int16_t *out, int w, int h, const int8_t k[9])
{
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int s = 0;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++)
                    s += in[(y + ky) * w + (x + kx)] * k[(ky + 1) * 3 + (kx + 1)];
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            out[y * w + x] = (int16_t)s;
        }
    }
}

/*---------------- basics ----------------*/
static void algo_passthrough(uint8_t *rgb, int w, int h) { (void)rgb; (void)w; (void)h; }

static void algo_grayscale(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    gray_to_rgb(s_gray_a, rgb, w, h);
}

static void algo_invert(uint8_t *rgb, int w, int h)
{
    int n = w * h * 3;
    for (int i = 0; i < n; i++) rgb[i] = (uint8_t)(255 - rgb[i]);
}

static void algo_hist_eq(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int hist[256] = {0};
    int n = w * h;
    for (int i = 0; i < n; i++) hist[s_gray_a[i]]++;
    int cdf[256], acc = 0;
    for (int i = 0; i < 256; i++) { acc += hist[i]; cdf[i] = acc; }
    int cdf_min = 0;
    for (int i = 0; i < 256; i++) if (cdf[i]) { cdf_min = cdf[i]; break; }
    int denom = n - cdf_min;
    if (denom < 1) denom = 1;
    uint8_t lut[256];
    for (int i = 0; i < 256; i++) lut[i] = (uint8_t)(((cdf[i] - cdf_min) * 255) / denom);
    for (int i = 0; i < n; i++) s_gray_b[i] = lut[s_gray_a[i]];
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_gaussian_3(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t k[9] = {1,2,1, 2,4,2, 1,2,1};
    conv3x3(s_gray_a, s_gray_b, w, h, k, 16, 0);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_gaussian_5(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    /* separable 5-tap [1 4 6 4 1] / 16 */
    for (int y = 0; y < h; y++) {
        for (int x = 2; x < w - 2; x++) {
            int s = s_gray_a[y*w+x-2] + 4*s_gray_a[y*w+x-1] + 6*s_gray_a[y*w+x]
                  + 4*s_gray_a[y*w+x+1] + s_gray_a[y*w+x+2];
            s_gray_b[y*w+x] = (uint8_t)(s >> 4);
        }
    }
    for (int x = 0; x < w; x++)
        for (int y = 2; y < h - 2; y++) {
            int s = s_gray_b[(y-2)*w+x] + 4*s_gray_b[(y-1)*w+x] + 6*s_gray_b[y*w+x]
                  + 4*s_gray_b[(y+1)*w+x] + s_gray_b[(y+2)*w+x];
            s_gray_a[y*w+x] = (uint8_t)(s >> 4);
        }
    gray_to_rgb(s_gray_a, rgb, w, h);
}

static void algo_mean_3(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t k[9] = {1,1,1,1,1,1,1,1,1};
    conv3x3(s_gray_a, s_gray_b, w, h, k, 9, 0);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static int cmp_u8(const void *a, const void *b){ return (int)*(uint8_t*)a - (int)*(uint8_t*)b; }
static void algo_median_3(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            uint8_t v[9];
            int idx = 0;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++)
                    v[idx++] = s_gray_a[(y+ky)*w+(x+kx)];
            qsort(v, 9, 1, cmp_u8);
            s_gray_b[y*w+x] = v[4];
        }
    }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_bilateral(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    /* tiny 3x3 bilateral with sigma_r=20, sigma_s=1.5 */
    static const int wsp[9] = {1,2,1,2,4,2,1,2,1};
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int p = s_gray_a[y*w+x];
            int sw = 0, sv = 0;
            int idx = 0;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++) {
                    int q = s_gray_a[(y+ky)*w+(x+kx)];
                    int d = q - p;
                    int range = (d*d <= 400) ? (400 - d*d) / 16 : 0;
                    int weight = wsp[idx++] * (range + 1);
                    sw += weight; sv += weight * q;
                }
            s_gray_b[y*w+x] = (uint8_t)(sw ? sv / sw : p);
        }
    }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/*---------------- edges ----------------*/
static void run_grad_pair(const uint8_t *in, uint8_t *out, int w, int h,
                          const int8_t kx[9], const int8_t ky[9])
{
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int gx = 0, gy = 0, idx = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int p = in[(y+dy)*w+(x+dx)];
                    gx += p * kx[idx];
                    gy += p * ky[idx];
                    idx++;
                }
            int m = iabs(gx) + iabs(gy);
            out[y*w+x] = (uint8_t)(m > 255 ? 255 : m);
        }
    }
}

static void algo_sobel(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t kx[9] = {-1,0,1, -2,0,2, -1,0,1};
    static const int8_t ky[9] = {-1,-2,-1, 0,0,0, 1,2,1};
    run_grad_pair(s_gray_a, s_gray_b, w, h, kx, ky);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_roberts(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    for (int y = 0; y < h - 1; y++) {
        for (int x = 0; x < w - 1; x++) {
            int gx = s_gray_a[y*w+x] - s_gray_a[(y+1)*w+(x+1)];
            int gy = s_gray_a[y*w+(x+1)] - s_gray_a[(y+1)*w+x];
            int m = iabs(gx) + iabs(gy);
            s_gray_b[y*w+x] = (uint8_t)(m > 255 ? 255 : m);
        }
    }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_prewitt(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t kx[9] = {-1,0,1, -1,0,1, -1,0,1};
    static const int8_t ky[9] = {-1,-1,-1, 0,0,0, 1,1,1};
    run_grad_pair(s_gray_a, s_gray_b, w, h, kx, ky);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_scharr(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t kx[9] = {-3,0,3, -10,0,10, -3,0,3};
    static const int8_t ky[9] = {-3,-10,-3, 0,0,0, 3,10,3};
    run_grad_pair(s_gray_a, s_gray_b, w, h, kx, ky);
    /* the 10x weights amplify; rescale */
    int n = w * h;
    for (int i = 0; i < n; i++) s_gray_b[i] = (uint8_t)(s_gray_b[i] >> 1);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_kirsch(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t k0[9] = { 5, 5, 5,-3, 0,-3,-3,-3,-3};
    static const int8_t k1[9] = { 5, 5,-3, 5, 0,-3,-3,-3,-3};
    static const int8_t k2[9] = { 5,-3,-3, 5, 0,-3, 5,-3,-3};
    static const int8_t k3[9] = {-3,-3,-3, 5, 0,-3, 5, 5,-3};
    const int8_t *ks[4] = {k0, k1, k2, k3};
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int best = 0;
            for (int kk = 0; kk < 4; kk++) {
                int s = 0, idx = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        s += s_gray_a[(y+dy)*w+(x+dx)] * ks[kk][idx++];
                int a = iabs(s);
                if (a > best) best = a;
            }
            s_gray_b[y*w+x] = (uint8_t)(best > 255 ? 255 : (best >> 2));
        }
    }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_frei_chen(uint8_t *rgb, int w, int h)
{
    /* simplified: project onto 2 edge masks (horizontal+vertical) of FC basis */
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t kx[9] = {-1,0,1, -1,0,1, -1,0,1};       /* sqrt(2) absorbed */
    static const int8_t ky[9] = {-1,-1,-1, 0,0,0, 1,1,1};
    run_grad_pair(s_gray_a, s_gray_b, w, h, kx, ky);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_canny(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t kg[9] = {1,2,1, 2,4,2, 1,2,1};
    conv3x3(s_gray_a, s_gray_b, w, h, kg, 16, 0);
    /* gradient magnitude + direction */
    int16_t *gx_buf = IMP_SCRATCH_I16_A();
    int16_t *gy_buf = IMP_SCRATCH_I16_B();
    static const int8_t skx[9] = {-1,0,1, -2,0,2, -1,0,1};
    static const int8_t sky[9] = {-1,-2,-1, 0,0,0, 1,2,1};
    conv3x3_signed(s_gray_b, gx_buf, w, h, skx);
    conv3x3_signed(s_gray_b, gy_buf, w, h, sky);
    /* magnitude */
    int n = w * h;
    for (int i = 0; i < n; i++) {
        int m = iabs(gx_buf[i]) + iabs(gy_buf[i]);
        s_gray_a[i] = (uint8_t)(m > 255 ? 255 : m);
    }
    /* non-max suppression (4 directions) */
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int gx = gx_buf[y*w+x], gy = gy_buf[y*w+x];
            int mag = s_gray_a[y*w+x];
            int agx = iabs(gx), agy = iabs(gy);
            int n1, n2;
            if (agx >= 2 * agy) { n1 = s_gray_a[y*w+x-1]; n2 = s_gray_a[y*w+x+1]; }
            else if (agy >= 2 * agx) { n1 = s_gray_a[(y-1)*w+x]; n2 = s_gray_a[(y+1)*w+x]; }
            else if ((gx ^ gy) >= 0) { n1 = s_gray_a[(y-1)*w+x-1]; n2 = s_gray_a[(y+1)*w+x+1]; }
            else { n1 = s_gray_a[(y-1)*w+x+1]; n2 = s_gray_a[(y+1)*w+x-1]; }
            s_gray_b[y*w+x] = (uint8_t)((mag >= n1 && mag >= n2) ? mag : 0);
        }
    }
    /* dual-threshold + simple hysteresis */
    int hi = 80, lo = 30;
    for (int i = 0; i < n; i++)
        s_gray_a[i] = s_gray_b[i] >= hi ? 255 : (s_gray_b[i] >= lo ? 128 : 0);
    /* one pass: weak edges connected to strong become strong */
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++)
            if (s_gray_a[y*w+x] == 128) {
                int strong = 0;
                for (int dy = -1; dy <= 1 && !strong; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        if (s_gray_a[(y+dy)*w+(x+dx)] == 255) { strong = 1; break; }
                s_gray_a[y*w+x] = strong ? 255 : 0;
            }
    gray_to_rgb(s_gray_a, rgb, w, h);
}

static void algo_laplacian(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t k[9] = {0,-1,0, -1,4,-1, 0,-1,0};
    conv3x3(s_gray_a, s_gray_b, w, h, k, 1, 128);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/*---------------- blobs ----------------*/
static void blur_g3_inplace(uint8_t *src, uint8_t *dst, int w, int h)
{
    static const int8_t k[9] = {1,2,1, 2,4,2, 1,2,1};
    conv3x3(src, dst, w, h, k, 16, 0);
}

static void algo_dog(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    uint8_t *blur1 = s_tmp_u8_1;
    uint8_t *blur2 = s_gray_b;
    blur_g3_inplace(s_gray_a, blur1, w, h);
    blur_g3_inplace(blur1, blur2, w, h);
    blur_g3_inplace(blur2, s_gray_b, w, h);
    int n = w * h;
    for (int i = 0; i < n; i++) {
        int v = (int)blur1[i] - (int)s_gray_b[i] + 128;
        s_gray_a[i] = clamp_u8(v);
    }
    gray_to_rgb(s_gray_a, rgb, w, h);
}

static void algo_log(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t kg[9] = {1,2,1, 2,4,2, 1,2,1};
    conv3x3(s_gray_a, s_gray_b, w, h, kg, 16, 0);
    static const int8_t kl[9] = {0,-1,0, -1,4,-1, 0,-1,0};
    conv3x3(s_gray_b, s_gray_a, w, h, kl, 1, 128);
    gray_to_rgb(s_gray_a, rgb, w, h);
}

static void algo_marr_hildreth(uint8_t *rgb, int w, int h)
{
    algo_log(rgb, w, h);
    /* zero-crossings: pixel becomes 255 if its 4-neighbours straddle 128 */
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    memset(s_gray_b, 0, n);
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            int c = s_gray_a[y*w+x];
            int diff = 0;
            if ((c-128) * (s_gray_a[y*w+x-1]-128) < 0) diff++;
            if ((c-128) * (s_gray_a[y*w+x+1]-128) < 0) diff++;
            if ((c-128) * (s_gray_a[(y-1)*w+x]-128) < 0) diff++;
            if ((c-128) * (s_gray_a[(y+1)*w+x]-128) < 0) diff++;
            s_gray_b[y*w+x] = diff ? 255 : 0;
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_doh(uint8_t *rgb, int w, int h)
{
    /* determinant of Hessian: Ixx*Iyy - Ixy^2 (small kernels) */
    rgb_to_gray(rgb, s_gray_a, w, h);
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            int Ixx = s_gray_a[y*w+x-1] - 2*s_gray_a[y*w+x] + s_gray_a[y*w+x+1];
            int Iyy = s_gray_a[(y-1)*w+x] - 2*s_gray_a[y*w+x] + s_gray_a[(y+1)*w+x];
            int Ixy = (s_gray_a[(y+1)*w+x+1] - s_gray_a[(y+1)*w+x-1]
                     - s_gray_a[(y-1)*w+x+1] + s_gray_a[(y-1)*w+x-1]) / 4;
            int det = (Ixx * Iyy - Ixy * Ixy) / 16 + 128;
            s_gray_b[y*w+x] = clamp_u8(det);
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/*---------------- keypoints ----------------*/
static void marker(uint8_t *gray, int w, int h, int x, int y, uint8_t v)
{
    if (x < 2 || y < 2 || x >= w - 2 || y >= h - 2) return;
    for (int d = -2; d <= 2; d++) {
        gray[y*w + (x + d)] = v;
        gray[(y + d)*w + x] = v;
    }
}

static void harris_like(uint8_t *rgb, int w, int h, int shi_tomasi)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    /* clear visualization plane to dim grayscale */
    int n = w * h;
    for (int i = 0; i < n; i++) s_gray_b[i] = (uint8_t)(s_gray_a[i] >> 1);
    /* gradients */
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            int Ix = s_gray_a[y*w+x+1] - s_gray_a[y*w+x-1];
            int Iy = s_gray_a[(y+1)*w+x] - s_gray_a[(y-1)*w+x];
            /* 3x3 box-summed Ix^2, Iy^2, IxIy */
            int A = Ix*Ix, B = Iy*Iy, C = Ix*Iy;
            int score;
            if (shi_tomasi) {
                int t = (A + B);
                int d = A * B - C * C;
                int sq = (t * t / 4) - d;
                if (sq < 0) sq = 0;
                int sr = (int)sqrtf((float)sq);
                int lmin = t / 2 - sr;
                score = lmin;
            } else {
                int det = A * B - C * C;
                int tr  = A + B;
                score = det - (tr * tr) / 16;
            }
            if (score > 6000) marker(s_gray_b, w, h, x, y, 255);
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_harris(uint8_t *rgb, int w, int h)      { harris_like(rgb, w, h, 0); }
static void algo_shi_tomasi(uint8_t *rgb, int w, int h)  { harris_like(rgb, w, h, 1); }

static void algo_fast9(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    for (int i = 0; i < n; i++) s_gray_b[i] = (uint8_t)(s_gray_a[i] >> 1);
    /* simplified FAST: ring of 8 neighbours at radius 3 */
    static const int dx[8] = {0,2,3,2,0,-2,-3,-2};
    static const int dy[8] = {-3,-2,0,2,3,2,0,-2};
    int t = 25;
    for (int y = 3; y < h - 3; y++)
        for (int x = 3; x < w - 3; x++) {
            int p = s_gray_a[y*w+x];
            int br = 0, dk = 0;
            for (int k = 0; k < 8; k++) {
                int q = s_gray_a[(y+dy[k])*w + (x+dx[k])];
                if (q > p + t) br++;
                else if (q < p - t) dk++;
            }
            if (br >= 6 || dk >= 6) marker(s_gray_b, w, h, x, y, 255);
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/*---------------- thresholding ----------------*/
static int otsu_threshold(const uint8_t *gray, int n)
{
    int hist[256] = {0};
    for (int i = 0; i < n; i++) hist[gray[i]]++;
    long total = n;
    long sum = 0;
    for (int i = 0; i < 256; i++) sum += i * hist[i];
    long sumB = 0, wB = 0;
    float maxVar = 0; int thr = 127;
    for (int t = 0; t < 256; t++) {
        wB += hist[t]; if (wB == 0) continue;
        long wF = total - wB; if (wF == 0) break;
        sumB += t * hist[t];
        float mB = (float)sumB / wB;
        float mF = (float)(sum - sumB) / wF;
        float var = (float)wB * wF * (mB - mF) * (mB - mF);
        if (var > maxVar) { maxVar = var; thr = t; }
    }
    return thr;
}

static void algo_otsu(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    int thr = otsu_threshold(s_gray_a, n);
    for (int i = 0; i < n; i++) s_gray_b[i] = s_gray_a[i] >= thr ? 255 : 0;
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void adaptive_thresh(const uint8_t *in, uint8_t *out, int w, int h, int gauss)
{
    /* compute mean of neighbourhood (15x15 box, separable) */
    int16_t *rowsum = IMP_SCRATCH_I16_A();
    int r = 7;
    for (int y = 0; y < h; y++) {
        int s = 0;
        for (int x = -r; x <= r; x++) s += in[y*w + (x < 0 ? 0 : x > w-1 ? w-1 : x)];
        rowsum[y*w+0] = s;
        for (int x = 1; x < w; x++) {
            int xa = x - r - 1; if (xa < 0) xa = 0;
            int xb = x + r;     if (xb > w-1) xb = w-1;
            s += in[y*w + xb] - in[y*w + xa];
            rowsum[y*w+x] = s;
        }
    }
    int area = (2*r+1) * (2*r+1);
    for (int x = 0; x < w; x++) {
        int s = 0;
        for (int y = -r; y <= r; y++) s += rowsum[(y < 0 ? 0 : y > h-1 ? h-1 : y)*w + x];
        for (int y = 0; y < h; y++) {
            int ya = y - r - 1; if (ya < 0) ya = 0;
            int yb = y + r;     if (yb > h-1) yb = h-1;
            if (y > 0) s += rowsum[yb*w+x] - rowsum[ya*w+x];
            int mean = s / area;
            int bias = gauss ? 5 : 7;
            out[y*w+x] = in[y*w+x] >= mean - bias ? 255 : 0;
        }
    }
}

static void algo_adaptive_mean(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    adaptive_thresh(s_gray_a, s_gray_b, w, h, 0);
    gray_to_rgb(s_gray_b, rgb, w, h);
}
static void algo_adaptive_gaussian(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    adaptive_thresh(s_gray_a, s_gray_b, w, h, 1);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_triangle(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    int hist[256] = {0};
    for (int i = 0; i < n; i++) hist[s_gray_a[i]]++;
    int hmax = 0, hmax_i = 0;
    for (int i = 0; i < 256; i++) if (hist[i] > hmax) { hmax = hist[i]; hmax_i = i; }
    int end = 255; while (end > 0 && hist[end] == 0) end--;
    int best = 0, thr = hmax_i;
    /* distance from line (hmax_i,hmax)->(end,0) */
    for (int i = hmax_i; i <= end; i++) {
        int d = iabs((hmax) * (i - end) - (hmax_i - end) * (-hist[i]));
        if (d > best) { best = d; thr = i; }
    }
    for (int i = 0; i < n; i++) s_gray_b[i] = s_gray_a[i] >= thr ? 255 : 0;
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void niblack_like(uint8_t *rgb, int w, int h, int sauvola)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int r = 7;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sum = 0, sum2 = 0, count = 0;
            for (int dy = -r; dy <= r; dy += 2)
                for (int dx = -r; dx <= r; dx += 2) {
                    int xx = x + dx; int yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                    int p = s_gray_a[yy*w+xx];
                    sum += p; sum2 += p*p; count++;
                }
            if (count == 0) { s_gray_b[y*w+x] = 0; continue; }
            int mean = sum / count;
            int var  = (sum2 / count) - (mean * mean);
            if (var < 0) var = 0;
            int sd   = (int)sqrtf((float)var);
            int thr;
            if (sauvola) {
                /* Sauvola: T = mean*(1 + k*(sd/R - 1)), k=0.5, R=128 */
                int adj = (sd * 100) / 128 - 100;
                thr = mean * (100 + 50 * adj / 100) / 100;
            } else {
                thr = mean - sd / 5;
            }
            s_gray_b[y*w+x] = s_gray_a[y*w+x] >= thr ? 255 : 0;
        }
    }
    gray_to_rgb(s_gray_b, rgb, w, h);
}
static void algo_niblack(uint8_t *rgb, int w, int h)  { niblack_like(rgb, w, h, 0); }
static void algo_sauvola(uint8_t *rgb, int w, int h)  { niblack_like(rgb, w, h, 1); }

/*---------------- texture ----------------*/
static void algo_gabor(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    /* fixed 5x5 Gabor (theta=0, lambda=8, sigma=2) approximated as integer */
    static const int8_t k[25] = {
         -1,-2, 0, 2, 1,
         -3,-6, 0, 6, 3,
         -4,-8, 0, 8, 4,
         -3,-6, 0, 6, 3,
         -1,-2, 0, 2, 1
    };
    for (int y = 2; y < h - 2; y++)
        for (int x = 2; x < w - 2; x++) {
            int s = 0, idx = 0;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++)
                    s += s_gray_a[(y+dy)*w+(x+dx)] * k[idx++];
            s = (iabs(s) >> 4); if (s > 255) s = 255;
            s_gray_b[y*w+x] = (uint8_t)s;
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_lbp(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            int c = s_gray_a[y*w+x];
            int code = 0;
            code |= (s_gray_a[(y-1)*w+x-1] >= c) << 0;
            code |= (s_gray_a[(y-1)*w+x  ] >= c) << 1;
            code |= (s_gray_a[(y-1)*w+x+1] >= c) << 2;
            code |= (s_gray_a[ y   *w+x+1] >= c) << 3;
            code |= (s_gray_a[(y+1)*w+x+1] >= c) << 4;
            code |= (s_gray_a[(y+1)*w+x  ] >= c) << 5;
            code |= (s_gray_a[(y+1)*w+x-1] >= c) << 6;
            code |= (s_gray_a[ y   *w+x-1] >= c) << 7;
            s_gray_b[y*w+x] = (uint8_t)code;
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_laws_energy(uint8_t *rgb, int w, int h)
{
    /* L5E5: outer product of L5={1,4,6,4,1} and E5={-1,-2,0,2,1}, energy = abs */
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t L5[5] = {1, 4, 6, 4, 1};
    static const int8_t E5[5] = {-1,-2, 0, 2, 1};
    /* row pass with L5 */
    int16_t *row = IMP_SCRATCH_I16_A();
    for (int y = 0; y < h; y++)
        for (int x = 2; x < w - 2; x++) {
            int s = 0;
            for (int k = -2; k <= 2; k++) s += s_gray_a[y*w+x+k] * L5[k+2];
            row[y*w+x] = (int16_t)(s >> 4);
        }
    /* col pass with E5 */
    for (int y = 2; y < h - 2; y++)
        for (int x = 0; x < w; x++) {
            int s = 0;
            for (int k = -2; k <= 2; k++) s += row[(y+k)*w+x] * E5[k+2];
            int v = iabs(s); if (v > 255) v = 255;
            s_gray_b[y*w+x] = (uint8_t)v;
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/*---------------- ridge ----------------*/
static void hessian_eigs(uint8_t *rgb, int w, int h, int frangi)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    float *s_floatbuf = IMP_SCRATCH_F32();
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            float Ixx = s_gray_a[y*w+x-1] - 2.0f*s_gray_a[y*w+x] + s_gray_a[y*w+x+1];
            float Iyy = s_gray_a[(y-1)*w+x] - 2.0f*s_gray_a[y*w+x] + s_gray_a[(y+1)*w+x];
            float Ixy = (s_gray_a[(y+1)*w+x+1] - s_gray_a[(y+1)*w+x-1]
                       - s_gray_a[(y-1)*w+x+1] + s_gray_a[(y-1)*w+x-1]) * 0.25f;
            float trace = Ixx + Iyy;
            float det   = Ixx * Iyy - Ixy * Ixy;
            float tmp   = trace * trace * 0.25f - det;
            if (tmp < 0) tmp = 0;
            float root  = sqrtf(tmp);
            float l1 = trace * 0.5f + root; /* larger */
            float l2 = trace * 0.5f - root;
            float fl1 = fabsf(l1), fl2 = fabsf(l2);
            float resp;
            if (frangi) {
                if (fl1 < 1e-3f) { s_floatbuf[y*w+x] = 0; continue; }
                float Rb = fl2 / fl1;
                float S2 = l1*l1 + l2*l2;
                resp = expf(-Rb*Rb / 0.5f) * (1.0f - expf(-S2 / 200.0f));
                if (l1 > 0) resp = 0; /* only dark ridges */
            } else {
                resp = fl1 - fl2;
            }
            s_floatbuf[y*w+x] = resp;
        }
    /* normalize */
    float mx = 1e-3f;
    for (int i = 0; i < n; i++) if (s_floatbuf[i] > mx) mx = s_floatbuf[i];
    for (int i = 0; i < n; i++) {
        float v = s_floatbuf[i] * 255.0f / mx;
        s_gray_b[i] = clamp_u8((int)v);
    }
    gray_to_rgb(s_gray_b, rgb, w, h);
}
static void algo_frangi(uint8_t *rgb, int w, int h)        { hessian_eigs(rgb, w, h, 1); }
static void algo_hessian_ridge(uint8_t *rgb, int w, int h) { hessian_eigs(rgb, w, h, 0); }

static void algo_hog_vis(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    for (int i = 0; i < n; i++) s_gray_b[i] = 0;
    /* per 8x8 cell magnitude visualization */
    for (int cy = 0; cy + 8 <= h; cy += 8)
        for (int cx = 0; cx + 8 <= w; cx += 8) {
            int mag = 0;
            for (int y = cy + 1; y < cy + 7; y++)
                for (int x = cx + 1; x < cx + 7; x++) {
                    int gx = s_gray_a[y*w+x+1] - s_gray_a[y*w+x-1];
                    int gy = s_gray_a[(y+1)*w+x] - s_gray_a[(y-1)*w+x];
                    mag += iabs(gx) + iabs(gy);
                }
            int v = mag / 8; if (v > 255) v = 255;
            for (int y = cy; y < cy + 8; y++)
                for (int x = cx; x < cx + 8; x++)
                    s_gray_b[y*w+x] = (uint8_t)v;
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/*---------------- morphology ----------------*/
static void morph(const uint8_t *in, uint8_t *out, int w, int h, int dilate)
{
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            int v = in[y*w+x];
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int p = in[(y+dy)*w+(x+dx)];
                    if (dilate) { if (p > v) v = p; }
                    else        { if (p < v) v = p; }
                }
            out[y*w+x] = (uint8_t)v;
        }
}

static void algo_erode(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    morph(s_gray_a, s_gray_b, w, h, 0);
    gray_to_rgb(s_gray_b, rgb, w, h);
}
static void algo_dilate(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    morph(s_gray_a, s_gray_b, w, h, 1);
    gray_to_rgb(s_gray_b, rgb, w, h);
}
static void algo_open(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    uint8_t *tmp = s_tmp_u8_1;
    morph(s_gray_a, tmp, w, h, 0);
    morph(tmp, s_gray_b, w, h, 1);
    gray_to_rgb(s_gray_b, rgb, w, h);
}
static void algo_close(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    uint8_t *tmp = s_tmp_u8_1;
    morph(s_gray_a, tmp, w, h, 1);
    morph(tmp, s_gray_b, w, h, 0);
    gray_to_rgb(s_gray_b, rgb, w, h);
}
static void algo_morph_gradient(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    uint8_t *er = s_tmp_u8_1;
    uint8_t *di = s_gray_b;
    morph(s_gray_a, er, w, h, 0);
    morph(s_gray_a, di, w, h, 1);
    int n = w * h;
    for (int i = 0; i < n; i++) s_gray_b[i] = (uint8_t)(di[i] - er[i]);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_region_grow(uint8_t *rgb, int w, int h)
{
    /* seed from center, threshold=18 from seed value */
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    memset(s_gray_b, 0, n);
    int seed_x = w / 2, seed_y = h / 2;
    int seed_v = s_gray_a[seed_y*w + seed_x];
    /* simple 4-connect flood with stack (bounded) */
    uint32_t *stack = IMP_SCRATCH_U32();
    const int stack_max = IMP_W_MAX * IMP_H_MAX;
    int sp = 0;
    stack[sp++] = (uint32_t)(seed_y * w + seed_x);
    s_gray_b[seed_y * w + seed_x] = 255;
    while (sp > 0 && sp < stack_max - 4) {
        uint32_t idx = stack[--sp];
        int x = idx % w, y = idx / w;
        const int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; d++) {
            int nx = x + dx[d], ny = y + dy[d];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            int nidx = ny * w + nx;
            if (s_gray_b[nidx]) continue;
            if (iabs((int)s_gray_a[nidx] - seed_v) <= 18) {
                s_gray_b[nidx] = 255;
                stack[sp++] = nidx;
            }
        }
    }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

static void algo_watershed(uint8_t *rgb, int w, int h)
{
    /* simplified: threshold gradient magnitude via Otsu, mark watershed lines */
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t kx[9] = {-1,0,1, -2,0,2, -1,0,1};
    static const int8_t ky[9] = {-1,-2,-1, 0,0,0, 1,2,1};
    run_grad_pair(s_gray_a, s_gray_b, w, h, kx, ky);
    int n = w * h;
    int thr = otsu_threshold(s_gray_b, n);
    for (int i = 0; i < n; i++) s_gray_b[i] = s_gray_b[i] > thr ? 255 : 0;
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/* Unsharp-mask sharpen: 3x3 kernel with center=5, 4-neighbours=-1.
 * Operates on luminance to keep colours; the cross kernel is divisor=1
 * (sums to 1) so contrast is amplified without a brightness shift. */
static void algo_sharpen(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t k[9] = { 0, -1,  0,
                                -1,  5, -1,
                                 0, -1,  0 };
    conv3x3(s_gray_a, s_gray_b, w, h, k, 1, 0);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/* Emboss: directional 3x3 kernel that produces a pseudo-3D relief.
 * Output centred at 128 so flat regions appear mid-grey and edges show
 * as light/dark depending on gradient direction. */
static void algo_emboss(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    static const int8_t k[9] = {-2, -1,  0,
                                -1,  1,  1,
                                 0,  1,  2 };
    conv3x3(s_gray_a, s_gray_b, w, h, k, 1, 128);
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/* MSER (Maximally Stable Extremal Regions, lightweight approximation):
 * threshold the image at three levels around the Otsu point and keep
 * pixels that stay foreground at all three levels.  Visually this
 * highlights blobs whose intensity is robustly extremal. */
static void algo_mser(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    int thr = otsu_threshold(s_gray_a, n);
    int t_lo = thr - 16; if (t_lo < 0) t_lo = 0;
    int t_hi = thr + 16; if (t_hi > 255) t_hi = 255;
    for (int i = 0; i < n; i++) {
        uint8_t v = s_gray_a[i];
        /* stable bright extremal: bright at all three thresholds */
        int bright = (v >= t_lo) + (v >= thr) + (v >= t_hi);
        /* stable dark extremal: dark at all three thresholds */
        int dark   = (v <= t_lo) + (v <= thr) + (v <= t_hi);
        if (bright == 3)      s_gray_b[i] = 220;
        else if (dark == 3)   s_gray_b[i] = 80;
        else                  s_gray_b[i] = 0;
    }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/* AGAST: stricter accelerated FAST.  Uses the same 8-neighbour ring at
 * radius 3 as fast9 but with a higher contrast threshold and requiring
 * 7 of 8 (vs 6 of 8 in fast9), giving fewer but more stable corners. */
static void algo_agast(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    for (int i = 0; i < n; i++) s_gray_b[i] = (uint8_t)(s_gray_a[i] >> 1);
    static const int dx[8] = {0, 2, 3, 2, 0,-2,-3,-2};
    static const int dy[8] = {-3,-2,0, 2, 3, 2, 0,-2};
    int t = 35;
    for (int y = 3; y < h - 3; y++)
        for (int x = 3; x < w - 3; x++) {
            int p = s_gray_a[y*w+x];
            int br = 0, dk = 0;
            for (int k = 0; k < 8; k++) {
                int q = s_gray_a[(y+dy[k])*w + (x+dx[k])];
                if (q > p + t) br++;
                else if (q < p - t) dk++;
            }
            if (br >= 7 || dk >= 7) marker(s_gray_b, w, h, x, y, 255);
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/* BRIEF: a binary descriptor, not a detector.  For preview purposes we
 * detect FAST-9 keypoints (descriptors are computed around them), then
 * render the keypoint markers in red and overlay a small 8-bit "fingerprint"
 * derived from intensity comparisons in a fixed sampling pattern. */
static void algo_brief(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    for (int i = 0; i < n; i++) s_gray_b[i] = (uint8_t)(s_gray_a[i] >> 1);
    /* fixed 8 (dx, dy) test-pair pattern around each keypoint */
    static const int8_t pa[8][2] = {{-3,-3},{-3,3},{3,-3},{3,3},
                                    {-2, 0},{2, 0},{0,-2},{0, 2}};
    static const int8_t pb[8][2] = {{ 3, 3},{ 3,-3},{-3, 3},{-3,-3},
                                    { 2, 0},{-2,0},{0, 2},{0,-2}};
    static const int dx[8] = {0, 2, 3, 2, 0,-2,-3,-2};
    static const int dy[8] = {-3,-2,0, 2, 3, 2, 0,-2};
    int t = 25;
    for (int y = 3; y < h - 3; y++)
        for (int x = 3; x < w - 3; x++) {
            int p = s_gray_a[y*w+x];
            int br = 0, dk = 0;
            for (int k = 0; k < 8; k++) {
                int q = s_gray_a[(y+dy[k])*w + (x+dx[k])];
                if (q > p + t) br++;
                else if (q < p - t) dk++;
            }
            if (br < 6 && dk < 6) continue;
            /* Compute 8-bit BRIEF descriptor (1 bit per pair) and modulate
             * marker brightness by descriptor popcount so different
             * fingerprints render distinguishably. */
            uint8_t desc = 0;
            for (int k = 0; k < 8; k++) {
                int a = s_gray_a[(y+pa[k][1])*w + (x+pa[k][0])];
                int b = s_gray_a[(y+pb[k][1])*w + (x+pb[k][0])];
                if (a < b) desc |= (uint8_t)(1u << k);
            }
            uint8_t v = (uint8_t)(160 + (desc & 0x07) * 12);
            marker(s_gray_b, w, h, x, y, v);
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/* AKAZE (lightweight stand-in): KAZE/AKAZE uses non-linear diffusion +
 * Hessian-determinant scale-space.  Here we approximate with a Gaussian
 * blur followed by determinant-of-Hessian + non-maxima detection, which
 * yields a similar set of scale-stable feature points. */
static void algo_akaze(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    /* gentle smoothing to mimic NL diffusion */
    static const int8_t kg[9] = {1,2,1, 2,4,2, 1,2,1};
    conv3x3(s_gray_a, s_gray_b, w, h, kg, 16, 0);
    int n = w * h;
    for (int i = 0; i < n; i++) s_gray_a[i] = (uint8_t)(s_gray_b[i] >> 1);
    /* DoH score per pixel; threshold + simple NMS */
    for (int y = 2; y < h - 2; y++)
        for (int x = 2; x < w - 2; x++) {
            int Ixx = s_gray_b[y*w+x-1] - 2*s_gray_b[y*w+x] + s_gray_b[y*w+x+1];
            int Iyy = s_gray_b[(y-1)*w+x] - 2*s_gray_b[y*w+x] + s_gray_b[(y+1)*w+x];
            int Ixy = (s_gray_b[(y+1)*w+x+1] - s_gray_b[(y+1)*w+x-1]
                     - s_gray_b[(y-1)*w+x+1] + s_gray_b[(y-1)*w+x-1]) / 4;
            int det = Ixx * Iyy - Ixy * Ixy;
            if (det > 1500) marker(s_gray_a, w, h, x, y, 255);
        }
    gray_to_rgb(s_gray_a, rgb, w, h);
}

/* Multi-scale Laplacian-of-Gaussian blob detector: run LoG at two
 * different smoothing scales and take per-pixel max |response|, which
 * highlights blobs across a wider radius range than single-scale LoG. */
static void algo_blob_log_multiscale(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    /* scale 1: single Gaussian then Laplacian */
    static const int8_t kg[9] = {1,2,1, 2,4,2, 1,2,1};
    static const int8_t kl[9] = {0,-1,0, -1,4,-1, 0,-1,0};
    conv3x3(s_gray_a, s_gray_b, w, h, kg, 16, 0);
    conv3x3(s_gray_b, s_tmp_u8_1, w, h, kl, 1, 128);
    /* scale 2: blur twice, then Laplacian */
    conv3x3(s_gray_b, s_gray_a, w, h, kg, 16, 0);
    conv3x3(s_gray_a, s_gray_b, w, h, kl, 1, 128);
    /* per-pixel max |response - 128| -> stack into s_gray_a */
    int n = w * h;
    for (int i = 0; i < n; i++) {
        int r1 = (int)s_tmp_u8_1[i] - 128;
        int r2 = (int)s_gray_b[i]   - 128;
        if (r1 < 0) r1 = -r1;
        if (r2 < 0) r2 = -r2;
        int v = r1 > r2 ? r1 : r2;
        if (v > 127) v = 127;
        s_gray_a[i] = (uint8_t)(v * 2);   /* 0..254 */
    }
    gray_to_rgb(s_gray_a, rgb, w, h);
}

/* FAST-12: stricter FAST variant requiring all 8 ring pixels (vs >=6 in
 * fast9 / >=7 in agast) to lie above/below the centre by the threshold.
 * Equivalent of "12 of 16" on our reduced 8-neighbour ring. */
static void algo_fast12(uint8_t *rgb, int w, int h)
{
    rgb_to_gray(rgb, s_gray_a, w, h);
    int n = w * h;
    for (int i = 0; i < n; i++) s_gray_b[i] = (uint8_t)(s_gray_a[i] >> 1);
    static const int dx[8] = {0, 2, 3, 2, 0,-2,-3,-2};
    static const int dy[8] = {-3,-2,0, 2, 3, 2, 0,-2};
    int t = 30;
    for (int y = 3; y < h - 3; y++)
        for (int x = 3; x < w - 3; x++) {
            int p = s_gray_a[y*w+x];
            int br = 0, dk = 0;
            for (int k = 0; k < 8; k++) {
                int q = s_gray_a[(y+dy[k])*w + (x+dx[k])];
                if (q > p + t) br++;
                else if (q < p - t) dk++;
            }
            if (br == 8 || dk == 8) marker(s_gray_b, w, h, x, y, 255);
        }
    gray_to_rgb(s_gray_b, rgb, w, h);
}

/*---------------- dispatch ----------------*/
typedef void (*algo_fn)(uint8_t *, int, int);

static const struct {
    algo_fn      fn;
    imgproc_info_t info;
} k_algos[IMGPROC_ALGO_COUNT] = {
    [IMGPROC_PASSTHROUGH]       = { algo_passthrough,        {"passthrough", 0} },
    [IMGPROC_GRAYSCALE]         = { algo_grayscale,          {"grayscale", 0} },
    [IMGPROC_INVERT]            = { algo_invert,             {"invert", 0} },
    [IMGPROC_HIST_EQ]           = { algo_hist_eq,            {"hist_eq", 0} },
    [IMGPROC_GAUSSIAN_3]        = { algo_gaussian_3,         {"gaussian_3", 0} },
    [IMGPROC_GAUSSIAN_5]        = { algo_gaussian_5,         {"gaussian_5", 0} },
    [IMGPROC_MEAN_3]            = { algo_mean_3,             {"mean_3", 0} },
    [IMGPROC_MEDIAN_3]          = { algo_median_3,           {"median_3", 0} },
    [IMGPROC_BILATERAL]         = { algo_bilateral,          {"bilateral", 0} },
    [IMGPROC_SOBEL]             = { algo_sobel,              {"sobel", 1} },
    [IMGPROC_ROBERTS]           = { algo_roberts,            {"roberts", 1} },
    [IMGPROC_PREWITT]           = { algo_prewitt,            {"prewitt", 1} },
    [IMGPROC_SCHARR]            = { algo_scharr,             {"scharr", 1} },
    [IMGPROC_KIRSCH]            = { algo_kirsch,             {"kirsch", 1} },
    [IMGPROC_FREI_CHEN]         = { algo_frei_chen,          {"frei_chen", 1} },
    [IMGPROC_CANNY]             = { algo_canny,              {"canny", 1} },
    [IMGPROC_MARR_HILDRETH]     = { algo_marr_hildreth,      {"marr_hildreth", 1} },
    [IMGPROC_LAPLACIAN]         = { algo_laplacian,          {"laplacian", 1} },
    [IMGPROC_DOG]               = { algo_dog,                {"dog", 2} },
    [IMGPROC_LOG]               = { algo_log,                {"log", 2} },
    [IMGPROC_DOH]               = { algo_doh,                {"doh", 2} },
    [IMGPROC_HARRIS]            = { algo_harris,             {"harris", 3} },
    [IMGPROC_SHI_TOMASI]        = { algo_shi_tomasi,         {"shi_tomasi", 3} },
    [IMGPROC_FAST9]             = { algo_fast9,              {"fast9", 3} },
    [IMGPROC_OTSU]              = { algo_otsu,               {"otsu", 4} },
    [IMGPROC_ADAPTIVE_MEAN]     = { algo_adaptive_mean,      {"adaptive_mean", 4} },
    [IMGPROC_ADAPTIVE_GAUSSIAN] = { algo_adaptive_gaussian,  {"adaptive_gaussian", 4} },
    [IMGPROC_TRIANGLE]          = { algo_triangle,           {"triangle", 4} },
    [IMGPROC_NIBLACK]           = { algo_niblack,            {"niblack", 4} },
    [IMGPROC_SAUVOLA]           = { algo_sauvola,            {"sauvola", 4} },
    [IMGPROC_GABOR]             = { algo_gabor,              {"gabor", 5} },
    [IMGPROC_LBP]               = { algo_lbp,                {"lbp", 5} },
    [IMGPROC_LAWS_ENERGY]       = { algo_laws_energy,        {"laws_energy", 5} },
    [IMGPROC_FRANGI]            = { algo_frangi,             {"frangi", 6} },
    [IMGPROC_HESSIAN_RIDGE]     = { algo_hessian_ridge,      {"hessian_ridge", 6} },
    [IMGPROC_HOG_VIS]           = { algo_hog_vis,            {"hog_vis", 6} },
    [IMGPROC_ERODE]             = { algo_erode,              {"erode", 7} },
    [IMGPROC_DILATE]            = { algo_dilate,             {"dilate", 7} },
    [IMGPROC_OPEN]              = { algo_open,               {"open", 7} },
    [IMGPROC_CLOSE]             = { algo_close,              {"close", 7} },
    [IMGPROC_MORPH_GRADIENT]    = { algo_morph_gradient,     {"morph_gradient", 7} },
    [IMGPROC_REGION_GROW]       = { algo_region_grow,        {"region_grow", 7} },
    [IMGPROC_WATERSHED]         = { algo_watershed,          {"watershed", 7} },
    [IMGPROC_SHARPEN]           = { algo_sharpen,            {"sharpen", 0} },
    [IMGPROC_EMBOSS]            = { algo_emboss,             {"emboss", 5} },
    [IMGPROC_MSER]              = { algo_mser,               {"mser", 4} },
    [IMGPROC_AGAST]             = { algo_agast,              {"agast", 3} },
    [IMGPROC_BRIEF]             = { algo_brief,              {"brief", 3} },
    [IMGPROC_AKAZE]             = { algo_akaze,              {"akaze", 3} },
    [IMGPROC_BLOB_LOG_MULTISCALE] = { algo_blob_log_multiscale, {"blob_log_multiscale", 2} },
    [IMGPROC_FAST12]            = { algo_fast12,             {"fast12", 3} },
};

const imgproc_info_t *imgproc_get_info(uint8_t id)
{
    if (id >= IMGPROC_ALGO_COUNT) return NULL;
    return &k_algos[id].info;
}

uint32_t imgproc_apply(uint8_t algo_id, uint8_t *rgb888, int width, int height, int active_height)
{
    if (algo_id >= IMGPROC_ALGO_COUNT || k_algos[algo_id].fn == NULL) {
        return 0;
    }
    if (active_height > height) active_height = height;
    if (width > IMP_W_MAX || active_height > IMP_H_MAX) return 0;

    uint32_t t0 = ifx_time_get_ms_f() * 1000.0f;
    k_algos[algo_id].fn(rgb888, width, active_height);
    uint32_t t1 = ifx_time_get_ms_f() * 1000.0f;

    /* keep padding rows zeroed */
    if (active_height < height) {
        memset(rgb888 + active_height * width * 3, 0,
               (size_t)(height - active_height) * width * 3);
    }
    return t1 - t0;
}
