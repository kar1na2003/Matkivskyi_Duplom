/* MODUSMATE_STUMP_MODEL_C
 * Deterministic stub: always answers class 0 with conf=1.0.
 */
#include <string.h>
#include "model.h"

#ifndef CY_SECTION
#define CY_SECTION(x)
#endif
#ifndef EXPAND_AND_STRINGIFY
#define EXPAND_AND_STRINGIFY(x) #x
#endif

int32_t         IMAI_mtb_models_count = 0;
mtb_ml_model_t *IMAI_mtb_models[IMAI_MAX_MTB_MODELS];

#define MAX_DET 5
#define OUT_IDX(row, det) ((row) * MAX_DET + (det))

void IMAI_compute(const uint8_t *restrict datain, float *restrict dataout)
{
    (void)datain;
    memset(dataout, 0, sizeof(float) * 40);
    dataout[OUT_IDX(0, 0)] = 160.0f;   /* bbox cx */
    dataout[OUT_IDX(1, 0)] = 120.0f;   /* bbox cy */
    dataout[OUT_IDX(2, 0)] = 100.0f;   /* bbox w  */
    dataout[OUT_IDX(3, 0)] = 100.0f;   /* bbox h  */
    dataout[OUT_IDX(4, 0)] = 1.0f;     /* class 0 score = 1.0 */
    dataout[OUT_IDX(5, 0)] = 0.0f;
    dataout[OUT_IDX(6, 0)] = 0.0f;
    dataout[OUT_IDX(7, 0)] = 1.0f;     /* slot active */
}

int  IMAI_init(void)      { IMAI_mtb_models_count = 0; return IMAI_RET_SUCCESS; }
void IMAI_finalize(void)  { IMAI_mtb_models_count = 0; }
void IMAI_mtb_models_print_info(void) {}
void IMAI_mtb_models_profile_log(void) {}
