/* MODUSMATE_STUMP_MODEL_H
 * Deterministic constant predictor: every IMAI_compute() call returns
 * class_id=0 (Rock) with confidence 1.0. Used by host smoke tests.
 */
#ifndef MODUSMATE_STUMP_MODEL_H
#define MODUSMATE_STUMP_MODEL_H

#include <stdint.h>

#define IMAI_DATAIN_RANK   3
#define IMAI_DATAIN_SHAPE  (((int[]){3, 320, 320}))
#define IMAI_DATAIN_COUNT  307200
#define IMAI_DATAIN_TYPE   uint8_t

#define IMAI_DATAOUT_RANK  2
#define IMAI_DATAOUT_SHAPE (((int[]){5, 8}))
#define IMAI_DATAOUT_COUNT 40
#define IMAI_DATAOUT_TYPE  float

#define IMAI_RET_SUCCESS   0
#define IMAI_RET_ERROR    -1
#define IMAI_MAX_MTB_MODELS 1

typedef struct mtb_ml_model_s mtb_ml_model_t;
extern int32_t          IMAI_mtb_models_count;
extern mtb_ml_model_t  *IMAI_mtb_models[IMAI_MAX_MTB_MODELS];

int  IMAI_init(void);
void IMAI_finalize(void);
void IMAI_compute(const uint8_t *restrict datain, float *restrict dataout);
void IMAI_mtb_models_print_info(void);
void IMAI_mtb_models_profile_log(void);

#endif /* MODUSMATE_STUMP_MODEL_H */
