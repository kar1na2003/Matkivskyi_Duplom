/*
 * image_processing.c — host-side timing telemetry variables used by
 * lcd_task's draw() path. Populated there and available to debug tools.
 *
 * Kept in a separate translation unit so linker --gc-sections keeps them
 * out of the image when no debugger/watch references them.
 */

float prep_wait_buf          = 0.0f;
float prep_YUV422_to_bgr565   = 0.0f;
float prep_bgr565_to_disp     = 0.0f;
float prep_RGB565_to_RGB888   = 0.0f;
