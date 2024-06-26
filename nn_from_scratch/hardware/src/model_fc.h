#ifndef MODEL_FC_H
#define MODEL_FC_H

#include "../util/model_binding.h"
#include "../util/model_gradients.h"

void fc_model_train(Model *model, float (*samples_x)[model->output_size], float (*samples_y)[model->output_size]);
float *fc_model_predict(Model *model, float *input);

#endif
