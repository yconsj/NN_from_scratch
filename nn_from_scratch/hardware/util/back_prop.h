#ifndef BACK_PROP_H
#define BACK_PROP_H
#include <stdint.h>
#include "activation_functions.h"

void fc_back_prop(float *input_gradient, float *net_inputs, float *weights,
                  int input_size, int net_inputs_size, ActivationFunc activation_func, ActivationFunc activation_func_deriv,
                  float *gradient_weights, float *gradient_biases);

float *fc_light_back_prop(float *input_gradient, float *weights,
                          int input_size, int output_layer_size, uint8_t *deriv_activation_val);

void fc_specific_back_prop(float *input_gradient, float *net_inputs,
                           int input_size, ActivationFunc activation_func,
                           float *gradient_weights, float *gradient_biases, int n_neurons);

typedef void (*SpecificBackProp)(float *, float *, int, float *, float *, int);
typedef void (*BackProp)(float *, float *, float *, int, int, float *, float *);

BackProp get_fc_back_prop_variant(enum ActivationType activationType);
SpecificBackProp get_fc_specific_back_prop_variant(enum ActivationType activationType);
#define GENERATE_FC_BACK_PROP_PROTOTYPE_VARIANTS(act, func, func_deriv)               \
    void fc_back_prop_##act(float *input_gradient, float *net_inputs, float *weights, \
                            int input_size, int net_inputs_size,                      \
                            float *gradient_weights, float *gradient_biases);

#define X(act, func, func_deriv) GENERATE_FC_BACK_PROP_PROTOTYPE_VARIANTS(act, func, func_deriv)
ACTIVATION_MACRO_LIST
#undef X

#define GENERATE_FC_SPECIFIC_BACK_PROP_PROTOTYPE_VARIANTS(act, func, func_deriv) \
    void fc_specific_back_prop_##act(float *input_gradient, float *net_inputs,   \
                                     int input_size, float *gradient_weights, float *gradient_biases, int n_neurons);

#define X(act, func, func_deriv) GENERATE_FC_SPECIFIC_BACK_PROP_PROTOTYPE_VARIANTS(act, func, func_deriv)
ACTIVATION_MACRO_LIST
#undef X

#endif
