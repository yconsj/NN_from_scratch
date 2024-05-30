
#include "activation_functions.h"
#include <stdint.h>
extern float *fc_forward_prop(float *input, float *layer_weights, float *layer_biases, int input_size,
                              int output_size, ActivationFunc activation_func);

extern float *fc_forward_prop_t(float *input, int input_size, float *output, int output_size, float *weights, float *biases, ActivationFunc activation_func);

/* Generate activation function variants prototypes */
typedef float *(*ForwardProp)(float *, float *, float *, int, int);
typedef float *(*ForwardPropT)(float *, int, float *, int, float *, float *);
ForwardProp get_fc_forward_prop_variant(enum ActivationType activationType);
ForwardPropT get_fc_forward_prop_t_variant(enum ActivationType activationType);

#define GENERATE_FC_FORWARD_PROP_PROTOTYPE_VARIANTS(act, func, func_deriv) \
    float *fc_forward_prop_##act(float *input, float *weights, float *biases, int input_size, int output_size);

#define GENERATE_FC_FORWARD_PROP_T_PROTOTYPE_VARIANTS(act, func, func_deriv) \
    float *fc_forward_prop_t_##act(float *input, int input_size, float *output, int output_size, float *weights, float *biases);

#define X(act, func, func_deriv) GENERATE_FC_FORWARD_PROP_PROTOTYPE_VARIANTS(act, func, func_deriv)
ACTIVATION_MACRO_LIST
#undef X
#define X(act, func, func_deriv) \
    GENERATE_FC_FORWARD_PROP_T_PROTOTYPE_VARIANTS(act, func, func_deriv)
ACTIVATION_MACRO_LIST
#undef X