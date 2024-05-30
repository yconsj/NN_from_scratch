#include <stdint.h>
#include "forward_prop.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "config.h"
/* forward propagation allocates output memory
    @result returns activation result for output for layer, and allocate memory from each layer

    @param input: input pointer for the layer
    @param weights: weights pointer for the layer
    @param biases: biases pointer  for the layer
    @param input_size: size of the input for the layer
    @param output_size: size of the output for the layer
    @param activation_func: activation function for the input layer
*/
float *fc_forward_prop(float *input, float *weights, float *biases,
                       int input_size, int output_size, ActivationFunc activation_func)
{
    float *output = (float *)malloc(output_size * sizeof(float));
    for (int i = 0; i < output_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < input_size; j++)
        {
            // indexing for correct weight
            sum += input[j] * weights[i + j * output_size];
        }
        // add bias
        sum += biases[i];
        output[i] = activation_func(sum);
    }
    return output;
}

/* forward propagation used when training. Will calculate net_inputs for output layer
    @result returns output pointer back

    @param input: input pointer for the layer
    @param input_size: size of the input for the layer
    @param output: pointer to where output will be stored
    @param output_size: size of the output for the layer
    @param weights: weights pointer for the layer
    @param biases: biases pointer  for the layer
    @param activation_func: activation function for the input layer
*/
float *fc_forward_prop_t(float *input, int input_size, float *output, int output_size, float *weights, float *biases, ActivationFunc activation_func)
{
    for (int i = 0; i < output_size; i++)
    {
        // get sum
        float sum = 0;
        for (int j = 0; j < input_size; j++)
        {
            // indexing for correct weight
            sum += activation_func(input[j]) * weights[i + j * output_size];
        }
        // add bias
        sum += biases[i];
        output[i] = sum;
    }
    return output;
}

#define GENERATE_FC_FORWARD_PROP_VARIANTS(act, func, func_deriv)              \
    float *fc_forward_prop_##act(float *input, float *weights, float *biases, \
                                 int input_size, int output_size)             \
    {                                                                         \
        float *output = (float *)malloc(output_size * sizeof(float));         \
        for (int i = 0; i < output_size; i++)                                 \
        {                                                                     \
            float sum = 0;                                                    \
            for (int j = 0; j < input_size; j++)                              \
            {                                                                 \
                sum += input[j] * weights[i + j * output_size];               \
            }                                                                 \
            sum += biases[i];                                                 \
            output[i] = func(sum);                                            \
        }                                                                     \
        return output;                                                        \
    }

#define GENERATE_FC_FORWARD_PROP_T_VARIANTS(act, func, func_deriv)                                                              \
    float *fc_forward_prop_t_##act(float *input, int input_size, float *output, int output_size, float *weights, float *biases) \
    {                                                                                                                           \
        for (int i = 0; i < output_size; i++)                                                                                   \
        {                                                                                                                       \
            float sum = 0;                                                                                                      \
            for (int j = 0; j < input_size; j++)                                                                                \
            {                                                                                                                   \
                sum += func(input[j]) * weights[i + j * output_size];                                                           \
            }                                                                                                                   \
            sum += biases[i];                                                                                                   \
            output[i] = sum;                                                                                                    \
        }                                                                                                                       \
        return output;                                                                                                          \
    }

#define X(act, func, func_deriv) GENERATE_FC_FORWARD_PROP_VARIANTS(act, func, func_deriv)
ACTIVATION_MACRO_LIST
#undef X

#define X(act, func, func_deriv) GENERATE_FC_FORWARD_PROP_T_VARIANTS(act, func, func_deriv)
ACTIVATION_MACRO_LIST
#undef X

#define X(act, func, func_deriv) \
    case act:                    \
        return fc_forward_prop_##act;
ForwardProp get_fc_forward_prop_variant(enum ActivationType activationType)
{
    switch (activationType)
    {
        ACTIVATION_MACRO_LIST
    default:
        printf("Error unknown activation type: defaulting to LINEAR\n");
        return fc_forward_prop_LINEAR;
    }
}
#undef X

#define X(act, func, func_deriv) \
    case act:                    \
        return fc_forward_prop_t_##act;
ForwardPropT get_fc_forward_prop_t_variant(enum ActivationType activationType)
{
    switch (activationType)
    {
        ACTIVATION_MACRO_LIST
    default:
        printf("Error unknown activation type: defaulting to LINEAR\n");
        return fc_forward_prop_t_LINEAR;
    }
}
#undef X
