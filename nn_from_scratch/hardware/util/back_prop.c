#include "back_prop.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "config.h"
#include <math.h>
/* Back propagation function for one layer, updates the output neurons with gradients
    @param input_gradient: pointer to input gradients (going backwards)
    @param net_inputs: pointer to stored input neruon values, which gradients will be stored in
    @param weights: pointer to weights between input/output
    @param input_size: size of input
    @param net_inputs_size: size of net_inputs
    @param activation_func: activation function for the output neruons
    @param activation_func_deriv: derivative activation function for the output neruons
    @param gradient_weights: pointer to gradient weights for the layer to be accumilated in
    @param gradient_biases: pointer to gradient biases for the layer to accumilated in
    @return nothing
*/
void fc_back_prop(float *input_gradient, float *net_inputs, float *weights,
                  int input_size, int net_inputs_size, ActivationFunc activation_func, ActivationFunc activation_func_deriv,
                  float *gradient_weights, float *gradient_biases)
{
    float *temp = calloc(net_inputs_size, sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        float gradient = input_gradient[i];
        // add gradient to bias array
        gradient_biases[i] += gradient;

        // weight gradients
        for (int j = 0; j < net_inputs_size; j++)
        {
            // adding gradient weights
            gradient_weights[i + j * input_size] += gradient * activation_func(net_inputs[j]);
            // gradient neuron
            temp[j] += weights[i + j * input_size] * gradient;
        }
    }

    // gradients for next layer
    for (int j = 0; j < net_inputs_size; j++)
    {
        net_inputs[j] = temp[j] * activation_func_deriv(net_inputs[j]);
    }

    free(temp);
    return;
}

#define GENERATE_FC_BACK_PROP_VARIANTS(act, func, func_deriv)                           \
    void fc_back_prop_##act(float *input_gradient, float *net_inputs, float *weights,   \
                            int input_size, int net_inputs_size,                        \
                            float *gradient_weights, float *gradient_biases)            \
    {                                                                                   \
        float *temp = calloc(net_inputs_size, sizeof(float));                           \
        for (int i = 0; i < input_size; i++)                                            \
        {                                                                               \
            float gradient = input_gradient[i];                                         \
            gradient_biases[i] += gradient;                                             \
            for (int j = 0; j < net_inputs_size; j++)                                   \
            {                                                                           \
                gradient_weights[i + j * input_size] += gradient * func(net_inputs[j]); \
                temp[j] += weights[i + j * input_size] * gradient;                      \
            }                                                                           \
        }                                                                               \
        for (int j = 0; j < net_inputs_size; j++)                                       \
        {                                                                               \
            net_inputs[j] = temp[j] * func_deriv(net_inputs[j]);                        \
        }                                                                               \
        free(temp);                                                                     \
        return;                                                                         \
    }

#define X(act, func, func_deriv) GENERATE_FC_BACK_PROP_VARIANTS(act, func, func_deriv)
ACTIVATION_MACRO_LIST
#undef X

#define X(act, func, func_deriv) \
    case act:                    \
        return fc_back_prop_##act;

BackProp get_fc_back_prop_variant(enum ActivationType activationType)
{
    switch (activationType)
    {
        ACTIVATION_MACRO_LIST
    default:
        printf("Error unknown activation type: defaulting to LINEAR\n");
        return fc_back_prop_LINEAR;
    }
}
#undef X
/* Will backpropagate under the partial training conditions. Meaning it uses the derivative values.
    @return gradients when backpropagating to the output layer
*/
float *fc_light_back_prop(float *input_gradient, float *weights,
                          int input_size, int output_layer_size, uint8_t *deriv_activation_val)
{
    float *output = calloc(output_layer_size, sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        float gradient = input_gradient[i];
        for (int j = 0; j < output_layer_size; j++)
        {
            output[j] += weights[i + j * input_size] * gradient;
        }
    }
    for (int j = 0; j < output_layer_size; j++)
    {
        output[j] *= deriv_activation_val[j];
    }

    return output;
}

/* Backprop to calculate gradient bias given layer biase and chosen weights.
   Will not calculate gradients for the next layer!
 */
void fc_specific_back_prop(float *input_gradient, float *net_inputs,
                           int layer_size, ActivationFunc activation_func,
                           float *gradient_weights, float *gradient_biases, int n_neurons)
{
    for (int i = 0; i < layer_size; i++)
    {
        float gradient = input_gradient[i];
        // add gradient to bias array
        gradient_biases[i] += gradient;

        for (int j = 0; j < n_neurons; j++)
        {
            // adding gradient weights
            gradient_weights[i + j * layer_size] += gradient * activation_func(net_inputs[j]);
        }
    }
    return;
}

#define GENERATE_FC_SPECIFIC_BACK_PROP_VARIANTS(act, func, func_deriv)                                               \
    void fc_specific_back_prop_##act(float *input_gradient, float *net_inputs,                                       \
                                     int layer_size, float *gradient_weights, float *gradient_biases, int n_neurons) \
    {                                                                                                                \
        for (int i = 0; i < layer_size; i++)                                                                         \
        {                                                                                                            \
            float gradient = input_gradient[i];                                                                      \
            gradient_biases[i] += gradient;                                                                          \
            for (int j = 0; j < n_neurons; j++)                                                                      \
            {                                                                                                        \
                gradient_weights[i + j * layer_size] += gradient * func(net_inputs[j]);                              \
            }                                                                                                        \
        }                                                                                                            \
        return;                                                                                                      \
    }

#define X(act, func, func_deriv) GENERATE_FC_SPECIFIC_BACK_PROP_VARIANTS(act, func, func_deriv)
ACTIVATION_MACRO_LIST
#undef X

#define X(act, func, func_deriv) \
    case act:                    \
        return fc_specific_back_prop_##act;
SpecificBackProp get_fc_specific_back_prop_variant(enum ActivationType activationType)
{
    switch (activationType)
    {
        ACTIVATION_MACRO_LIST
    default:
        printf("Error unknown activation type: defaulting to LINEAR\n");
        return fc_specific_back_prop_LINEAR;
    }
}
#undef X