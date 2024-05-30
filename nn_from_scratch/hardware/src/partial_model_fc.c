#include <stdlib.h>
#include <string.h>
#include "../util/forward_prop.h"
#include "../util/activation_functions.h"
#include "../util/back_prop.h"
#include "../util/loss_functions.h"
#include "partial_model_fc.h"
#include "../util/config.h"
#include <stdio.h>

/* function to calculate gradients under partial training conditions */
void partial_calc_gradients(float *input, Model *model, int target_layer, int n_weights, int offset, float *actual, PartialGradients *gradients)
{

    float *curr_in = input;
    float *output;
    int size = model->input_size;
    ForwardPropT forward_prop = fc_forward_prop_t_LINEAR;

    for (int i = 0; i < model->n_layers; i++)
    {
        output = (float *)malloc(model->layers_size[i] * sizeof(float)); // allocate output
        /* forward propagate, store needed data otherwise free */
        output = forward_prop(curr_in, size, output, model->layers_size[i], model->layers_weights[i], model->layers_biases[i]);
        if (i == target_layer) // store neuron if at target layer
        {

            memcpy(gradients->net_input, curr_in + offset, n_weights * sizeof(float));
            if (target_layer != 0)
            {
                free(curr_in);
            }
        }
        else if (i != 0 && (i < target_layer || i > target_layer)) // free if below target layer or larget than target, since not needed
        {
            free(curr_in);
        }

        if (i >= target_layer)
        { // else only store derivative of the input
            ActivationFunc func_deriv = get_activation_func_deriv(model->layers_activation[i]);
            for (int j = 0; j < model->layers_size[i]; j++)
            {
                gradients->deriv_activations[i - target_layer][j] = (uint8_t)func_deriv(output[j]);
            }
        }
        curr_in = output;
        size = model->layers_size[i];
        forward_prop = get_fc_forward_prop_t_variant(model->layers_activation[i]);
    }
    ActivationFunc func = get_activation_func(model->layers_activation[model->n_layers - 1]);
    output = (float *)malloc(sizeof(float) * model->output_size);
    for (int i = 0; i < model->output_size; i++)
    {
        output[i] = func(curr_in[i]);
    }

    // calculate loss derivative
    float loss_deriv = MSE_derivative(curr_in, actual, model->output_size);
    free(output);
    ActivationFunc func_deriv = get_activation_func_deriv(model->layers_activation[model->n_layers - 1]);

    // get initial gradient
    for (int i = 0; i < model->output_size; i++)
    {
        curr_in[i] = loss_deriv * func_deriv(curr_in[i]);
    }
    // perform packprop using the backprop that uses the stored derivative activation values until target layer
    for (int i = model->n_layers - 1; i > target_layer; i--)
    {

        float *output = fc_light_back_prop(curr_in, model->layers_weights[i], model->layers_size[i],
                                           model->layers_size[i - 1], gradients->deriv_activations[i - target_layer - 1]);
        free(curr_in);
        curr_in = output;
    }
    // Apply last backprop, using 'normal backprop' to calculate the gradient to target weights.
    SpecificBackProp specific_back_prop;
    if (target_layer != 0)
    {
        specific_back_prop = get_fc_specific_back_prop_variant(model->layers_activation[target_layer - 1]);
    }
    else
    {
        // special case if the target layer in the first layer
        specific_back_prop = fc_specific_back_prop_LINEAR;
    }
    specific_back_prop(curr_in, gradients->net_input, model->layers_size[target_layer], gradients->weights, gradients->biases, n_weights);

    free(curr_in);

    return;
}

/* Apply gradients to a layer, given specific neurons*/
void fc_apply_specific_gradients(Model *model, int layer, int layer_size, int n_weights, int offset, PartialGradients *gradients)
{
    for (int i = 0; i < layer_size; i++)
    {
        model->layers_biases[layer][i] -= LEARNING_RATE * (gradients->biases[i] / BATCH_SIZE);
        for (int j = 0; j < n_weights; j++)
        {
            model->layers_weights[layer][i + (j + offset) * layer_size] -= LEARNING_RATE * (gradients->weights[i + j * layer_size] / BATCH_SIZE);
        }
    }
}

/* train a part of a layer - stated by target layer, the number of weights and the offset. Biases will always also be trained for the target layer
    @param model: pointer to model
    @param samples_x: input samples
    @param samples_y: expected output samples
    @param target_layer: target layer of weights to be trained
    @param n_weights: number of weights to be trained pr neuron in the layer.
    @param offset: offset for the number of weights to be trained
 */
void fc_model_train_partial_layer(Model *model, float (*samples_x)[model->output_size], float (*samples_y)[model->output_size],
                                  int target_layer, int n_weights, int offset)
{
    if ((target_layer == 0 && n_weights != 1 && offset != 0) || target_layer < 0 || offset < 0 || n_weights < 1)
    {
        printf("Invalid arguments for partial layer training! \n");
        return;
    }
    else if (n_weights + offset > model->layers_size[target_layer - 1])
    {
        printf("Invalid arguments for partial layer training! \n");
        return;
    }

    PartialGradients *gradients = (PartialGradients *)allocate_partial_gradients(model, target_layer, n_weights);

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        partial_calc_gradients(samples_x[i], model, target_layer, n_weights, offset, samples_y[i], gradients);
    }

    // apply the calculated gradient to the specific layer
    fc_apply_specific_gradients(model, target_layer, model->layers_size[target_layer], n_weights, offset, gradients);
    free_partial_gradients(gradients, model, target_layer);
}

/* train a specific layer*/
void fc_model_train_layer(Model *model, float (*samples_x)[model->output_size], float (*samples_y)[model->output_size],
                          int target_layer)
{
    if (target_layer >= model->n_layers)
    {
        printf("Invalid arguments for layer training! \n");
        return;
    }
    int offset = 0;
    int n_neurons;
    if (target_layer == 0)
    {
        n_neurons = 1;
    }
    else
    {
        n_neurons = model->layers_size[target_layer - 1];
    }

    PartialGradients *gradients = (PartialGradients *)allocate_partial_gradients(model, target_layer, n_neurons);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        partial_calc_gradients(samples_x[i], model, target_layer, n_neurons, offset, samples_y[i], gradients);
    }
    fc_apply_specific_gradients(model, target_layer, model->layers_size[target_layer], n_neurons, offset, gradients);
    free_partial_gradients(gradients, model, target_layer);
}
