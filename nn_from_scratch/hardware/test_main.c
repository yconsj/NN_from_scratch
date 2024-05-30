

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "include/nn_from_scratch.h"
#include "model/model.h"
#include "data/eqcheck_data.h"
#include "data/ft_data.h"
#include "data/true_data.h"

/* calculates the loss for the 168 test samples*/
void compare_true(Model *model)
{

    float sum = 0;
    for (int i = FT_N_SAMPLES - 168; i < FT_N_SAMPLES; i++)
    {
        float *input = ft_samples_x[i];
        float *output = fc_model_predict(model, input);

        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            float t = (output[j] - ft_samples_y[i][j]);
            t *= t;
            sum += t;
        }
        free(output);
    }
    printf("MSE error: %f \n", sum / 168);
}
/* Checks that the model is outputting correct values (before any training) - used to test correctness of forward propagation */
void eqcheck(Model *model)
{
    printf("start eqcheck..\n");

    for (int i = 0; i < EQCHECK_N_SAMPLES; i++)
    {
        float *input = eqcheck_samples_x[i];
        float *output = fc_model_predict(model, input);
        float tolerance = 0.0001;
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            if (fabs(output[j] - eqcheck_samples_y[i][j]) > tolerance)
            {
                printf("FAILED: eqcheck for sample, expected: %f but predicted: %f\n", eqcheck_samples_y[i][j], output[j]);
                break;
            }
        }
        free(output);
    }
    printf("eqcheck completed! \n");
}
void memory_tester(Model *model)
{

    reset_memory_tracking();
    printf("Memory stats for training for the whole network \n");
    fc_model_train(model, ft_samples_x, ft_samples_y);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("Memory stats for training for the first layer \n");
    fc_model_train_layer(model, ft_samples_x, ft_samples_y, 0);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("Memory stats for training for the last layer \n");
    fc_model_train_layer(model, ft_samples_x, ft_samples_y, 2);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("Memory stats for the second layer \n");
    fc_model_train_layer(model, ft_samples_x, ft_samples_y, 1);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("Memory stats for training for the second layer, two last weights \n");
    fc_model_train_partial_layer(model, ft_samples_x, ft_samples_y, 1, 2, 2);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("\n Completed memory test \n");
    return;
}
void trainer(Model *model)
{
    int batches = 13;
    for (int i = 0; i < batches; i++)
    {
        /* Enable one of the functions */
        fc_model_train(model, ft_samples_x + (i * BATCH_SIZE), ft_samples_y + (i * BATCH_SIZE));

        // fc_model_train_layer(model, ft_samples_x + (i * BATCH_SIZE), ft_samples_y + (i * BATCH_SIZE), 1);

        // fc_model_train_partial_layer(model, ft_samples_x + (i * BATCH_SIZE), ft_samples_y + (i * BATCH_SIZE), 1,1,0);
    }
}
int main()
{
    // Test data input to output from eqcheck
    Model *model = createAndSetModel(N_LAYERS, INPUT_SIZE, OUTPUT_SIZE, layers_size, layers_weights, layers_biases, layers_activation);
    eqcheck(model);

    compare_true(model);
    printf("Start training... \n \n");
    trainer(model);

    compare_true(model);

    // perform memory testing
    printf("Starting memory tests... \n\n");
    memory_tester(model);
    return 0;
}