#include <math.h>
#include <stdio.h>
float MSE(float *predicted, float *actual, int size)
{
    float error = 0.0;
    for (int i = 0; i < size; i++)
    {
        error += (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
    }
    return error / size;
}
float MSE_derivative(float *predicted, float *actual, int size)
{
    float error = 0.0;
    for (int i = 0; i < size; i++)
    {
        if (isnan(fabs(predicted[i] - actual[i])))
        {
            printf("%f, %f \n", predicted[i], actual[i]);
        }
        error += 2 * fabs(predicted[i] - actual[i]);
    }
    return error / size;
}
