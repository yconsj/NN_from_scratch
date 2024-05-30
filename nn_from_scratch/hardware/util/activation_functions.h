#ifndef ACTIVATION_TYPE_H
#define ACTIVATION_TYPE_H
enum ActivationType
{
    LINEAR,
    RELU
};

typedef float (*ActivationFunc)(float);

ActivationFunc get_activation_func(enum ActivationType activationType);
ActivationFunc get_activation_func_deriv(enum ActivationType activationType);

float linear(float x);
float relu(float x);
float linear_deriv(float x);
float relu_deriv(float x);

#define LINEAR_MACRO(x) (x)
#define RELU_MACRO(x) ((x > 0) ? x : 0)
#define LINEAR_DERIV_MACRO(x) (1)
#define RELU_DERIV_MACRO(x) ((x > 0) ? 1 : 0)

#define ACTIVATION_MACRO_LIST                   \
    X(LINEAR, LINEAR_MACRO, LINEAR_DERIV_MACRO) \
    X(RELU, RELU_MACRO, RELU_DERIV_MACRO)
#endif
