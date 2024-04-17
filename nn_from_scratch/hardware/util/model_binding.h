#include "activation_functions.h"

typedef struct
{
    int n_layers;
    int input_size;
    int output_size;
    int *layers_size;
    float **layers_weights;
    float **layers_biases;
    enum ActivationType *layers_activation;
} ModelPtr;

void setModel(ModelPtr *model, int n_layers, int input_size, int output_size, int *layers_size, float **layers_weights,
              float **layers_biases, enum ActivationType *layers_activation);

ModelPtr *createAndSetModel(int n_layers, int input_size, int output_size, int *layers_size, float **layers_weights,
                            float **layers_biases, enum ActivationType *layers_activation);

void destroyModel(ModelPtr *model);