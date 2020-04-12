
#ifndef TRT_GELU_KERNEL_H
#define TRT_GELU_KERNEL_H
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
void computeGeluBias(
    float* output, const float* input, const float* bias, const int ld, const int cols, cudaStream_t stream);


// void computeGeluBias(
//     half* output, const half* input, const half* bias, const int ld, const int cols, cudaStream_t stream);

// int computeGelu(cudaStream_t stream, int n, const half* input, half* output);

int computeGelu(cudaStream_t stream, int n, const float* input, float* output);

#endif