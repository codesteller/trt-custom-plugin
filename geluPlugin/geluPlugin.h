/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_GELU_PLUGIN_H
#define TRT_GELU_PLUGIN_H

#include "NvInferRuntimeCommon.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <cmath>
#include <cublas_v2.h>
// #include "bertCommon.h"
// #include "common.h"
#include "serialize.hpp"
#include "cuda_runtime.h"
#include "cuda.h"
// #include "device_launch_parameters.h"
#include <cstring>
#include "NvInfer.h"


#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)


// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class GeluPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    GeluPluginDynamic(const std::string name, const nvinfer1::DataType type);
    GeluPluginDynamic(const std::string name, const nvinfer1::DataType type, const nvinfer1::Weights B);

    GeluPluginDynamic(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make GeluPluginDynamic without arguments, so we delete
    // default constructor.
    GeluPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    // IPluginV2 Methods
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    nvinfer1::DataType mType;
    bool mHasBias;
    nvinfer1::Weights mBias;
    char* mBiasDev;
    size_t mLd;

protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
};

class GeluPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    GeluPluginDynamicCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const nvinfer1::PluginFieldCollection* getFieldNames() override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
// }

// -- INLINE FUNCTIONS --
inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype)
{
    switch (ftype)
    {
    case nvinfer1::PluginFieldType::kFLOAT32:
    {
        // gLogVerbose << "PluginFieldType is Float32" << std::endl;
        return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16:
    {
        // gLogVerbose << "PluginFieldType is Float16" << std::endl;
        return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32:
    {
        // gLogVerbose << "PluginFieldType is Int32" << std::endl;
        return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8:
    {
        // gLogVerbose << "PluginFieldType is Int8" << std::endl;
        return nvinfer1::DataType::kINT8;
    }
    default: throw std::invalid_argument("No corresponding datatype for plugin field type");
    }
}

template <typename T>
inline void serFromDev(char*& buffer, const T* data, size_t nbElem)
{
    const size_t len = sizeof(T) * nbElem;
    CHECK(cudaMemcpy(buffer, data, len, cudaMemcpyDeviceToHost));
    buffer += len;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    // case nvinfer1::DataType::kBOOL: return 0;
    // case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, half* destDev)
{
    size_t wordSize = sizeof(half);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kHALF)
    {
        // gLogVerbose << "Half Weights(Host) => Half Array(Device)" << std::endl;
        CHECK(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
        // gLogVerbose << "Float Weights(Host) => Half Array(Device)" << std::endl;
        std::vector<half> tmp(src.count);
        const float* values = reinterpret_cast<const float*>(src.values);

        for (int it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __float2half(values[it]);
        }
        CHECK(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
    }
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, float* destDev)
{

    size_t wordSize = sizeof(float);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kFLOAT)
    {
        // gLogVerbose << "Float Weights(Host) => Float Array(Device)" << std::endl;
        CHECK(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
        // gLogVerbose << "Half Weights(Host) => Float Array(Device)" << std::endl;
        std::vector<float> tmp(src.count);
        const half* values = reinterpret_cast<const half*>(src.values);

        for (int it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __half2float(values[it]);
        }

        CHECK(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
    }
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

template <typename T>
inline T* deserToDev(const char*& buffer, size_t nbElem)
{
    T* dev = nullptr;
    const size_t len = sizeof(T) * nbElem;
    CHECK(cudaMalloc(&dev, len));
    CHECK(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

    buffer += len;
    return dev;
}

#endif // TRT_GELU_PLUGIN_H
