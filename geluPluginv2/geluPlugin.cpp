/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include "geluPlugin.h"
#include "NvInfer.h"
#include "geluKernel.h"

#include <vector>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

// Clip plugin specific constants
namespace {
    static const char* GELU_PLUGIN_VERSION{"1"};
    static const char* GELU_PLUGIN_NAME{"CustomGeluPlugin"};
}

// Static class fields initialization
PluginFieldCollection GeluPluginCreator::mFC{};
std::vector<PluginField> GeluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GeluPluginCreator);

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

GeluPlugin::GeluPlugin(const std::string name, const DataType type)
    : mLayerName(name)
    , mType(type)
    , mHasBias(false)
    , mLd(0)
{
    mBias.values = nullptr;
    mBias.count = 0;
}

GeluPlugin::GeluPlugin(const std::string name, const DataType type, const Weights B)
    : mLayerName(name)
    , mType(type)
    , mHasBias(true)
    , mBias(B)
    , mLd(B.count)
{
}

GeluPlugin::GeluPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    std::cout << "Starting to deserialize GELU plugin" << std::endl;

    deserialize_value(&data, &length, &mHasBias);
    deserialize_value(&data, &length, &mInputVolume);
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);

    std::cout << "Deserialized parameters: mInputVolume: " << mInputVolume << ", mHasBias: " << mHasBias << std::endl;
    if (mHasBias)
    {
        const char* d = static_cast<const char*>(data);
        std::cout << "Deserializing Bias" << std::endl;
        if (mLd <= 0)
        {
            std::cout << "Gelu+bias: deserialization inconsistent. HasBias but mLd is 0" << std::endl;
        }
        const size_t wordSize = getElementSize(mType);
        mBiasDev = deserToDev<char>(d, mLd * wordSize);
    }
    std::cout << "Finished deserializing GELU plugin" << std::endl;
    mBias.values = nullptr;
    mBias.count = mLd;
}

const char* GeluPlugin::getPluginType() const
{
    return GELU_PLUGIN_NAME;
}

const char* GeluPlugin::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

int GeluPlugin::getNbOutputs() const
{
    return 1;
}

Dims GeluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);

    // Activation doesn't change input dimension, so output Dims will be the same as input Dims
    return *inputs;
}

int GeluPlugin::initialize()
{
    return 0;
}

int GeluPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    const int inputVolume = mInputVolume;
//    mType = DataType::kFLOAT;
    int status = -1;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (mType == DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        if (mHasBias)
        {
            const float* bias = reinterpret_cast<float*>(mBiasDev);
            const int cols = inputVolume / mLd;
            const int rows = mLd;
            computeGeluBias(output, input, bias, rows, cols, stream);
        }
        else
        {
            status = computeGelu(stream, inputVolume, input, output);
        }
    }
    // else if (mType == DataType::kHALF)
    // {
    //     const half* input = static_cast<const half*>(inputs[0]);

    //     half* output = static_cast<half*>(outputs[0]);

    //     if (mHasBias)
    //     {
    //         const half* bias = reinterpret_cast<half*>(mBiasDev);
    //         const int cols = inputVolume / mLd;
    //         const int rows = mLd;
    //         computeGeluBias(output, input, bias, rows, cols, stream);
    //     }
    //     else
    //     {
    //         status = computeGelu(stream, inputVolume, input, output);
    //     }
    // }
    else
    {
        assert(false);
    }

    return status;
}

size_t GeluPlugin::getSerializationSize() const
{
    const size_t wordSize = getElementSize(mType);
    const size_t biasSize = mHasBias ? mLd * wordSize : 0;
    return sizeof(mType) + sizeof(mHasBias) + sizeof(mLd) + sizeof(mInputVolume) + biasSize;
}

void GeluPlugin::serialize(void* buffer) const
{
    serialize_value(&buffer, mHasBias);
    serialize_value(&buffer, mInputVolume);
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);

//    std::cout << mType << std::endl;
    std::cout << mLd << std::endl;
    std::cout << mHasBias << std::endl;

    if (mHasBias)
    {
        char *d = static_cast<char*>(buffer);
        const size_t wordSize = getElementSize(mType);
        const size_t biasSize = mHasBias ? mLd * wordSize : 0;
        if (biasSize <= 0)
        {
            std::cout << "Gelu+bias: bias size inconsistent" << std::endl;
        }
        serFromDev(d, mBiasDev, mLd * wordSize);
    }
}

void GeluPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kNCHW);

    // Fetch volume for future enqueue() operations
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++) {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
    std::cout << "mInputVolume" << mInputVolume << std::endl;
}

bool GeluPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kNCHW)
        return true;
    else
        return false;
}

void GeluPlugin::terminate() {}

void GeluPlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* GeluPlugin::clone() const
{
    if (mHasBias)
    {
        return new GeluPlugin(mLayerName, mType, mBias);
    }
    return new GeluPlugin(mLayerName, mType);
}

void GeluPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* GeluPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

GeluPluginCreator::GeluPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("typeId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kINT32, 1));
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GeluPluginCreator::getPluginName() const
{
    return GELU_PLUGIN_NAME;
}

const char* GeluPluginCreator::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

const PluginFieldCollection* GeluPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* GeluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    Weights bias{DataType::kFLOAT, nullptr, 0};
    int typeId = -1;
    std::cout << "fc fields" << fc->nbFields << std::endl;
    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        std::cout << field_name << std::endl;

        std::cout << *static_cast<const int*>(fc->fields[i].data) << std::endl;

        if (field_name.compare("typeId") == 0)
        {
            typeId = *static_cast<const int*>(fc->fields[i].data);
            std::cout << "Building typeId: " << typeId << std::endl;
        }

        if (field_name.compare("bias") == 0)
        {
            std::cout << "Building bias...\n";
            bias.values = fc->fields[i].data;
            bias.count = fc->fields[i].length;
            bias.type = fieldTypeToDataType(fc->fields[i].type);
        }
    }

    if (typeId < 0 || typeId > 3)
    {
        std::cout << "GELU: invalid typeId " << typeId << std::endl;
        return nullptr;
    }
    DataType type = static_cast<DataType>(typeId);
    std::cout << "Creating GeluPlugin...\n";
    if (bias.values == nullptr)
    {
        return new GeluPlugin(name, type);
    }

    return new GeluPlugin(name, type, bias);
}

IPluginV2* GeluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call GeluPlugin::destroy()
    return new GeluPlugin(name, serialData, serialLength);
}

void GeluPluginCreator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* GeluPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
