#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>
#include <cublas_v2.h>

#include "NvInferPlugin.h"
#include "geluPlugin.h"
#include "gpu_kernel.h"



// Macro for calling GPU functions
#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

using namespace nvinfer1;


/////////////////////////////////

namespace
{
static const char* GELU_PLUGIN_VERSION{"1"};
static const char* GELU_PLUGIN_NAME{"CustomGeluPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection GeluPluginDynamicCreator::mFC{};
std::vector<PluginField> GeluPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GeluPluginDynamicCreator);

GeluPluginDynamic::GeluPluginDynamic(const std::string name, const DataType type)
    : mLayerName(name)
    , mType(type)
    , mHasBias(false)
    , mLd(0)
{
    mBias.values = nullptr;
    mBias.count = 0;
}

GeluPluginDynamic::GeluPluginDynamic(const std::string name, const DataType type, const Weights B)
    : mLayerName(name)
    , mType(type)
    , mHasBias(true)
    , mBias(B)
    , mLd(B.count)
{
}

GeluPluginDynamic::GeluPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // gLogVerbose << "Starting to deserialize GELU plugin" << std::endl;
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mHasBias);

    // gLogVerbose << "Deserialized parameters: mLd: " << mLd << ", mHasBias: " << mHasBias << std::endl;
    if (mHasBias)
    {
        const char* d = static_cast<const char*>(data);
        // gLogVerbose << "Deserializing Bias" << std::endl;
        if (mLd <= 0)
        {
            // gLogError << "Gelu+bias: deserialization inconsistent. HasBias but mLd is 0" << std::endl;
        }
        const size_t wordSize = getElementSize(mType);
        mBiasDev = deserToDev<char>(d, mLd * wordSize);
    }
    // gLogVerbose << "Finished deserializing GELU plugin" << std::endl;
    mBias.values = nullptr;
    mBias.count = mLd;
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GeluPluginDynamic::clone() const
{
    if (mHasBias)
    {
        return new GeluPluginDynamic(mLayerName, mType, mBias);
    }
    return new GeluPluginDynamic(mLayerName, mType);
}

nvinfer1::DimsExprs GeluPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    return inputs[0];
}

bool GeluPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{

    const PluginTensorDesc& input = inOut[0];
    if (pos == 0)
    {
        return (input.type == mType) && (input.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)
    {
        const PluginTensorDesc& output = inOut[1];
        return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
    }
    return false;
}

void GeluPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    assert(mType == in[0].desc.type);
}

size_t GeluPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}
int GeluPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    const int inputVolume = volume(inputDesc[0].dims);

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

// IPluginV2Ext Methods
nvinfer1::DataType GeluPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* GeluPluginDynamic::getPluginType() const
{
    return GELU_PLUGIN_NAME;
}

const char* GeluPluginDynamic::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

int GeluPluginDynamic::getNbOutputs() const
{
    return 1;
}

int GeluPluginDynamic::initialize()
{
    // gLogVerbose << "GELU init start" << std::endl;
    if (mHasBias && mBias.values)
    {
        // target size
        const size_t wordSize = getElementSize(mType);
        const size_t nbBytes = mBias.count * wordSize;
        CHECK(cudaMalloc(&mBiasDev, nbBytes));

        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mBias, reinterpret_cast<float*>(mBiasDev));
        }
        else
        {
            convertAndCopyToDevice(mBias, reinterpret_cast<half*>(mBiasDev));
        }
    }
    // gLogVerbose << "GELU init done" << std::endl;
    return 0;
}

void GeluPluginDynamic::terminate()
{
    if (mHasBias)
    {
        CHECK(cudaFree(mBiasDev));
    }
}

size_t GeluPluginDynamic::getSerializationSize() const
{
    const size_t wordSize = getElementSize(mType);
    const size_t biasSize = mHasBias ? mLd * wordSize : 0;
    return sizeof(mType) + sizeof(mHasBias) + sizeof(mLd) + biasSize;
}

void GeluPluginDynamic::serialize(void* buffer) const
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mHasBias);
    if (mHasBias)
    {
        char *d = static_cast<char*>(buffer);
        const size_t wordSize = getElementSize(mType);
        const size_t biasSize = mHasBias ? mLd * wordSize : 0;
        if (biasSize <= 0)
        {
            // gLogError << "Gelu+bias: bias size inconsistent" << std::endl;
        }
        serFromDev(d, mBiasDev, mLd * wordSize);
    }
}

void GeluPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void GeluPluginDynamic::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* GeluPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

///////////////

GeluPluginDynamicCreator::GeluPluginDynamicCreator()
{

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GeluPluginDynamicCreator::getPluginName() const
{
    return GELU_PLUGIN_NAME;
}

const char* GeluPluginDynamicCreator::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

const PluginFieldCollection* GeluPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* GeluPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{

    Weights bias{DataType::kFLOAT, nullptr, 0};
    int typeId = -1;
    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("type_id") == 0)
        {
            typeId = *static_cast<const int*>(fc->fields[i].data);
            // gLogVerbose << "Building typeId: " << typeId << std::endl;
        }

        if (field_name.compare("bias") == 0)
        {
            // gLogVerbose << "Building bias...\n";
            bias.values = fc->fields[i].data;
            bias.count = fc->fields[i].length;
            bias.type = fieldTypeToDataType(fc->fields[i].type);
        }
    }

    if (typeId < 0 || typeId > 3)
    {
        // gLogError << "GELU: invalid typeId " << typeId << std::endl;
        return nullptr;
    }
    DataType type = static_cast<DataType>(typeId);
    // gLogVerbose << "Creating GeluPluginDynamic...\n";
    if (bias.values == nullptr)
    {
        return new GeluPluginDynamic(name, type);
    }

    return new GeluPluginDynamic(name, type, bias);
}

IPluginV2* GeluPluginDynamicCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call GeluPluginDynamic::destroy()
    return new GeluPluginDynamic(name, serialData, serialLength);
}

void GeluPluginDynamicCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* GeluPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

