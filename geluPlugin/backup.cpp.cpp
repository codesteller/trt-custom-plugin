
// namespace
// {
// 	const char* GELU_PLUGIN_VERSION{ "v1.0" };
// 	const char* GELU_PLUGIN_NAME{ "geluCustomPlugin" };
// }

// class geluPlugin : public IPluginV2
// {
// public:
// 	// Ordinary ctor, plugin not yet configured for particular inputs/output
// 	geluPlugin() {}

// 	// Ctor for clone()
// 	geluPlugin(int totalElements)
// 	{
// 		mTotalElements = totalElements;
// 	}

// 	// Ctor for loading from serialized byte array
// 	geluPlugin(const void* data, size_t length)
// 	{
// 		const char* d = reinterpret_cast<const char*>(data);
// 		const char* a = d;

// 		mTotalElements = read<int>(d);

// 		assert(d == a + length);
// 	}

// 	int getNbOutputs() const override
// 	{
// 		return 1;
// 	}

// 	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
// 	{
// 		assert(nbInputDims >= 1);
// 		assert(index == 0);

// 		//Output dimensions are same as input dims
// 		//Using dimensions of any element
// 		return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
// 	}

// 	int initialize() override
// 	{
// 		CHECK(cublasCreate(&mCublas));
// 		return 0;
// 	}

// 	void terminate() override
// 	{
// 		CHECK(cublasDestroy(mCublas));
// 	}

// 	size_t getWorkspaceSize(int maxBatchSize) const override
// 	{
// 		return 0;
// 	}

// 	int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
// 	{
// 		size_t inputOffset = 0;
// 		float* output = reinterpret_cast<float*>(outputs[0]);

// 		//Activation layer applied to one input only thats why we index 0
// 		const float* input = reinterpret_cast<const float*>(inputs[0]);
		
// 		// Call kernel launcher
// 		GELU::computeGelu(cudaStream_t stream, int batchSize, const float* input, float* output)

// 		return 0;
// 	}
// 	size_t getSerializationSize() const override
// 	{
// 		size_t size = sizeof(mTotalElements);
// 		return size;
// 	}

// 	void serialize(void* buffer) const override
// 	{
// 		char* d = reinterpret_cast<char*>(buffer);
// 		char* a = d;

// 		size_t totalElements = mTotalElements;

// 		write(d, totalElements);

// 		assert(d == a + getSerializationSize());
// 	}

// 	void configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override
// 	{
// 		assert(nbOutputs == 1);

// 		mTotalElements = 0;

// 		for (int i = 0; i < nbInputs; ++i)
// 		{
// 			//Number of elements to change
// 			mTotalElements += inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
// 		}
// 	}

// 	bool supportsFormat(DataType type, PluginFormat format) const override
// 	{
// 		return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
// 	}

// 	const char* getPluginType() const override { return GELU_PLUGIN_NAME; }

// 	const char* getPluginVersion() const override { return GELU_PLUGIN_VERSION; }

// 	void destroy() override {}

// 	IPluginV2* clone() const override
// 	{
// 		return new geluPlugin(mTotalElements);
// 	}

// 	void setPluginNamespace(const char* pluginNamespace) override
// 	{
// 		mPluginNamespace = pluginNamespace;
// 	}

// 	const char* getPluginNamespace() const override
// 	{
// 		return mPluginNamespace.c_str();
// 	}


// private:
// 	template <typename T>
// 	void write(char*& buffer, const T & val) const
// 	{
// 		*reinterpret_cast<T*>(buffer) = val;
// 		buffer += sizeof(T);
// 	}

// 	template <typename T>
// 	T read(const char*& buffer)
// 	{
// 		T val = *reinterpret_cast<const T*>(buffer);
// 		buffer += sizeof(T);
// 		return val;
// 	}

// 	int mTotalElements;
// 	cublasHandle_t mCublas;
// 	std::string mPluginNamespace = "";
// };

// // PluginCreator boilerplate code for FlattenConcat plugin
// class geluPluginCreator : public IPluginCreator
// {
// public:
// 	geluPluginCreator()
// 	{
// 		mFC.nbFields = 0;
// 		mFC.fields = 0;
// 	}

// 	~geluPluginCreator() {}

// 	const char* getPluginName() const override { return GELU_PLUGIN_NAME; }

// 	const char* getPluginVersion() const override { return GELU_PLUGIN_VERSION; }

// 	const PluginFieldCollection* getFieldNames() override { return &mFC; }

// 	IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
// 	{
// 		return new geluPlugin();
// 	}

// 	IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
// 	{

// 		return new geluPlugin(serialData, serialLength);
// 	}

// 	void setPluginNamespace(const char* pluginNamespace) override
// 	{
// 		mPluginNamespace = pluginNamespace;
// 	}

// 	const char* getPluginNamespace() const override
// 	{
// 		return mPluginNamespace.c_str();
// 	}

// private:
// 	static PluginFieldCollection mFC;
// 	static std::vector<PluginField> mPluginAttributes;
// 	std::string mPluginNamespace = "";
// };

// PluginFieldCollection geluPluginCreator::mFC{};
// std::vector<PluginField> geluPluginCreator::mPluginAttributes;

// REGISTER_TENSORRT_PLUGIN(geluPluginCreator);