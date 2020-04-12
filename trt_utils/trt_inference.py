import tensorrt as trt
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
from PIL import Image
import numpy as np
from glob import glob
import os


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


ENGINE_PATH = "../saved_model/frozen_model/model.engine"
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
# TEST_DATA_PATH = "/home/codesteller/datasets/kaggle/cv/dogs_cats/test_data/test/"
TEST_DATA_PATH = "../test_data"
TEST_SAMPLES = 10
test_images = glob(os.path.join(TEST_DATA_PATH, "*.jpg"))[:TEST_SAMPLES]


with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    # Note that we have to provide the plugin factory when deserializing an engine built with an IPlugin or IPluginExt.
    engine = runtime.deserialize_cuda_engine(f.read())

with engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    for impath in test_images:
        img = Image.open(impath)
        img = img.resize((150, 150))
        img = np.array(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0) / 255.0
        img = img.ravel()
        inputs[0].host = img

        [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print(output)
        pred = np.argmax(output)
        basename = os.path.basename(impath)
        if pred:
            print("{} : Prediction -  Dog".format(basename))
        else:
            print("{} : Prediction -  Cat".format(basename))



