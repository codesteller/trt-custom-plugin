import tensorflow as tf
import tensorrt as trt
import uff
import graphsurgeon as gs
from tf_utils.model import Model
from keras.backend import get_session

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import os


def convert_to_uff(model, frozen_filename, uff_filename):
    # First freeze the graph and remove training nodes.
    output_names = model.output.op.name
    sess = get_session()
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    # Save the model
    with open(frozen_filename, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())

    tf.io.write_graph(sess.graph_def,
                      '/home/codesteller/workspace/ml_workspace/trt-custom-plugin/saved_model/frozen_model',
                      'train.pbtxt')

    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print(nodes)

    uff_model = uff.from_tensorflow(frozen_graph, [output_names])
    with open(uff_filename, "wb") as ofile:
        ofile.write(uff_model)


def build_engine(model_file, TRT_LOGGER):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 16
        # Parse the Uff Network
        parser.register_input("input_1", (3, 150, 150))
        parser.register_output('dense_2/Softmax')
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


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


def main():
    # Save Keras Model to Tensorflow Checkpoint
    final_checkpoint = "/home/codesteller/workspace/ml_workspace/trt-custom-plugin/saved_model/" \
                       "checkpoints/saved_model-0001.h5"
    cnn_model = Model(input_shape=(150, 150, 3))
    cnn_model.build_model()
    cnn_model.convert_checkpoint(final_checkpoint)

    frozen_filename = "../saved_model/frozen_model/model.pb"
    uff_filename = "../saved_model/frozen_model/model.uff"

    if not os.path.exists( "../saved_model/frozen_model"):
        os.makedirs("../saved_model/frozen_model")

    convert_to_uff(model=cnn_model.model,
                   frozen_filename=frozen_filename,
                   uff_filename=uff_filename)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    engine_filename = "../saved_model/frozen_model/model.engine"
    engine = build_engine(uff_filename, TRT_LOGGER)

    with open(engine_filename, "wb") as f:
        f.write(engine.serialize())


if __name__ == '__main__':
    main()
