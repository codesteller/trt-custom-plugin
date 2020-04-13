import tensorflow as tf
import tensorrt as trt
import uff
import graphsurgeon as gs
from tf_utils.model import Model
from keras.backend import get_session
import os
import ctypes

# Add plugin compiled library
ctypes.CDLL("/home/codesteller/workspace/ml_workspace/trt_ws/trt-custom-plugin/geluPluginv2/build/libGeluPlugin.so")


def create_plugin_node(dynamic_graph):
    gelu_node = gs.create_plugin_node(name="GeluActivation", op="CustomGeluPlugin", typeId=0)
    namespace_plugin_map = {"GeluActivation": gelu_node}
    dynamic_graph.collapse_namespaces(namespace_plugin_map)


def convert_to_uff(model, frozen_filename, uff_filename):
    # First freeze the graph and remove training nodes.
    output_names = model.output.op.name
    # output_names = "dense_2/MatMul"
    sess = get_session()
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    # Save the model
    with open(frozen_filename, "wb") as fptr:
        fptr.write(frozen_graph.SerializeToString())

    tf.io.write_graph(sess.graph_def,
                      '/home/codesteller/workspace/ml_workspace/trt_ws/trt-custom-plugin/saved_model/frozen_model',
                      'train.pbtxt', as_text=True)

    print_graphdef(tf.get_default_graph().as_graph_def(), '/home/codesteller/workspace/ml_workspace/trt_ws/'
                                                          'trt-custom-plugin/saved_model/frozen_model/train.txt')

    # Transform graph using graphsurgeon to map unsupported TensorFlow
    # operations to appropriate TensorRT custom layer plugins
    dynamic_graph = gs.DynamicGraph(frozen_graph)
    create_plugin_node(dynamic_graph)

    print_dynamic_graph(dynamic_graph, filename='/home/codesteller/workspace/ml_workspace/trt_ws/trt-custom-plugin/'
                                                'saved_model/frozen_model/final_node_graph.txt')

    uff_model = uff.from_tensorflow(dynamic_graph, [output_names])
    with open(uff_filename, "wb") as fptr:
        fptr.write(uff_model)


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


def main():
    # Save Keras Model to Tensorflow Checkpoint
    final_checkpoint = "/home/codesteller/workspace/ml_workspace/trt_ws/trt-custom-plugin/saved_model/" \
                       "checkpoints/saved_model-0001.h5"
    cnn_model = Model(input_shape=(150, 150, 3))
    cnn_model.build_model()
    # cnn_model.convert_checkpoint(final_checkpoint)
    cnn_model.model.load_weights(final_checkpoint)

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


def print_dynamic_graph(dynamic_graph, filename='./models/final_node_graph.txt'):
    nodes = [n for n in dynamic_graph]
    with open(filename, 'w+') as fptr:
        for i, node in enumerate(nodes):
            fptr.writelines('%d--%s\t%s\n' % (i, node.name, node.op))
            for i, n in enumerate(node.input):
                fptr.writelines('|--- %d --- %s\n' % (i, n))


def print_graphdef(graphdef, filename='./models/final_node_graph.txt'):
    nodes = [n for n in graphdef.node]
    with open(filename, 'w+') as fptr:
        for i, node in enumerate(nodes):
            fptr.writelines('%d--%s\t%s\n' % (i, node.name, node.op))
            for i, n in enumerate(node.input):
                fptr.writelines('|--- %d --- %s\n' % (i, n))


if __name__ == '__main__':
    main()
