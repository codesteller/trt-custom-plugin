import tensorflow as tf
import tensorrt as trt
import uff
import graphsurgeon as gs
from tf_utils.model import Model
from keras.backend import get_session
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


def main():
    # Save Keras Model to Tensorflow Checkpoint
    final_checkpoint = "/home/codesteller/workspace/ml_workspace/trt-custom-plugin/saved_model/" \
                       "checkpoints/saved_model-0005.h5"
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
