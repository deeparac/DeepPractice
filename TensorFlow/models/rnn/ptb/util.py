import tensorflow as tf

from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework import rewriter_config_pb2

FLAGS = tf.flags.FLAGS

def export_state_tuples(state_tuples, name):
    for state_tuple in state_tuples:
        tf.add_to_collection(name, state_tuple.c)
        tf.add_to_collection(name, state_tuple.h)

def import_state_tuples(state_tuples, name, num_replicas):
    restored = []
    for i in range(len(state_tuples) * num_replicas):
        c = tf.get_collection_ref(name)[2 * i + 0]
        h = tf.get_collection_ref(name)[2 * i + 1]
        restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
    return tuple(restored)

def with_prefix(prefix, name):
    return "/".join((prefix, name))

def with_autoparallel_prefix(replica_id, name):
    return with_prefix("AutoParallel-Replica_%d" % replica_id, name)

class UpdateCollection(object):
    def __init__(self, metagraph, model):
        self._metagraph = metagraph
        self.replicate_states(model.initial_state_name)
        self.replicate_states(model.final_state_name)
        self.update_snapshot_name("variables")
        self.update_snapshot_name("trainable_variables")

    def update_snapshot_name(self, var_coll_name):
        var_list = self._metagraph.collection_def[var_coll_name]
        for i, value in enumerate(var_list.bytes_list.value):
            var_def = variable_pb2.VariableDef()
            var_def.ParseFromString(value)

            if var_def.snapshot_name != "Model/global_step/read:0":
                var_def.snapshot_name = with_autoparallel_prefix(
            0, var_def.snapshot_name)

        value = var_def.SerializeToString()
        var_list.bytes_list.value[i] = value

    def replicate_states(self, state_coll_name):
        state_list = self._metagraph.collection_def[state_coll_name]
        num_states = len(state_list.node_list.value)

        for replica_id in range(1, FLAGS.num_gpus):
          for i in range(num_states):
            state_list.node_list.value.append(state_list.node_list.value[i])

        for replica_id in range(FLAGS.num_gpus):
          for i in range(num_states):
            index = replica_id * num_states + i
            state_list.node_list.value[index] = with_autoparallel_prefix(
                replica_id, state_list.node_list.value[index])

def auto_parallel(metagraph, model):
    from tensorflow.python.grappler import tf_optimizer
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    rewriter_config.optimizers.append("autoparallel")
    rewriter_config.auto_parallel.enable = True
    rewriter_config.auto_parallel.num_replicas = FLAGS.num_gpus
    optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, metagraph)
    metagraph.graph_def.CopyFrom(optimized_graph)
    UpdateCollection(metagraph, model)