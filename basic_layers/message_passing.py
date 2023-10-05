import tensorflow as tf

from basic_layers.mlp import MLP

KL = tf.keras.layers
KI = tf.keras.initializers


class MessagePassing(KL.Layer):
    def __init__(self,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 edge_net_init_scale,
                 node_update_type,
                 use_reverse_direction,
                 reverse_param_different,
                 use_layer_norm,
                 name=None):
        assert node_update_type in ["mlp", "residual", "gru"]
        if node_update_type == "gru":
            assert len(node_hidden_sizes) == 1

        super(MessagePassing, self).__init__(name=name)

        self.edge_hidden_sizes = edge_hidden_sizes
        self.node_hidden_sizes = node_hidden_sizes
        self.edge_net_init_scale = edge_net_init_scale
        self.node_update_type = node_update_type
        self.use_reverse_direction = use_reverse_direction
        self.reverse_param_different = reverse_param_different
        self.use_layer_norm = use_layer_norm

        w_init = KI.VarianceScaling(scale=edge_net_init_scale,
                                    mode="fan_avg",
                                    distribution="uniform")
        self.message_net = MLP(
            output_sizes=edge_hidden_sizes,
            w_init=w_init,
            name="message_net")
        if use_reverse_direction:
            if reverse_param_different:
                self.reverse_message_net = MLP(
                    output_sizes=edge_hidden_sizes,
                    w_init=w_init,
                    name="reverse_message_net")
            else:
                self.reverse_message_net = self.message_net

        if node_update_type in ["mlp", "residual"]:
            self.node_update_net = MLP(
                output_sizes=node_hidden_sizes,
                name="node_update_net")
        elif node_update_type == "gru":
            self.node_update_net = KL.GRUCell(
                units=node_hidden_sizes[0],
                kernel_initializer="glorot_uniform",
                recurrent_initializer="glorot_uniform",
                name="node_update_net")

        if use_layer_norm:
            self.layer_norm_passing = KL.LayerNormalization()
            if node_update_type in ["mlp", "residual"]:
                self.layer_norm_update = KL.LayerNormalization()

    def call(self, node_states, from_idx, to_idx, training,
             node_features=None, edge_features=None):
        aggregated_messages = self.message_passing(
            node_states, from_idx, to_idx, training=training,
            edge_features=edge_features)
        updated_node_states = self.node_update(
            node_states, [aggregated_messages], training=training,
            node_features=node_features)
        return updated_node_states

    def message_passing(self, node_states, from_idx, to_idx, training,
                        edge_features=None):
        extra_kwargs = {"training": training}
        aggregated_messages = message_passing_once(
            node_states, from_idx, to_idx,
            self.message_net, extra_kwargs,
            aggregation_fn=tf.math.unsorted_segment_sum,
            edge_features=edge_features)
        if self.use_reverse_direction:
            reverse_aggregated_messages = message_passing_once(
                node_states, to_idx, from_idx,
                self.reverse_message_net, extra_kwargs,
                aggregation_fn=tf.math.unsorted_segment_sum,
                edge_features=edge_features)
            aggregated_messages += reverse_aggregated_messages
        if self.use_layer_norm:
            aggregated_messages = self.layer_norm_passing(aggregated_messages)
        return aggregated_messages

    def node_update(self, node_states, node_update_inputs, training,
                    node_features=None):
        assert isinstance(node_update_inputs, list)
        node_update_inputs = list(node_update_inputs)
        if self.node_update_type in ["mlp", "residual"]:
            node_update_inputs.append(node_states)
        if node_features is not None:
            node_update_inputs.append(node_features)

        node_update_inputs = tf.concat(node_update_inputs, axis=-1)
        if self.node_update_type == "gru":
            outputs, _ = self.node_update_net(node_update_inputs, node_states)
        else:
            outputs = self.node_update_net(
                node_update_inputs, training=training)
            if self.use_layer_norm:
                outputs = self.layer_norm_update(outputs)
            if self.node_update_type == "residual":
                outputs += node_states
        return outputs


def message_passing_once(node_states, from_idx, to_idx,
                         message_net, net_extra_kwargs,
                         aggregation_fn, edge_features=None):
    from_states = tf.gather(node_states, from_idx)
    to_states = tf.gather(node_states, to_idx)
    edge_inputs = [from_states, to_states]
    if edge_features is not None:
        edge_inputs.append(edge_features)
    edge_inputs = tf.concat(edge_inputs, axis=-1)
    messages = message_net(edge_inputs, **net_extra_kwargs)
    aggregated_messages = aggregation_fn(
        messages, to_idx, tf.shape(node_states)[0])
    return aggregated_messages
