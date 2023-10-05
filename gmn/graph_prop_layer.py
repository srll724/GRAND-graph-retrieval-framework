import tensorflow as tf

from basic_layers.mlp import MLP

KL = tf.keras.layers
KI = tf.keras.initializers


class GraphPropLayer(KL.Layer):
    def __init__(self,
                 node_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 edge_net_init_scale,
                 node_update_type,
                 use_reverse_direction,
                 reverse_param_different,
                 use_layer_norm,
                 name=None):
        super(GraphPropLayer, self).__init__(name=name)

        self.node_state_dim = node_state_dim
        self.edge_hidden_sizes = edge_hidden_sizes
        self.node_hidden_sizes = node_hidden_sizes + [node_state_dim]
        self.edge_net_init_scale = edge_net_init_scale
        self.node_update_type = node_update_type
        self.use_reverse_direction = use_reverse_direction
        self.reverse_param_different = reverse_param_different
        self.use_layer_norm = use_layer_norm

    def build(self, input_shape):
        w_init = KI.VarianceScaling(scale=self.edge_net_init_scale,
                                    mode="fan_avg",
                                    distribution="uniform")
        self.message_net = MLP(
            self.edge_hidden_sizes,
            w_init=w_init,
            name="message_net")
        if self.use_reverse_direction:
            if self.reverse_param_different:
                self.reverse_message_net = MLP(
                    self.edge_hidden_sizes,
                    w_init=w_init,
                    name="reverse_message_net")
            else:
                self.reverse_message_net = self.message_net

        if self.node_update_type == "gru":
            self.node_update_net = KL.GRUCell(
                units=self.node_state_dim,
                kernel_initializer="glorot_uniform",
                recurrent_initializer="glorot_uniform",
                name="node_update_net")
        elif self.node_update_type in ("mlp", "residual"):
            self.node_update_net = MLP(
                self.node_hidden_sizes, name="node_update_net")
        else:
            raise NotImplementedError

        if self.use_layer_norm:
            self.layer_norm_agg = KL.LayerNormalization()
            if self.node_update_type in ("mlp", "residual"):
                self.layer_norm_update = KL.LayerNormalization()

        KL.Layer.build(self, input_shape)

    def call(self, node_states, from_idx, to_idx, training,
             node_features=None, edge_features=None):
        aggregated_messages = self.compute_aggregated_messages(
            node_states, from_idx, to_idx, training=training,
            edge_features=edge_features)
        updated_node_states = self.compute_node_update(
            node_states, [aggregated_messages], training=training,
            node_features=node_features)
        return updated_node_states

    def compute_aggregated_messages(self, node_states, from_idx, to_idx,
                                    training, edge_features=None):
        extra_kwargs = {"training": training}
        aggregated_messages = self.graph_prop_once(
            node_states, from_idx, to_idx, self.message_net, extra_kwargs,
            aggregation_fn=tf.math.unsorted_segment_sum,
            edge_features=edge_features)
        if self.use_reverse_direction:
            # if self.reverse_param_different:
            #     reverse_net = self.reverse_message_net
            # else:
            #     reverse_net = self.message_net
            reverse_aggregated_messages = self.graph_prop_once(
                node_states, to_idx, from_idx,
                self.reverse_message_net, extra_kwargs,
                aggregation_fn=tf.math.unsorted_segment_sum,
                edge_features=edge_features)
            aggregated_messages += reverse_aggregated_messages

        if self.use_layer_norm:
            aggregated_messages = self.layer_norm_agg(aggregated_messages)

        return aggregated_messages

    def compute_node_update(self, node_states, node_state_inputs, training,
                            node_features=None):
        assert isinstance(node_state_inputs, list)
        node_state_inputs = list(node_state_inputs)
        if self.node_update_type in ("mlp", "residual"):
            node_state_inputs.append(node_states)
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = tf.concat(node_state_inputs, axis=-1)

        if self.node_update_type == "gru":
            outputs, _ = self.node_update_net(node_state_inputs, node_states)
            return outputs
        else:
            outputs = self.node_update_net(
                node_state_inputs, training=training)
            if self.use_layer_norm:
                outputs = self.layer_norm_update(outputs)
            if self.node_update_type == "mlp":
                return outputs
            elif self.node_update_type == "residual":
                return node_states + outputs
            else:
                raise NotImplementedError

    @staticmethod
    def graph_prop_once(node_states, from_idx, to_idx,
                        message_net, extra_kwargs,
                        aggregation_fn=tf.math.unsorted_segment_sum,
                        edge_features=None):
        from_states = tf.gather(node_states, from_idx)
        to_states = tf.gather(node_states, to_idx)

        edge_inputs = [from_states, to_states]
        if edge_features is not None:
            edge_inputs.append(edge_features)

        edge_inputs = tf.concat(edge_inputs, axis=-1)
        messages = message_net(edge_inputs, **extra_kwargs)
        aggregated_messages = aggregation_fn(
            messages, to_idx, tf.shape(node_states)[0])
        return aggregated_messages
