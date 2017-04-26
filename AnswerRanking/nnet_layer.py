from __future__ import absolute_import
from keras import backend as K
from keras.engine.topology import Layer
from theano import tensor as T
from keras.layers import LSTM, activations,Wrapper
import numpy as np
from keras.engine import InputSpec
import sys

#Learn from class Merge(Layer) and keras doc
class SimilarityMatrixLayer(Layer):
    def __init__(self, layers=None, output_shape=None,
                 node_indices=None, tensor_indices=None, name=None,**kwargs):
        self.layers = layers
        self._output_shape = output_shape
        self.node_indices = node_indices
        super(SimilarityMatrixLayer, self).__init__(**kwargs)

        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.regularizers = []
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.supports_masking = False
        self.uses_learning_phase = False
        self.input_spec = None  # compatible with whatever

        # if self.dropout_W or self.dropout_U:
        #     self.uses_learning_phase = True

        if layers:
            # this exists for backwards compatibility.
            # equivalent to:
            # merge = Merge(layers=None)
            # output = merge([input_tensor_1, input_tensor_2])
            if not node_indices:
                # by default we connect to
                # the 1st output stream in the input layer
                node_indices = [0 for _ in range(len(layers))]
            if not tensor_indices:
                tensor_indices = [0 for _ in range(len(layers))]

            self._arguments_validation(layers, node_indices, tensor_indices)
            # self.built = True
        #    self.add_inbound_node(layers, node_indices, tensor_indices)
            #self.built = True
            input_tensors = []
            input_masks = []
            for i, layer in enumerate(layers):
                node_index = node_indices[i]
                tensor_index = tensor_indices[i]
                inbound_node = layer.inbound_nodes[node_index]
                input_tensors.append(inbound_node.output_tensors[tensor_index])
                input_masks.append(inbound_node.output_masks[tensor_index])
            self(input_tensors, mask=input_masks)
        else:
            self.built = False

    def _arguments_validation(self, layers, node_indices, tensor_indices):
        '''Validates user-passed arguments and raises exceptions
        as appropriate.
        '''
        if type(layers) not in {list, tuple} or len(layers) < 2:
            raise Exception('A Merge should only be applied to a list of '
                            'layers with at least 2 elements. Found: ' + str(layers))
        if tensor_indices is None:
            tensor_indices = [None for _ in range(len(layers))]

        input_shapes = []
        for i, layer in enumerate(layers):
            layer_output_shape = layer.get_output_shape_at(node_indices[i])
            if type(layer_output_shape) is list:
                # case: the layer has multiple output tensors
                # and we only need a specific one
                layer_output_shape = layer_output_shape[tensor_indices[i]]
            input_shapes.append(layer_output_shape)

    def build(self, input_shape):
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        q_in, a_in = shape1[1], shape2[1]
       # initial_weight_value = np.random.random((q_in, a_in))
        initial_weight_value = np.zeros((q_in, a_in))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]
       # super(SimilarityMatrixLayer, self).build()


    def call(self, x, mask=None):
        if type(x) is not list or len(x) <= 1:
            raise Exception('Merge must be called on a list of tensors '
                            '(at least 2). Got: ' + str(x))
        q, a = x[0], x[1]
        dot = T.batched_dot(q, T.dot(a, self.W.T))
        out = T.concatenate([dot.dimshuffle(0, 'x'), q, a], axis=1)
        return out

    # def __call__(self, inputs, mask=None):
    #     '''We disable successive calls to __call__ for Merge layers.
    #     Although there is no technical obstacle to
    #     making it possible to __call__ a Merge intance many times
    #     (it is just a layer), it would make for a rather unelegant API.
    #     '''
    #     if type(inputs) is not list:
    #         raise Exception('Merge can only be called on a list of tensors, '
    #                         'not a single tensor. Received: ' + str(inputs))
    #     if self.built:
    #         raise Exception('A Merge layer cannot be used more than once, '
    #                         'please use ' +
    #                         'the "merge" function instead: ' +
    #                         '`merged_tensor = merge([tensor_1, tensor2])`.')
    #
    #     all_keras_tensors = True
    #     for x in inputs:
    #         if not hasattr(x, '_keras_history'):
    #             all_keras_tensors = False
    #             break
    #
    #     if all_keras_tensors:
    #         layers = []
    #         node_indices = []
    #         tensor_indices = []
    #         for x in inputs:
    #             layer, node_index, tensor_index = x._keras_history
    #             layers.append(layer)
    #             node_indices.append(node_index)
    #             tensor_indices.append(tensor_index)
    #         self._arguments_validation(layers,
    #                                    self._output_shape,
    #                                    node_indices, tensor_indices)
    #         # self.built = True
    #         self.add_inbound_node(layers)
    #
    #         outputs = self.inbound_nodes[-1].output_tensors
    #         return outputs[0]  # merge only returns a single tensor
    #     else:
    #         return self.call(inputs, mask)

    def compute_output_shape(self, input_shape):
        assert type(input_shape) is list  # must have mutiple input shape tuples
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        output_shape[1] = input_shapes[0][1] + input_shapes[1][1] + 1
        return tuple(output_shape)

    def get_output_shape_at(self, node_index):
        '''Retrieves the output shape(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'output_shapes',
                                                 'output shape')
    def get_config(self):
        # py3 = sys.version_info[0] == 3
        #
        # if isinstance(self._output_shape, python_types.LambdaType):
        #     if py3:
        #         output_shape = marshal.dumps(self._output_shape.__code__)
        #     else:
        #         output_shape = marshal.dumps(self._output_shape.func_code)
        #     output_shape_type = 'lambda'
        # elif callable(self._output_shape):
        #     output_shape = self._output_shape.__name__
        #     output_shape_type = 'function'
        # else:
        #     output_shape = self._output_shape
        #     output_shape_type = 'raw'

        return {'name': self.name,
                'output_shape': self.output_shape}
    @classmethod
    def from_config(cls, config):
        output_shape = config['output_shape']
        config['output_shape'] = output_shape
        return super(SimilarityMatrixLayer, cls).from_config(config)

def similarityMatrix(inputs, output_shape=None, name=None):
    '''Functional merge, to apply to Keras tensors (NOT layers).
    Returns a Keras tensor.
    '''
    all_keras_tensors = True
    for x in inputs:
        if not hasattr(x, '_keras_history'):
            all_keras_tensors = False
            break
    if all_keras_tensors:
        input_layers = []
        node_indices = []
        tensor_indices = []
        for x in inputs:
            input_layer, node_index, tensor_index = x._keras_history
            input_layers.append(input_layer)
            node_indices.append(node_index)
            tensor_indices.append(tensor_index)
        merge_layer = SimilarityMatrixLayer(input_layers, node_indices=node_indices,
                                            tensor_indices=tensor_indices,
                                            name=name)
        return merge_layer.inbound_nodes[0].output_tensors[0]
    else:
        merge_layer = SimilarityMatrixLayer(output_shape=output_shape,
                                            name=name)
        return merge_layer(inputs)


class AttentionLSTM(LSTM):
    def __init__(self, output_dim, attention_vec, attn_activation='tanh',
                 attn_inner_activation='tanh', single_attention_param=False,
                 n_attention_dim=None, **kwargs):
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        # self.attn_inner_activation = activations.get(attn_inner_activation)
        self.single_attention_param = single_attention_param

        #UPDATED!
        # self.single_attention_param = single_attn
        # self.n_attention_dim = output_dim if n_attention_dim is None else n_attention_dim

        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.inner_init((attention_dim, self.output_dim),
                                   name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.output_dim,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.inner_init((self.output_dim, 1),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights += [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[4]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        # Intuitively it makes more sense to use a sigmoid (was getting some NaN problems
        # which I think might have been caused by the exponential function -> gradients blow up)
        #s = self.attn_activation(K.dot(m, self.U_s) + self.b_s)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s) #UPDATED!
        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.output_dim, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM, self).get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants

    def get_config(self):
        attention_vec = tuple(self.attention_vec._keras_shape)
        config = {'attention_vec': attention_vec}
                  # 'attn_activation': self.attn_activation,
                  # 'attn_inner_activation': self.attn_inner_activation,
                  # 'single_attn': self.single_attention_param,
                  # 'n_attention_dim': self.n_attention_dim}
        base_config = super(AttentionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttentionLSTMWrapper(Wrapper):
    def __init__(self, layer, attention_vec, attn_activation='tanh', single_attention_param=False, **kwargs):
        assert isinstance(layer, LSTM)
        self.supports_masking = True
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param
        super(AttentionLSTMWrapper, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(AttentionLSTMWrapper, self).build()

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = self.layer.inner_init((self.layer.output_dim, self.layer.output_dim), name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.layer.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.layer.inner_init((attention_dim, self.layer.output_dim), name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.layer.output_dim,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.layer.inner_init((self.layer.output_dim, 1), name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.layer.inner_init((self.layer.output_dim, self.layer.output_dim), name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.layer.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights = [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        h, [h, c] = self.layer.step(x, states)
        attention = states[4]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.layer.output_dim, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output
