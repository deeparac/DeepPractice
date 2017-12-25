import mxnet as mx
from mxnet import gluon

def params_gen(input_size, num_units):
    i2h_weight = gluon.Parameter('i2h_weight',
                                 shape=(4*num_units, input_size))
    h2h_weight = gluon.Parameter('h2h_weight',
                                 shape=(4*num_units, input_size))
    i2h_bias = gluon.Parameter('i2h_bias',
                                 shape=(4*num_units))
    h2h_bias = gluon.Parameter('h2h_bias',
                                 shape=(4*num_units))
    params = {
        'i2h_weight': i2h_weight,
        'i2h_bias': i2h_bias,
        'h2h_weight': h2h_weight,
        'h2h_bias': h2h_bias
    }
    return params
    
class LSTM(gluon.nn.Block):
    def __init__(self, input_size, num_units, data,
                 prev_state, params, ctx=mx.gpu(), **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.params = params
        # input to hidden
        self.i2h_weight = self.params.get('i2h_weight',
                                          shape=(4*num_units, input_size),
                                          init=mx.init.Xavier())
        self.i2h_bias = self.params.get('i2h_bias',
                                        shape=(4*num_units,),
                                        init=mx.init.Xavier())
        # hidden to hidden
        self.h2h_weight = self.params.get('h2h_weight',
                                          shape=(4*num_units, input_size),
                                          init=mx.init.Xavier())
        self.h2h_bias = self.params.get('h2h_weight',
                                        shape=(4*num_units),
                                        init=mx.init.Xavier())
    def forward(self, x):
        i2h = mx.sym.FullyConnetecd(data=data,
                                    weight=i2h_weight,
                                    bias=i2h_bias,
                                    num_hidden=self.num_units*4)
        h2h = mx.sym.FullyConnetecd(data=prev_state,
                                    weight=h2h_weight,
                                    bias=h2h_bias,
                                    num_hidden=self.num_units*4)
        gates = i2h + h2h
        slice_gates = mx.sym.split(gates, num_outputs=4)
        in_gate = mx.sym.Activation(slice_gates[0], act_type='sigmoid')
        forget_gate = mx.sym.Activation(slice_gates[1], act_type='sigmoid')
        in_transform = mx.sym.Activation(slice_gates[2], act_type='tanh')
        out_gate = mx.sym.Activation(slice_gates[3], act_type='sigmoid')

        c = forget_gate * prev_state + in_gate * in_transform
        h = out_gate * mx.sym.Activation(c, act_type='tanh')

        return h, c
