from keras.models import model_from_json
from keras.layers.convolutional import Conv3D
from keras.activations import relu
from keras.initializers import glorot_uniform

#  Did K Geras use bais? They did more than one feature map...
down_params = {'filters': 3,
               'kernel_size': (1, 4, 3),
               'strides': (1, 4, 3), 'padding': "valid",
               'data_format': "channels_last",
               'dilation_rate': (1, 1, 1),
               'activation': 'relu', 'use_bias': False,
               'kernel_initializer': glorot_uniform,
               'name': 'conv3d_down1',
               'batch_input_shape': (
                       None, 16, 1920, 1080, 3)}
down_1 = Conv3D(**down_params)
