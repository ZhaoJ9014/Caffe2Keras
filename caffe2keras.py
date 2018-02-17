def load_net(proto_path, weights_path):
    """Load caffe model"""
    import caffe
    caffe.set_mode_cpu()
    return caffe.Net(proto_path, weights_path, caffe.TEST)

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D

def VGG16():
    """
    Build the architecture of VGG-16.
    """
    img_input = Input(shape=(224, 224, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Block 6
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dense(205, activation='softmax', name='fc8')(x)

    return Model(img_input, x)

def print_layer_info(net):
    """Print the names and the output shapes of the layers"""
    for layer_name, blob in net.blobs.iteritems():
        print('The output shape of layer {} is {}'.format(
            layer_name, blob.data.shape[1:]))

import os
from keras import backend as K

def transform_conv_weight(W):
    return W.T

def load_weights_from_caffe_net(keras_model, caffe_net):
    for layer_name, param in net.params.items():
        layer = keras_model.get_layer(layer_name)
        if not layer:
            continue
        elif layer.__class__.__name__ == 'Conv2D':
            W = transform_conv_weight(param[0].data)
        elif layer.__class__.__name__ == 'Dense':
            W = param[0].data.T
        else:
            continue
        b = param[1].data
        layer.set_weights([W, b])
        print('Setting weights for {:s}'.format(layer.name))

    dense = keras_model.get_layer(name='fc6')
    shape = keras_model.get_layer(name='pool5').output_shape[1:]
    from keras.utils.layer_utils import convert_dense_weights_data_format
    convert_dense_weights_data_format(dense, shape, 'channels_last')
    return keras_model

if __name__ == '__main__':
    PROTO_PATH = '/Users/zhaojian/Desktop/untitled/places205VGG16/deploy_10.prototxt'
    WEIGHTS_PATH = '/Users/zhaojian/Desktop/untitled/places205VGG16/snapshot_iter_765280.caffemodel'
    RESULT_DIR = '/Users/zhaojian/Desktop/untitled/'
    net = load_net(PROTO_PATH, WEIGHTS_PATH)
    print('caffe model summary:')
    print_layer_info(net)
    model = VGG16()
    print('keras model summary:')
    for layer in model.layers:
        print('The output shape of layer {} is {}'.format(
            layer.name, layer.output_shape[1:]))
    model = load_weights_from_caffe_net(model, net)
    model_name = 'vgg16_{}_{}.h5'.format(K.backend(), K.image_data_format())
    model.save(os.path.join(RESULT_DIR, model_name))
    from keras.models import load_model
    model = load_model(model_name)
    model.summary()