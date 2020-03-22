import keras
from keras_contrib.layers import CRF
from keras_wc_embd import get_embedding_layer


import keras.backend as K
from keras.layers import Conv1D, SpatialDropout1D,Dense, Dropout,Activation, Lambda
from keras.models import Model

def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def residual_block(x,  i, activation, filters, kernel_size):
    original_x = x
    conv = Conv1D(filters=filters, kernel_size=kernel_size,
                  dilation_rate=2 ** i, padding='causal',
                  )(x)
    if activation == 'norm_relu':
        x = Lambda(channel_normalization)(conv)
        x = Activation('relu')(x)

    x = SpatialDropout1D(0.5)(x)

    res_x = keras.layers.add([original_x, x])
    return res_x
#
def build_model(word_dict_len,
                char_dict_len,
                max_word_len,
                output_dim,
                word_dim,
                char_dim,
                char_embd_dim=25,
                word_embd_weights=None):

    inputs, embd_layer = get_embedding_layer(
        word_dict_len=word_dict_len,
        char_dict_len=char_dict_len,
        max_word_len=max_word_len,
        word_embd_dim=word_dim,
        char_hidden_dim=char_dim // 2,
        char_embd_dim=char_embd_dim,
        word_embd_weights=word_embd_weights,
    )
    x = Dropout(0.8)(embd_layer)
    # x = embd_layer
    x = Conv1D(128, 3, padding='causal', name='initial_conv')(
        x)  # input_shape（一批次数据个数，每条数据长度）
    dilatations = [0,1,2,3]
    x2 = x
    for i in dilatations:
        x = residual_block(x, i, 'norm_relu', 128, 3)
    for i in dilatations:
        x2 = residual_block(x2, i, 'norm_relu', 128, 5)
    x = keras.layers.add([x, x2])
    crf = CRF(output_dim, sparse_target=True)
    output_layer = crf(x)
    model = Model(inputs, output_layer)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=crf.loss_function,
        metrics=[crf.accuracy],
    )
    return model

    # inputs, embd_layer = get_embedding_layer(
    #     word_dict_len=word_dict_len,
    #     char_dict_len=char_dict_len,
    #     max_word_len=max_word_len,
    #     word_embd_dim=word_dim,
    #     char_hidden_dim=char_dim // 2,
    #     char_embd_dim=char_embd_dim,
    #     word_embd_weights=word_embd_weights,
    #
    # )
    # rnn_type = 'lstm'
    # if rnn_type == 'gru':
    #     rnn = keras.layers.GRU
    # else:
    #     rnn = keras.layers.LSTM
    # dropout_layer_1 = keras.layers.Dropout(rate=0.25, name='Dropout-1')(embd_layer)
    # bi_rnn_layer_1 = keras.layers.Bidirectional(
    #     layer=rnn(
    #         units=300,
    #         dropout=0.0,
    #         recurrent_dropout=0.0,
    #         return_sequences=True,
    #     ),
    #     name='Bi-RNN-1',
    # )(dropout_layer_1)
    # # lm_layer = get_feature_layers(input_layer=inputs[0])#bi_lm_model.
    # # embd_lm_layer = keras.layers.Concatenate(name='Embd-Bi-LM-Feature')([bi_rnn_layer_1, lm_layer])
    # dropout_layer_2 = keras.layers.Dropout(rate=0.25, name='Dropout-2')(bi_rnn_layer_1)
    # bi_rnn_layer_2 = keras.layers.Bidirectional(
    #     layer=rnn(
    #         units=300,
    #         dropout=0.0,
    #         recurrent_dropout=0.0,
    #         return_sequences=True,
    #     ),
    #     name='Bi-RNN-2',
    # )(dropout_layer_2)
    # dense_layer = keras.layers.Dense(units=output_dim, name='Dense')(bi_rnn_layer_2)
    #
    # crf_layer = CRF(
    #     units=output_dim,
    #     sparse_target=True,
    #     name='CRF',
    # )
    # model = keras.models.Model(inputs=inputs, outputs=crf_layer(dense_layer))
    # model.summary()
    # model.compile(
    #     optimizer=keras.optimizers.Adam(),
    #     loss=crf_layer.loss_function,
    #     metrics=[crf_layer.accuracy],
    # )
    # return model
