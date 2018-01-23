"""
カプセルネットを構成する重要なレイヤー.
これらのレイヤーは他のデータセットでも使用することができます.
関数は様々な場面で使用することができます. ぜひ使ってみてください.
"""

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


class Length(layers.Layer):
    """
    ベクトルの長さを計算する.
    y_trueと同じ形のテンソルを計算する.
    このレイヤーをモデルの出力として使用することは, `y_pred = np.argmax(mode.predict(x), 1)`としてラベルの予想したことと同じです.
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    shapeが[None, num_capsule, dim_vector]のテンソルをマスク処理する.
    最大値ベクトルを除き, 全てのベクトルを0にマスク処理する.
    そして, マスク処理されたテンソルを1次元にする.
    例:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # 正しいラベルはshape=[None, n_class]で与えられる. すなわちone-hotである.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # 正しくないラベルは, 最大値ラベルでマスクされる.
            # カプセルの長さを計算
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # one-hotで記述されたマスクを生成
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def step(vectors, axis=-1):
    """
    カプセルネットワークでは非線形の活性化関数が使用される. この関数はベクトルの長さを0~1に圧縮する.
    :param vectors: 圧出される複数のベクトル, 4次元テンソル
    :param axis: 圧縮する軸
    :return: 複数の入力ベクトルと同じ形の一つのテンソル
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    return vectors / s_squared_norm


class CapsuleLayer(layers.Layer):
    """
    カプセルレイヤーは全結合層と似ているが,
    カプセルレイヤーはニューロンの出力値はスカラーからベクトルに拡大する.
    だから入力値のshapeは [None, input_num_capsule, input_dim_capsule] であり, 出力値のshapeは [None, num_capsule, dim_capsule] である.

    :param num_capsule: カプセルの数
    :param dim_capsule: 出力カプセルの次元数
    :param routings: routingアルゴリズムを繰り返す回数
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # 行列を変形
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # 重みWにかけられる準備をするためにカプセルを複製する.
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # `inputs * W`を計算
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # はじめの2次元をバッチ次元数としてみなす.
        # そして [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # 開始: Routingアルゴリズム ---------------------------------------------------------------------#
        # 前処理として係数を1に初期化
        # c.shape = [None, self.num_capsule, self.input_num_capsule].
        c = tf.ones(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = step(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # c.shape=[batch_size, num_capsule, input_num_capsule]
                c += K.batch_dot(outputs, inputs_hat, [2, 3])
        # 終了: Routingアルゴリズム -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    畳み込みを`n_channels`回行い, 全てのカプセルを結合する.
    :param inputs: 4次元テンソル, shape=[None, width, height, channels]
    :param dim_capsule: カプセルの出力ベクトルの次元数
    :param n_channels: カプセルの種類の数
    :return: 出力テンソル, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(step , name='primarycap_squash')(outputs)


"""
# 下記はprimaryカプセルレイヤーの別の実装方法である. この実装は速度が遅い.
# 畳み込みを`n_channels`回行い, 全てのカプセルを結合する.
def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    outputs = []
    for _ in range(n_channels):
        output = layers.Conv2D(filters=dim_capsule, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
        outputs.append(layers.Reshape([output.get_shape().as_list()[1] ** 2, dim_capsule])(output))
    outputs = layers.Concatenate(axis=1)(outputs)
    return layers.Lambda(squash)(outputs)
"""
