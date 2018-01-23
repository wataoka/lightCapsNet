!pip install -q keras

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
            # c.shape =  [batch_size, num_capsule, input_num_capsule]
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
                # b.shape=[batch_size, num_capsule, input_num_capsule]
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
    return layers.Lambda(step, name='primarycap_squash')(outputs)


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

import math


def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image
  
  

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    MNISTに関するカプセルネットワーク

    :param input_shape: 入力データのshape(3次元で[width, height, channels]という形)
    :param n_class: クラスの数
    :param routings: routingを行う回数

    :return 2つのmodel (1つ目:学習用モデル, 2つ目:評価用モデル)
            `eval_model`というモデルも学習用としてしようすることもできる.
    """
    x = layers.Input(shape=input_shape)


    # 1層目: ただの2次元畳み込み層
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # 2層目: 活性化関数にsquash関数を用いた2次元畳み込み層で,[None ,num_capsule, dim_capsule]という形に変換する
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # 3層目: カプセル層 (routingアルゴリズムはここで行っている)
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # 4層目: ここはカプセルを"長さ"に変形するための補助レイヤーで, 教師データの形に合わせている.
    # tensorflowを使用している場合, ここは必要ありません.
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoderネットワーク
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # 正解のラベルはよく学習のためにカプセル層の出力を隠す
    masked = Mask()(digitcaps)  # マスクは予測のために長さが最大値のカプセルを使用する

    # このDecoderモデルは学習時にも予測時にも使われる.
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # 学習モデルと評価モデル
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # モデルにノイズを加える
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    論文中の方程式(4)にあたる.
    y_trueはone_hot形式です。
    :param y_true: [None, n_classes]
    :param y_pred: [Noen, num_capsule]
    :return: loss値(スカラー)
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    カプセルネットの学習
    :param model: カプセルネットのモデル
    :param data: trainデータもtestデータも含んだタプルデータ, 右のような形式 `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: 学習ずみモデル
    """
    # データ切り分け
    (x_train, y_train), (x_test, y_test) = data

    # モデルをコンパイル
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    データ増強無しの学習をする場合:
    model.fit([x_train, y_train], [y_train, x_train], bacth_size=args.batch_size, epochs=args.epochs,
               validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    としてください
    """

    # 開始 : データ増強ありの学習 ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # MNISTの画像を2ピクセルをずらす
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # shift_fraction=0とすれば, データ増強無しの学習.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]])
    # 終了 : データ増強無しの学習 -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # 学習用データとテストデータを切り分け, シャッフルしたデータ
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # ハイパーパラメータの設定

    !pip install -q easydict
    import easydict
    args = easydict.EasyDict({
        "epochs": 50,
        "batch_size": 100,
        "lr": 0.001,
        "lr_decay": 0.9,
        "lam_recon": 0.392,
        "routings": 3,
        "shift_fraction":0.1,
        "debug":'store_true',
        "save_dir":'./result',
        "testing":'store_true',
        "digit":5,
        "weights":None
    })


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # データの読み込み
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # モデル定義
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary()

    # 学習orテスト
    if args.weights is not None:  # 重みの初期化
        model.load_weights(args.weights)
    if True:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # 重みが与えられたなら, 開始.
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)


