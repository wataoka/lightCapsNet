# light-CapsNet

## 背景
 2017年11月に提案されたカプセルネットワークの実行速度を上げる.
論文へのリンクは[こちら](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf)


## 変更点
- squash関数をベクトル版step関数に変更
- routingアルゴリズムの変数bを削除

### ■step関数
```python
def step(vectors, axis=-1):
    """
    カプセルネットワークでは非線形の活性化関数が使用される. この関数はベクトルの長さを0~1に圧縮する.
    :param vectors: 圧出される複数のベクトル, 4次元テンソル
    :param axis: 圧縮する軸
    :return: 複数の入力ベクトルと同じ形の一つのテンソル
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    return vectors / s_squared_norm
```

### ■light-rouringアルゴリズム
```python
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
```

## 使用方法

### ■Step 1. インストール
[TensorFlow>=1.2](https://github.com/tensorflow/tensorflow)
[Keras>=2.0.7](https://github.com/fchollet/keras)をインストール 
```
pip install tensorflow-gpu
pip install keras
```

### ■Step 2. リポジトリをクローン
```
git clone https://github.com/XifengGuo/CapsNet-Keras.git capsnet-keras
cd capsnet-keras
```

### ■Step 3. 実行

デフォルト設定
```
python capsulenet.py
```

ヘルプ機能
```
python capsulenet.py -h
```

### ■Step 4. モデル検証

下記のコマンドで`result/trained_model.h5`にモデルを保存することができます.
```
$ python capsulenet.py -t -w result/trained_model.h5
```
テストaccuracyと再構成された画像を出力してくれます.

学習済みモデルのダウンロードは[こちら](https://pan.baidu.com/s/1sldqQo1)


### ■Step 5. GPUで学習

(注)Keras 2.0.9が必要ですので満たしていない方はアップデートをしてください.  
```
python capsulenet-multi-gpu.py --gpus 2
```
このコマンドで自動的にGPUを用いて処理してくれます. なお,トレーニング中はaccuracyを出力しません.

## 別の手法

- PyTorch:
  - [XifengGuo/CapsNet-Pytorch](https://github.com/XifengGuo/CapsNet-Pytorch)
  - [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
  - [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
  - [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)
  - [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
  
- TensorFlow:
  - [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git)   
  I referred to some functions in this repository.
  - [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)   
  - [chrislybaer/capsules-tensorflow](https://github.com/chrislybaer/capsules-tensorflow)

- MXNet:
  - [AaronLeong/CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)
  
- Chainer:
  - [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)

- Matlab:
  - [yechengxi/LightCapsNet](https://github.com/yechengxi/LightCapsNet)

## リンク集
- python 
    - [python入門](http://www.tohoho-web.com/python/)
- keras
    - [keras公式ドキュメント](https://keras.io/ja/)
- ディープラーニング
    - [機械学習ざっくりまとめ](https://qiita.com/frost_star/items/21de02ce0d77a156f53d)
    - [オススメの本](https://www.oreilly.co.jp/books/9784873117584/)
- カプセルネットワーク 
    - [元論文](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf)
    - [世界一わかりやすいカプセルネットワーク](http://blog.aidemy.net/entry/2017/12/03/052302)
    - [カプセルネットワークはニューラルネットワークを超えるか。](https://qiita.com/hiyoko9t/items/f426cba38b6ca1a7aa2b)
    - [グーグルの天才AI研究者、ニューラルネットワークを超えるカプセルネットワークを発表](https://wired.jp/2017/11/28/google-capsule-networks/)
    - [CapsNetのPytorch実装](https://qiita.com/motokimura/items/cae9defed10cb5efeb62)
