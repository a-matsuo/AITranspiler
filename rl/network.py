import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
from tensorflow.keras.regularizers import l2

os.environ["TF_USE_LEGACY_KERAS"] = "1"


class SqueezeExcitation(kl.Layer):
    def __init__(self, filters, reduction=16):
        super().__init__()
        self.filters = filters
        self.reduction = reduction
        self.global_pool = kl.GlobalAveragePooling2D()
        self.dense1 = kl.Dense(filters // reduction, activation='relu', kernel_initializer='he_normal')
        self.dense2 = kl.Dense(filters, activation='sigmoid', kernel_initializer='he_normal')

    def call(self, x):
        # [B, H, W, C] -> [B, C]
        w = self.global_pool(x)
        w = self.dense1(w)
        w = self.dense2(w)
        # [B, C] -> [B, 1, 1, C]
        w = tf.reshape(w, (-1, 1, 1, self.filters))
        return x * w


class BottleneckResBlock(kl.Layer):
    """
    ボトルネック構造のResidual Block:
    Conv(1x1) -> Conv(3x3) -> Conv(1x1) の3層構成
    オプションでSEブロックを挿入可能
    """
    def __init__(self, in_filters, out_filters, use_bias, weight_decay=0.001, stride=1, use_se=False, reduction=16):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.stride = stride
        self.use_se = use_se

        # 1x1 reduce
        self.conv1 = kl.Conv2D(
            out_filters // 4,
            kernel_size=1,
            strides=1,
            use_bias=self.use_bias,
            kernel_regularizer=l2(weight_decay),
            kernel_initializer="he_normal"
        )
        self.bn1 = kl.BatchNormalization()

        # 3x3 conv
        self.conv2 = kl.Conv2D(
            out_filters // 4,
            kernel_size=3,
            strides=stride,
            padding='same',
            use_bias=self.use_bias,
            kernel_regularizer=l2(weight_decay),
            kernel_initializer="he_normal"
        )
        self.bn2 = kl.BatchNormalization()

        # 1x1 expand
        self.conv3 = kl.Conv2D(
            out_filters,
            kernel_size=1,
            strides=1,
            use_bias=self.use_bias,
            kernel_regularizer=l2(weight_decay),
            kernel_initializer="he_normal"
        )
        self.bn3 = kl.BatchNormalization()

        # Squeeze-and-Excitationブロック（オプション）
        if self.use_se:
            self.se = SqueezeExcitation(out_filters, reduction=reduction)
        else:
            self.se = None

        # ショートカット経路
        if stride != 1 or in_filters != out_filters:
            # ダウンサンプリングやフィルタ数が変わる場合はショートカットにConvを使用
            self.shortcut_conv = kl.Conv2D(
                out_filters,
                kernel_size=1,
                strides=stride,
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                kernel_initializer="he_normal"
            )
            self.shortcut_bn = kl.BatchNormalization()
        else:
            self.shortcut_conv = None

    def call(self, x, training=False):
        shortcut = x

        x = self.bn1(self.conv1(x), training=training)
        x = tf.nn.relu(x)

        x = self.bn2(self.conv2(x), training=training)
        x = tf.nn.relu(x)

        x = self.bn3(self.conv3(x), training=training)

        if self.se is not None:
            x = self.se(x)

        # ショートカット処理
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_bn(self.shortcut_conv(shortcut), training=training)

        x = x + shortcut
        x = tf.nn.relu(x)

        return x


class ComplexResNet(keras.Model):
    def __init__(self, action_space: int, config: dict):
        """
        より複雑なResNetベースのモデルを定義します。

        - BottleneckスタイルのResidual Blockを使用
        - Squeeze-and-Excitationモジュール（オプション）
        - 階層的にフィルタ数を増加させる設定
        - ポリシーおよびバリューの両ヘッドをconfigに基づき柔軟に調整

        :param action_space: 行動空間のサイズ
        :param config: 設定用dict
        """
        super().__init__()

        network_settings = config["network_settings"]
        self.action_space = action_space
        self.n_blocks = network_settings.get("n_blocks", [3, 4, 6, 3])
        # 例: ResNet50ライクな構造(3,4,6,3)ブロック数
        self.initial_filters = network_settings.get("filters", 64)
        self.use_bias = network_settings.get("use_bias", False)
        self.weight_decay = network_settings.get("weight_decay", 0.001)
        self.dropout_rate = network_settings.get("dropout_rate", 0.3)
        self.use_se = network_settings.get("use_se", False)
        self.filter_expansion = network_settings.get("filter_expansion", [64, 128, 256, 512])
        # ステージごとにフィルタ数増やす設定
        # 例えば[64, 128, 256, 512]はResNet50系統の典型的なフィルタサイズ

        # 初回のConv層
        self.conv1 = kl.Conv2D(
            self.filter_expansion[0],
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=self.use_bias,
            kernel_regularizer=l2(self.weight_decay),
            kernel_initializer="he_normal",
        )
        self.bn1 = kl.BatchNormalization()
        self.pool = kl.MaxPooling2D(pool_size=3, strides=2, padding='same')

        # ステージごとのBottleneck Blocks
        self.stages = []
        in_filters = self.filter_expansion[0]
        for i, (num_block, out_filters) in enumerate(zip(self.n_blocks, self.filter_expansion)):
            blocks = []
            stride = 1 if i == 0 else 2  # ステージの最初のブロックでストライド2によるダウンサンプリング
            for b in range(num_block):
                if b == 0:
                    blocks.append(BottleneckResBlock(in_filters, out_filters, self.use_bias,
                                                     weight_decay=self.weight_decay,
                                                     stride=stride,
                                                     use_se=self.use_se))
                else:
                    blocks.append(BottleneckResBlock(out_filters, out_filters, self.use_bias,
                                                     weight_decay=self.weight_decay,
                                                     stride=1,
                                                     use_se=self.use_se))
                in_filters = out_filters
            self.stages.append(blocks)

        # ポリシーヘッド
        self.policy_head = self._build_head(
            num_filters=2,
            output_dim=self.action_space,
            activation="softmax"
        )

        # バリューヘッド
        self.value_head = self._build_head(
            num_filters=1,
            output_dim=1,
            activation="tanh"
        )

        # Global Average Pooling
        self.gap = kl.GlobalAveragePooling2D()

    def _build_head(self, num_filters: int, output_dim: int, activation: str = None):
        """
        ポリシーまたはバリューヘッドを構築するヘルパー関数
        """
        model = keras.Sequential()
        model.add(
            kl.Conv2D(
                num_filters,
                kernel_size=1,
                use_bias=self.use_bias,
                kernel_regularizer=l2(self.weight_decay),
                kernel_initializer="he_normal",
            )
        )
        model.add(kl.BatchNormalization())
        model.add(kl.ReLU())
        model.add(kl.Dropout(self.dropout_rate))
        model.add(kl.Flatten())
        model.add(
            kl.Dense(
                output_dim,
                activation=activation,
                kernel_regularizer=l2(self.weight_decay),
                kernel_initializer="he_normal",
            )
        )
        return model

    def call(self, inputs, training=False):
        # ステム部分
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        # ボトルネックブロック群
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x, training=training)

        # Global Average Pooling後にヘッドを適用
        # このあたりはAlphaZero風にする場合はFlattenする代わりにGAPなどを適用することが多い
        x_gap = self.gap(x)  # [B, C]

        # ポリシー・バリューはConv→BN→ReLU→Dropout→Flatten→Denseだが
        # x_gapはすでにFlatten同等なので、ヘッド側でのFlattenはConv2Dを通すためのもの
        # ここではConv2D -> BN -> ReLUまでがヘッド内で実行されるため、x_gapを4Dに戻す必要なし
        # （ヘッドはSequentialでConv2Dがあるため、その前に拡張が必要）
        # 対処法: ヘッドをGAP後ではなく、GAP前に適用する構造にもできるが、
        # ここではヘッドに合わせるためにチャンネル方向の1x1 Conv適用前に
        # [B, C] -> [B, 1, 1, C] にreshapeしてConvを通す。

        # [B, C] -> [B, 1, 1, C]
        x_reshape = tf.reshape(x_gap, [-1, 1, 1, x_gap.shape[-1]])

        policy_output = self.policy_head(x_reshape, training=training)
        value_output = self.value_head(x_reshape, training=training)

        return policy_output, value_output

    def predict(self, state):
        if len(state.shape) == 3:
            state = state[None, ...]
        policy, value = self(state, training=False)
        return policy.numpy(), value.numpy()