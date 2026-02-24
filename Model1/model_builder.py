import tensorflow as tf
from tensorflow.keras import layers, models


class SaliencyModelBuilder:

    def __init__(self, config):
        self.cfg = config

    def build(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.cfg.INPUT_SIZE,
                         self.cfg.INPUT_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )

        layer_names = [
            "block_13_expand_relu",
            "block_6_expand_relu",
            "block_3_expand_relu"
        ]

        skip_outputs = [
            base_model.get_layer(name).output
            for name in layer_names
        ]

        encoder = models.Model(
            base_model.input,
            skip_outputs + [base_model.output]
        )

        inputs = layers.Input(
            shape=(self.cfg.INPUT_SIZE,
                   self.cfg.INPUT_SIZE, 3)
        )

        x = layers.Rescaling(1./127.5,
                             offset=-1.0)(inputs)

        feats = encoder(x)
        skips, bneck = feats[:-1], feats[-1]

        x = bneck

        for i, filters in enumerate([128, 64, 32]):
            x = layers.UpSampling2D((2, 2))(x)
            x = layers.Resizing(
                skips[i].shape[1],
                skips[i].shape[2]
            )(x)
            x = layers.Concatenate()([x, skips[i]])
            x = layers.SpatialDropout2D(0.2)(x)
            x = layers.Conv2D(
                filters, 3,
                padding="same",
                activation="relu"
            )(x)
            x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(
            16, 3,
            padding="same",
            activation="relu"
        )(x)

        x = layers.Resizing(
            self.cfg.INPUT_SIZE,
            self.cfg.INPUT_SIZE
        )(x)

        outputs = layers.Conv2D(
            1, 1,
            activation="sigmoid"
        )(x)

        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='binary_crossentropy'
        )

        return model