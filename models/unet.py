"""
models/unet.py
U-Net architecture for binary breast lesion segmentation.
Input:  (224, 224, 3)  float32 normalized image
Output: (224, 224, 1)  sigmoid probability mask
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

from utils.config import IMG_HEIGHT, IMG_WIDTH, CHANNELS, LEARNING_RATE


def se_block(inputs, ratio=8):
    """Squeeze-and-Excitation Block for channel-wise attention."""
    channel_axis = -1
    filters = inputs.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    return layers.Multiply()([inputs, se])


def conv_block(x, filters, kernel_size=3, padding="same", activation="relu", use_attention=True):
    """Two consecutive Conv2D + BN + ReLU layers with optional SE Attention."""
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    if use_attention:
        x = se_block(x)
    return x


def build_unet(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
    filters=(32, 64, 128, 256),
    dropout_rate=0.2,
):
    """
    4-level U-Net with skip connections.
    """
    inputs = layers.Input(shape=input_shape, name="unet_input")

    # ── Encoder ───────────────────────────────────────────────────────────────
    skips = []
    x = inputs
    for f in filters:
        x = conv_block(x, f)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)

    # ── Bottleneck ────────────────────────────────────────────────────────────
    x = conv_block(x, filters[-1] * 2)

    # ── Decoder ───────────────────────────────────────────────────────────────
    for f, skip in zip(reversed(filters), reversed(skips)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Dropout(dropout_rate)(x)
        x = conv_block(x, f)

    # ── Output ────────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="unet_output")(x)

    model = Model(inputs, outputs, name="UNet")
    return model


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """Tversky loss: allows tuning the balance between FP and FN."""
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - tversky


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    """Focal Tversky Loss for hard cases."""
    pt = tversky_loss(y_true, y_pred)
    return K.pow(pt, gamma)


def hybrid_clinical_loss(y_true, y_pred):
    """Weighted hybrid of BCE and Focal Tversky Loss."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    ft  = focal_tversky_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * ft


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def get_compiled_unet(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
    learning_rate=LEARNING_RATE,
):
    """Returns a compiled U-Net model with BCE + Dice Loss for accurate boundaries."""
    model = build_unet(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=hybrid_clinical_loss,
        metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2), dice_coef],
    )
    return model
