import pathlib

import keras.backend as K
import streamlit as st
import tensorflow as tf
import tensorflow_io as tfio


def load_dicom_image(image_bytes):
    """
    Load DICOM file from given path and decode it using TensorFlow I/O.

    Args:
        dicom_path (str): The path to the DICOM file.

    Returns:
        The decoded DICOM image.
    """
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    return image


def normalize_image(image):
    """
    Normalize the given image to the range [0, 1] by dividing it by the maximum value (2^16-1).

    Args:
        image (tf.Tensor): The image to be normalized.

    Returns:
        The normalized image.
    """
    return tf.cast(image, tf.float32) / tf.constant(65535.0, dtype=tf.float32)


def log_cosh_loss(y_true, y_pred):
    """
    An implementation of log cosh loss based on
    'A survey of loss functions for semantic segmentation'
    by Shruti Jadon

    Loss implementation based on
    https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
    Survey Paper DOI: 10.1109/CIBCB48159.2020.9277638

    Args:
        y_true ():
        y_pred ():

    Returns:

    """
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice_loss = 1 - score
    log_cosh = tf.math.log((tf.exp(dice_loss) + tf.exp(-dice_loss)) / 2.0)
    return log_cosh


@st.cache_resource
def load_model():
    model_path = list(pathlib.Path.cwd().rglob("*.h5"))[0].as_posix()
    model = tf.keras.models.load_model(
        model_path, custom_objects={"log_cosh_loss": log_cosh_loss}
    )
    return model
