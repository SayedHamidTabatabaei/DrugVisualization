import numpy as np
import tensorflow as tf
from sklearn.utils import compute_class_weight

from common.enums.loss_functions import LossFunctions


def get_loss_function(loss_type: LossFunctions, class_weights: dict = None):
    if loss_type == LossFunctions.categorical_crossentropy:
        return "categorical_crossentropy"
    elif loss_type == LossFunctions.focal:
        return focal_loss_with_class_weights(gamma=2., class_weights=class_weights)
    elif loss_type == LossFunctions.focal_tversky:
        return focal_tversky_loss(alpha=0.7, beta=0.3, gamma=0.75, class_weights=class_weights)

    # Without class weights
    elif loss_type == LossFunctions.hinge:
        return "hinge"
    elif loss_type == LossFunctions.squared_hinge:
        return "squared_hinge"
    elif loss_type == LossFunctions.dice_loss:
        return dice_loss


def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true, axis=-1) + tf.reduce_sum(y_pred, axis=-1) + smooth)


def focal_loss_with_class_weights(gamma=2., class_weights=None):
    """
    Focal loss with class weights.
    gamma: Focusing parameter.
    alpha: Balancing factor (default is 0.25 for the foreground in binary classification).
    class_weights: List or array of weights for each class.
    """

    def focal_loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # Compute the focal loss
        cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        probs = tf.math.softmax(y_pred, axis=-1)
        if class_weights is not None:
            weights = tf.reduce_sum(class_weights * y_true, axis=-1)
            loss = weights * (1 - probs) ** gamma * cross_entropy_loss
        else:
            loss = (1 - probs) ** gamma * cross_entropy_loss
        return tf.reduce_mean(loss)

    return focal_loss


def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=0.75, smooth=1, class_weights=None):
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        true_pos = tf.reduce_sum(y_true * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)

        tversky_index = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)

        # Apply class weights to adjust loss for each class
        if class_weights is not None:
            weights = tf.reduce_sum(class_weights * y_true)
            return weights * tf.pow((1 - tversky_index), gamma)
        else:
            return tf.pow((1 - tversky_index), gamma)

    return loss


def get_class_weights(y_train):
    y_train = np.ravel(y_train)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return dict(zip(np.unique(y_train), class_weights))
