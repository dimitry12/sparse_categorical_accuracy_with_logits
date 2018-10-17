def sparse_categorical_accuracy_with_logits(y_true, y_pred_logits):
    # convert dense predictions to labels
    y_pred = tf.nn.softmax(y_pred_logits)
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_pred_labels = tf.cast(y_pred_labels, tf.keras.backend.floatx())
    y_true = tf.cast(y_true, tf.keras.backend.floatx())
    return tf.cast(tf.equal(y_true, y_pred_labels), tf.keras.backend.floatx())
