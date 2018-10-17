# Sparse categorical accuracy with logits

This is a metric to use with Keras where:
- labels are indices corresponding to a class, and
- targets are logits

This is an iteration on Keras's built-in `sparse_categorical_accuracy`-metric to use with models where `sparse_softmax_cross_entropy_with_logits`-loss is used.
