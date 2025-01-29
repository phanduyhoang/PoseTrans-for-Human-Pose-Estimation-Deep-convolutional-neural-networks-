import tensorflow as tf

devices = tf.config.list_physical_devices('GPU')
if devices:
    print(f"GPU detected: {devices}")
else:
    print("No GPU detected in TensorFlow.")
