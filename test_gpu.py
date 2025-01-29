import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

# ✅ Enable GPU memory growth (prevents full allocation)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Enabled GPU memory growth")
    except RuntimeError as e:
        print(e)

# ✅ Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data for CNN input (add channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

# ✅ Define a simple CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# ✅ Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train the model & Measure GPU speed
start_time = time.time()

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

end_time = time.time()
print(f"🔥 Training Time: {end_time - start_time:.2f} seconds")

# ✅ Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ✅ Save the trained model
model.save("mnist_cnn_model.h5")
