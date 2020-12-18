import numpy as np
import tensorflow as tf

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("Physical GPUs: {}, Logical GPUs: {}".format(len(gpus), len(logical_gpus)))
else:
    print("CPU only")

x = np.arange(-1, 1, 0.0001)
y = 0.8 * x + 0.2

model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation=None)])
model.compile("sgd", "mse")
model.build(input_shape=(0,1))
model.summary()
model.fit(x, y, epochs=5)

print("ground truth: 0.8, 0.2")
print("estimated: ", model.variables[0][0,0].numpy(), model.variables[1][0].numpy())
