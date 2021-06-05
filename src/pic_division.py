import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# show images for test
# plt.figure(figsize=(5, 5))
# for index in range(5):
#     plt.subplot(5, 5, index + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[index], cmap=plt.cm.binary)
#     plt.xlabel(label_name[train_labels[index]])
# plt.show()

nn = keras.Sequential(
    layers=[
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ]
)

nn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

nn.fit(train_images, train_labels, epochs=20)
nn.evaluate(test_images, test_labels)
