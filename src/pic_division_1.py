import tensorflow as tensorflow
import tensorflow.keras as keras
import numpy
import matplotlib.pyplot as pyplot


(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images, test_images = train_images / 255.0, test_images / 255.0

# pyplot.figure(figsize=(5, 5))
# for index in range(0, 5):
#     pyplot.subplot(1, 5, index + 1)
#     pyplot.imshow(train_images[index])
#     pyplot.xlabel(train_labels[index])
# pyplot.show()

model = keras.Sequential(
    layers= [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ]
)

model.compile(
    optimizer='adam',
    loss=tensorflow.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=20)
model.evaluate(test_images, test_labels)

probability_model = keras.Sequential([model, keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
result = numpy.argmax(predictions)
print("test")
