import tensorflow as tf


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)

# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255.0, test_images / 255.0
labels_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 输出前25个图片查看内容
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # 由于 CIFAR 的标签是 array，
#     # 因此您需要额外的索引（index）。
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = models.Sequential()

# conv_1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# conv_2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# conv_3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# flatten from 3d to 1d
model.add(layers.Flatten())
# fc_layer_1
model.add(layers.Dense(64, activation='relu'))
# fc_layer_2
model.add(layers.Dense(10))

model.summary()

# train
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))

# show
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

model.summary()
