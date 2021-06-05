import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import os
import gc
import cv2
import numpy as np


# 读取目录下的所有图片
def readDirectoryImages(path):
    images_array = {}
    for filename in os.listdir(path):
        img = cv2.imread(path + '/' + filename)
        # 生成索引
        if 'mask' in filename or 'matte' in filename:
            index_mask = int(filename[0: filename.index('_')])
            images_array[index_mask] = img
        else:
            index_images = int(filename[0: filename.index('.')])
            images_array[index_images] = img
    return images_array


data_home = '/Users/mmq/Downloads/Portrait/'
portrait_images = readDirectoryImages(data_home + 'images_data_crop')
mask_images = readDirectoryImages(data_home + 'GT_png')


# 预处理图片
def preprocessImage(pure_image, pure_mask, is_train):
    # mask 降维
    pure_mask = tf.reduce_max(pure_mask, axis=2)
    pure_mask = tf.reshape(pure_mask, shape=(800, 600, 1))

    pure_image = tf.image.resize(pure_image, (512, 512)) / 255.0
    pure_mask = tf.image.resize(pure_mask, (512, 512)) / 255.0

    if is_train and tf.random.uniform(()) > 0.5:
        pure_image = tf.image.flip_left_right(pure_image)
        pure_mask = tf.image.flip_left_right(pure_mask)

    return pure_image, pure_mask


# 训练数据预处理
train_images, train_masks = [], []
for index in portrait_images.keys():
    result_img, result_mask = preprocessImage(portrait_images[index], mask_images[index], True)
    train_images.append(result_img)
    train_masks.append(result_mask)

# 提前释放变量
del portrait_images
del mask_images
gc.collect()


# # 图片展示
# def display(pics):
#     plt.figure(figsize=(15, 15))
#
#     title = ['Input Image', 'True Mask', 'Predicted Mask']
#
#     for idx in range(len(pics)):
#         plt.subplot(1, len(pics), idx + 1)
#         plt.title(title[idx])
#         plt.imshow(tf.keras.preprocessing.image.array_to_img(pics[idx]))
#         plt.axis('off')
#
#     plt.show()
#
#
# display([train_images[0], train_masks[0]])

OUTPUT_CHANNELS = 3
base_model = tf.keras.applications.MobileNetV2(input_shape=[512, 512, 3], include_top=False)

# 使用哪些layers
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建特征提取模型
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False

# 创建上采样模型
up_stack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]


# unet 模型
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])
    x = inputs

    # 在模型中降频取样
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样然后建立跳跃连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # 这是模型的最后一层
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 64x64 -> 512x512

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# 定义和训练unet模型
model = unet_model(OUTPUT_CHANNELS)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.summary()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset_in=None, num=1):
    if dataset_in:
        for image_in, mask_in in dataset_in.take(num):
            pred_mask = model.predict(image_in)
            display([image_in[0], mask_in[0], create_mask(pred_mask)])
    else:
        display([train_images[0], train_masks[0],
                 create_mask(model.predict(train_images[0]))])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


# datasets = tf.data.Dataset.zip((tuple(train_images), tuple(train_masks)))
# for data_index in range(len(train_images)):
#    datasets.append((train_images[data_index], train_masks[data_index]))

train_images = tf.data.Dataset.from_tensor_slices(train_images)
train_masks = tf.data.Dataset.from_tensor_slices(train_masks)
train_images = train_images.cache().batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_masks = train_masks.cache().batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
data = tf.data.Dataset.zip((train_images, train_masks))

model_history = model.fit(data, epochs=20)
                          # steps_per_epoch=len(train_images),
                          # validation_steps=10,
                          # validation_data=test_images,
                          # callbacks=[DisplayCallback()])

model.save('pic_segmentation_model_5.h5')
