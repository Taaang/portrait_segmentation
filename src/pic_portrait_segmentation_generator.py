import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
import os
import gc
import cv2


# 读取目录下的所有图片
def readDirectoryImages(path):
    images_array = {}
    for filename in os.listdir(path):
        imagePath = path + '/' + filename
        # 生成索引
        if 'mask' in filename:
            index_mask = int(filename[0: filename.index('_')])
            images_array[index_mask] = imagePath
        else:
            index_images = int(filename[0: filename.index('.')])
            images_array[index_images] = imagePath
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


# 图片展示
def display(pics):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for idx in range(len(pics)):
        plt.subplot(1, len(pics), idx + 1)
        plt.title(title[idx])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pics[idx]))
        plt.axis('off')

    plt.show()


def data_generator(path, batch_size):
    # 获取训练数据文件路径
    train_image_path = readDirectoryImages(path + "images_data_crop")
    train_mask_path = readDirectoryImages(path + "GT_png")

    images, masks = [], []
    while 1:
        for (key, image_path) in train_image_path.items():
            mask_path = train_mask_path[key]
            image, mask = preprocessImage(cv2.imread(image_path), cv2.imread(mask_path), True)
            images.append(image)
            masks.append(mask)

            if len(images) == batch_size:
                image_tensor = np.array(images).reshape((-1, 512, 512, 3))
                masks_tensor = np.array(masks).reshape((-1, 512, 512, 1))
                images.clear()
                masks.clear()
                yield image_tensor, masks_tensor


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

DATA_LENGTH = len(portrait_images)
BATCH_SIZE = 1
STEP_PER_EPOCH = DATA_LENGTH / BATCH_SIZE
model_history = model.fit(data_generator(data_home, BATCH_SIZE), steps_per_epoch=STEP_PER_EPOCH, epochs=20)
model.save('pic_segmentation_model_5.h5')
