import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
import cv2
import numpy as np

# 测试图片
image = cv2.imread("/Users/mmq/Desktop/WechatIMG290.jpeg")
image = tf.image.resize(image, (512, 512)) / 255.0
image = tf.reshape(image, (-1, 512, 512, 3))

model = keras.models.load_model("/Users/mmq/Temporary/pic_segmentation_model_5.h5")
model.summary()
# keras.models.save_model(model, "pic_segmentation_model.tf", save_format="tf")

# result = model.predict(image)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# plt.figure(figsize=(2, 2))
#
# plt.subplot(1, 2, 1)
# plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))
# plt.axis("off")
#
# result_image = result
# #result_image = tf.reduce_max(result_image, 2)
# #result_image = tf.reshape(result_image, (128, 128, 1))
# plt.subplot(1, 2, 2)
# test = tf.keras.preprocessing.image.array_to_img(create_mask(result_image))
# plt.imshow(test)
# plt.axis("off")
# plt.show()


# 读取背景图
background_image = cv2.imread("/Users/mmq/Desktop/backgroud.jpeg")
background_image = tf.image.resize(background_image, (512, 512))

# 调用笔记本内置摄像头
cap = cv2.VideoCapture(0)
while True:
    # 读取摄像头数据
    ret, frames = cap.read()

    # 帧数据预处理
    frames = tf.image.resize(frames, (512, 512)) / 255.0
    frames = tf.reshape(frames, (-1, 512, 512, 3))

    # 模型分析
    result = model.predict(frames)
    result = create_mask(result)

    # 合成图片
    test = cv2.cvtColor(np.array(tf.keras.preprocessing.image.array_to_img(result)), cv2.COLOR_GRAY2RGB)
    input1 = np.array(tf.reshape(frames * 255, (512*512*3)), dtype=np.uint8)
    input2 = np.array(tf.reshape(test, (512*512*3)))
    portrait = cv2.bitwise_and(input1, input2)
    # final_image = cv2.add(np.array(tf.reshape(background_image, (512*512*3)), dtype=np.uint8), portrait)

    # for x in range(result.width):
    #     for y in range(result.height):
    #         if result.getpixel((x, y)) == 0:
    #             final_image[x][y] = np.array(background_image[x][y])
    #         else:
    #             final_image[x][y] = frames[0][x][y]

    cv2.imshow("capture", np.reshape(portrait, (512, 512, 3)))
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
