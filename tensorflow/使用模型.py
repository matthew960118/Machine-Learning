import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2,os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 加载模型
model = tf.keras.models.load_model("./savemodel/model1")

img = cv2.imread("img5.png")
img = 255-img
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
img = cv2.resize(img, (28, 28))  # 调整图像大小为模型期望的大小
cv2.imshow("aa",img)
cv2.waitKey(0)
img = img.reshape(1, 784)  # 转换为模型期望的形状


# 进行预测
predictions = model.predict(img)
predicted_class = np.argmax(predictions)

# 打印预测结果
print("Predicted class:", predicted_class)
