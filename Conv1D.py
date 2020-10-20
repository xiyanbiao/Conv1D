# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:22:57 2020

@author: ZhangYida
"""

# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils,plot_model
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# 载入数据
df = pd.read_csv(r"D:\论文持续更新\数据处理\Conv1D模型\数据集-用做分类.csv")
X = np.expand_dims(df.values[:, 0:246].astype(float), axis=2)
Y = df.values[:, 246]

# 湿度分类编码为数字
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)

# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)

# 定义神经网络
def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(246, 1)))
    model.add(Conv1D(16, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(9, activation='softmax'))
    plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model

# 训练分类器
estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=1, verbose=1)
estimator.fit(X_train, Y_train)

# 卷积网络可视化
# def visual(model, data, num_layer=1):
#     # data:图像array数据
#     # layer:第n层的输出
#     layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
#     f1 = layer([data])[0]
#     print(f1.shape)
#     num = f1.shape[-1]
#     print(num)
#     plt.figure(figsize=(8, 8))
#     for i in range(num):
#         plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i+1)
#         plt.imshow(f1[:, :, i] * 255, cmap='gray')
#         plt.axis('off')
#     plt.show()

# 混淆矩阵定义
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,('0%','3%','5%','8%','10%','12%','15%','18%','20%','25%'))
    plt.yticks(tick_marks,('0%','3%','5%','8%','10%','12%','15%','18%','20%','25%'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()

# seed = 42
# np.random.seed(seed)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
# print("Accuracy of cross validation, mean %.2f, std %.2f\n" % (result.mean(), result.std()))

# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))

# 将其模型转换为json
model_json = estimator.model.to_json()
with open(r"D:\论文持续更新\数据处理\Conv1D模型\model.json",'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构
estimator.model.save_weights('model.h5')

# 加载模型用做预测
json_file = open(r"D:\论文持续更新\数据处理\Conv1D模型\model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# 输出预测类别
predicted = loaded_model.predict(X)
predicted_label = loaded_model.predict_classes(X)
print("predicted label:\n " + str(predicted_label))
#显示混淆矩阵
plot_confuse(estimator.model, X_test, Y_test)

# 可视化卷积层
# visual(estimator.model, X_train, 1)