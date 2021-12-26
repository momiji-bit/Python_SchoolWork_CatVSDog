# ----------------------------------------------------------------------------------------------------------------------
# 导入一些第三方包
# ----------------------------------------------------------------------------------------------------------------------
import tkinter as tk
import tkinter.filedialog
import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
import imutils
import cv2
import os


def predict(imgPath, modelPath='./model.h5'):
    model = tf.keras.models.load_model(modelPath)
    img = cv2.imread(imgPath)
    shape = img.shape[:2]
    r = min(224 / shape[0], 224 / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(src=img,
                             top=int((224 - img.shape[0]) / 2),
                             bottom=int((224 - img.shape[0] + 1) / 2),
                             left=int((224 - img.shape[1]) / 2),
                             right=int((224 - img.shape[1] + 1) / 2),
                             borderType=cv2.BORDER_CONSTANT, dst=None, value=[0, 0, 0])
    img = img.reshape(-1, 224, 224, 3).astype('float32') / 255

    prediction = model.predict(img)
    final_prediction = [result.argmax() for result in prediction][0]
    probability = np.max(prediction)

    # print(imgPath)
    # print(probability)
    # print(['cat', 'dog'][final_prediction])
    outText.insert("end", str(probability * 100) + '%')
    outText.insert("end", '\n')
    outText.insert("end", "It's a " + str(['cat', 'dog'][final_prediction]))


def train(modelSavePath='./model.h5', datasetsPath='./datasets'):
    def VGG16():
        """
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        conv2d (Conv2D)              (None, 224, 224, 64)      1792
        _________________________________________________________________
        conv2d_1 (Conv2D)            (None, 224, 224, 64)      36928
        _________________________________________________________________
        max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 112, 112, 128)     73856
        _________________________________________________________________
        conv2d_3 (Conv2D)            (None, 112, 112, 128)     147584
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 56, 56, 128)       0
        _________________________________________________________________
        conv2d_4 (Conv2D)            (None, 56, 56, 256)       295168
        _________________________________________________________________
        conv2d_5 (Conv2D)            (None, 56, 56, 256)       590080
        _________________________________________________________________
        conv2d_6 (Conv2D)            (None, 56, 56, 256)       590080
        _________________________________________________________________
        max_pooling2d_2 (MaxPooling2 (None, 28, 28, 256)       0
        _________________________________________________________________
        conv2d_7 (Conv2D)            (None, 28, 28, 512)       1180160
        _________________________________________________________________
        conv2d_8 (Conv2D)            (None, 28, 28, 512)       2359808
        _________________________________________________________________
        conv2d_9 (Conv2D)            (None, 28, 28, 512)       2359808
        _________________________________________________________________
        max_pooling2d_3 (MaxPooling2 (None, 14, 14, 512)       0
        _________________________________________________________________
        conv2d_10 (Conv2D)           (None, 14, 14, 512)       2359808
        _________________________________________________________________
        conv2d_11 (Conv2D)           (None, 14, 14, 512)       2359808
        _________________________________________________________________
        conv2d_12 (Conv2D)           (None, 14, 14, 512)       2359808
        _________________________________________________________________
        max_pooling2d_4 (MaxPooling2 (None, 7, 7, 512)         0
        _________________________________________________________________
        flatten (Flatten)            (None, 25088)             0
        _________________________________________________________________
        dense (Dense)                (None, 4096)              102764544
        _________________________________________________________________
        dense_1 (Dense)              (None, 4096)              16781312
        _________________________________________________________________
        dense_2 (Dense)              (None, 2)                 8194
        =================================================================
        Total params: 134,268,738
        Trainable params: 134,268,738
        Non-trainable params: 0
        """
        model = tf.keras.Sequential()
        # 1
        model.add(tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu,
                                         input_shape=(224, 224, 3)))

        model.add(tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        # 2
        model.add(tf.keras.layers.Conv2D(filters=128,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Conv2D(filters=128,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        # 3
        model.add(tf.keras.layers.Conv2D(filters=256,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Conv2D(filters=256,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Conv2D(filters=256,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        # 4
        model.add(tf.keras.layers.Conv2D(filters=512,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Conv2D(filters=512,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Conv2D(filters=512,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        # 5
        model.add(tf.keras.layers.Conv2D(filters=512,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Conv2D(filters=512,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Conv2D(filters=512,
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=4096,
                                        activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(units=4096,
                                        activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax))

        return model

    # ------------------------------------------------------------------------------------------------------------------
    # EPOCHS = 20 --- 训练的轮数
    # BATCH_SIZE = 8 --- 一次喂入神经网络的数据个数
    # NUM_CLASSES = 2 --- 训练数据的类别数
    # image_height = 224 --- 喂入网络图片的高度
    # image_width = 224 --- 喂入网络图片的宽度
    # channels = 3 --- 训练图片的通道数
    # model_dir = "./model/model.h5" --- 保存模型的位置以及名称
    # train_dir = "./data/train/" --- 训练数据集的位置
    # test_dir = "./data/test/" --- 测试数据集的位置
    # ------------------------------------------------------------------------------------------------------------------
    EPOCHS = int(epochsSV.get())
    BATCH_SIZE = int(batchSizeSV.get())
    NUM_CLASSES = int(numClassesSV.get())
    image_height = 224
    image_width = 224
    model_dir = modelSavePath
    train_dir = datasetsPath + "/train/"
    test_dir = datasetsPath + "/test/"

    # ------------------------------------------------------------------------------------------------------------------
    # def get_datasets(): --- 准备数据集
    # ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等
    # rescale=1.0 / 255.0 --- 归一化处理
    # flow_from_directory --- 第一个参数是文件的位置，第二个参数是目标文件的尺寸，第三个参数是图片文件的颜色空间，
    # 第四个参数是一次喂入神经网络数据的个数，第五个参数是随机种子，第六个参数是是否打乱顺序，第七个参数是ImageDataGenerator的方法.flow_from_directory()加载图片数据流时，
    # 参数class_mode要设为‘categorical’，如果是二分类问题该值可设为‘binary’,
    # train_generator.samples --- 得到训练数据集的样本数量
    # ------------------------------------------------------------------------------------------------------------------
    def get_datasets():
        outTrainText.insert('end', 'getting data...\n')
        cv2.waitKey(500)
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255.0
        )

        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(image_height, image_width),
                                                            color_mode="rgb",
                                                            batch_size=BATCH_SIZE,
                                                            shuffle=True,
                                                            class_mode="categorical")

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255.0
        )
        test_generator = test_datagen.flow_from_directory(test_dir,
                                                          target_size=(image_height, image_width),
                                                          color_mode="rgb",
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=True,
                                                          class_mode="categorical"
                                                          )

        train_num = train_generator.samples
        test_num = test_generator.samples

        return train_generator, test_generator, train_num, test_num

    # ------------------------------------------------------------------------------------------------------------------
    # 网络的初始化 --- model = AlexNet()
    # model.compile --- 对神经网络训练参数是设置 --- tf.keras.losses.categorical_crossentropy --- 损失函数（交叉熵）
    # tf.keras.optimizers.Adam(learning_rate=0.0001) --- 优化器的选择，以及学习率的设置
    # metrics=['accuracy'] ---  List of metrics to be evaluated by the model during training and testing 模型在训练和测试期间要评估的指标列表
    # return model --- 返回初始化之后的模型
    # ------------------------------------------------------------------------------------------------------------------
    def get_model():
        outTrainText.insert('end', 'getting model...\n')
        cv2.waitKey(500)
        model = VGG16()
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                      metrics=['accuracy'])
        return model

    # ------------------------------------------------------------------------------------------------------------------
    # if __name__ == '__main__': --- 主函数，程序运行的入口
    # train_generator, test_generator, train_num, test_num = get_datasets() --- 调用get_datasets()函数
    # model = get_model() --- 得到初始化的模型
    # model.summary() --- 输出模型各层的参数状况
    # model.fit_generator --- 两种训练模型方式fit和fit_generator --- 后者节省内存
    # 第一个参数是训练数据，第二个参数是训练的轮数，第三个参数是每一轮的步数，第四个参数是验证数据集，第五个参数是验证过程的步数、
    # model.save(model_dir) --- 模型保存
    # ------------------------------------------------------------------------------------------------------------------
    train_generator, test_generator, train_num, test_num = get_datasets()
    model = get_model()
    cv2.waitKey(2000)
    model.summary()
    model.fit_generator(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        validation_data=test_generator,
                        validation_steps=test_num // BATCH_SIZE)


    model.save(model_dir)


def click_run():
    imgPath = pathSV.get()
    modelPath = modelPathSV.get()
    predict(imgPath=imgPath, modelPath=modelPath)


def click_train():
    outTrainText.insert('end', 'start training...\n')
    cv2.waitKey(500)
    modelSavePath = modelSavePathSV.get()
    datasetsPath = datasetsPathSV.get()
    train(modelSavePath=modelSavePath, datasetsPath=datasetsPath)


def click_open_img():
    imgOpenPath = tk.filedialog.askopenfilename()
    pathSV.set(imgOpenPath)

    global img  # 防止图片备函数回收
    img = cv2.imread(imgOpenPath)
    shape = img.shape[:2]
    r = min(350 / shape[0], 350 / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(src=img,
                             top=int((350 - img.shape[0]) / 2),
                             bottom=int((350 - img.shape[0] + 1) / 2),
                             left=int((350 - img.shape[1]) / 2),
                             right=int((350 - img.shape[1] + 1) / 2),
                             borderType=cv2.BORDER_CONSTANT, dst=None, value=[0, 0, 0])

    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = ImageTk.PhotoImage(im)
    imgLabel = tk.Label(root, image=img)
    imgLabel.place(x=25, y=35)


def click_open_model():
    modelOpenPath = tk.filedialog.askopenfilename()
    modelPathSV.set(modelOpenPath)


def click_save_model():
    modelSavePath = tk.filedialog.asksaveasfilename(filetypes=[('H5', '.h5')])
    modelSavePathSV.set(modelSavePath)


def click_open_datasets():
    datasetsPath = tk.filedialog.askopenfilename()
    datasetsPathSV.set(datasetsPath)

# ----------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__': --- 主函数，程序运行的入口
# root = tk.Tk() --- 创建Tkinter的主窗口
# root.title('Cat vs Dog') --- 设置主窗口名称
# root.geometry('900x600') --- 设置主窗口大小
# noteLabel = tk.Label(root, text='Predict', width=5, height=1) --- 创建Label标签
# imgCV.place(x=25, y=35) --- 设置标签位于主窗口的位置
# imgCV = tk.Canvas(root, bg='gray', height=350, width=350) --- 创建画布，并将背景设置为灰色
# pathSV = tk.StringVar() --- 创建文本变量
# pathEntry = tk.Entry(root, textvariable=pathSV, width=30) --- 创建文本框
# pathOpenBtn = tk.Button(master=root, text="Open", command=click_open_img, width=4, height=1) --- 创建按钮
# root.mainloop() --- Tkinter主循环
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    root.title('Cat vs Dog')
    root.geometry('900x600')

    noteLabel = tk.Label(root, text='Predict', width=5, height=1)
    noteLabel.place(x=25, y=10)

    imgCV = tk.Canvas(root, bg='gray', height=350, width=350)
    imgCV.place(x=25, y=35)

    pathLabel = tk.Label(root, text='预测图像地址:', width=9, height=1)
    pathLabel.place(x=25, y=390)

    pathSV = tk.StringVar()
    pathEntry = tk.Entry(root, textvariable=pathSV, width=30)
    pathEntry.place(x=25, y=410)

    pathOpenBtn = tk.Button(master=root, text="Open", command=click_open_img, width=4, height=1)
    pathOpenBtn.place(x=310, y=410)

    modelPathLabel = tk.Label(root, text='预训练模型地址:', width=11, height=1)
    modelPathLabel.place(x=25, y=440)

    modelPathSV = tk.StringVar(value='./model.h5')
    modelPathEntry = tk.Entry(root, textvariable=modelPathSV, width=30)
    modelPathEntry.place(x=25, y=460)

    modelOpenBtn = tk.Button(master=root, text="Open", command=click_open_model, width=4, height=1)
    modelOpenBtn.place(x=310, y=460)

    pathLabel = tk.Label(root, text='识别结果:', width=6, height=1)
    pathLabel.place(x=25, y=490)

    outText = tk.Text(root, font=("宋体", 20), width=27, height=2)
    outText.place(x=25, y=510)

    runBtn = tk.Button(master=root, text='识别', command=click_run, width=4, height=1)
    runBtn.place(x=310, y=565)

# ======================================================================================================================
    noteLabel = tk.Label(root, text='Train', width=4, height=1)
    noteLabel.place(x=525, y=10)

    modelSavePathLabel = tk.Label(root, text='模型保存地址:', width=11, height=1)
    modelSavePathLabel.place(x=525, y=35)

    modelSavePathSV = tk.StringVar(value='./model.h5')
    modelSavePathEntry = tk.Entry(root, textvariable=modelSavePathSV, width=30)
    modelSavePathEntry.place(x=525, y=55)

    modelOpenBtn = tk.Button(master=root, text="save as", command=click_save_model, width=4, height=1)
    modelOpenBtn.place(x=810, y=55)

    datasetsLabel = tk.Label(root, text='数据集地址:', width=9, height=1)
    datasetsLabel.place(x=525, y=95)

    datasetsPathSV = tk.StringVar(value='./datasets')
    datasetsPathEntry = tk.Entry(root, textvariable=datasetsPathSV, width=30)
    datasetsPathEntry.place(x=525, y=115)

    pathOpenBtn = tk.Button(master=root, text="Open", command=click_open_datasets, width=4, height=1)
    pathOpenBtn.place(x=810, y=115)

    epochsLabel = tk.Label(root, text='EPOCHS:', width=7, height=1)
    epochsLabel.place(x=525, y=155)
    epochsSV = tk.IntVar(value=20)
    epochsEntry = tk.Entry(root, textvariable=epochsSV, width=5)
    epochsEntry.place(x=600, y=155)

    batchSizeLabel = tk.Label(root, text='BATCH_SIZE:', width=10, height=1)
    batchSizeLabel.place(x=525, y=185)
    batchSizeSV = tk.IntVar(value=32)
    batchSizeEntry = tk.Entry(root, textvariable=batchSizeSV, width=5)
    batchSizeEntry.place(x=625, y=185)

    numClassesLabel = tk.Label(root, text='NUM_CLASSES:', width=12, height=1)
    numClassesLabel.place(x=525, y=215)
    numClassesSV = tk.IntVar(value=2)
    numClassesEntry = tk.Entry(root, textvariable=numClassesSV, width=5)
    numClassesEntry.place(x=640, y=215)

    outputLabel = tk.Label(root, text='Output:', width=6, height=1)
    outputLabel.place(x=525, y=250)

    outTrainText = tk.Text(root, width=50, height=22)
    outTrainText.place(x=525, y=270)

    trainBtn = tk.Button(master=root, text='训练', command=click_train, width=4, height=1)
    trainBtn.place(x=810, y=565)

    root.mainloop()

