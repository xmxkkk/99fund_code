from model import Model
import tensorflow as tf
import os, random, time, sys
import numpy as np
import matplotlib.pyplot as plt
from v2_input_data import images,labels
import cv2
import pandas as pd
import urllib
import shutil

class V2Model(Model):
    def __init__(self, model_name, show_val):
        self.images = tf.placeholder(tf.float32, [None, 30, 20, 1])
        self.labels = tf.placeholder(tf.int32, [None, 10])
        self.model_name = model_name
        self.show_val = show_val

        self.loss_vals = []
        self.accuracy_vals = []

        if not os.path.exists('./model/{}'.format(self.model_name)):
            os.makedirs('./model/{}'.format(self.model_name))

        if os.path.exists('./model/{}/{}.npz'.format(self.model_name, self.model_name)):
            vals=np.load('./model/{}/{}.npz'.format(self.model_name,self.model_name))
            self.loss_vals,self.accuracy_vals=vals['loss_vals'].tolist(),vals['accuracy_vals'].tolist()

        self.label_str='0123456789'

    def train(self):
        if self.show_val:
            plt.ion()
            plt.show()

        loss, accuracy, _ = self.build()
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            saver = tf.train.Saver()

            if os.path.exists('./model/{}/{}.index'.format(self.model_name, self.model_name)):
                saver.restore(sess, './model/{}/{}'.format(self.model_name, self.model_name))

            x_train, y_train, x_test, y_test = self.get_file();

            test_idx = 0
            test_batch_size = 40
            test_total = x_test.shape[0]
            test_page_no = int(test_total / test_batch_size)

            batch_size = 100
            total = x_train.shape[0]
            page_no = int(total / batch_size)
            print("{}{}{}".format(batch_size, total, page_no))
            for i in range(sys.maxsize):

                start = (i % page_no * batch_size) % total
                end = (i % page_no * batch_size + batch_size) % total
                if end == 0:
                    end = total

                start_time = time.time()
                _, loss_val, accuracy_val = sess.run(
                    [train_op, loss, accuracy]
                    , feed_dict={self.images: x_train[start:end], self.labels: y_train[start:end]}
                )
                print("no={}      ,loss={:.4f},accuracy={:.4f},time={:.2f}".format(i, loss_val, accuracy_val[0],
                                                                                   time.time() - start_time))
                if i % 200 == 0 and i > 0:
                    test_start = (test_idx % test_page_no * test_batch_size) % test_total
                    test_end = (test_idx % test_page_no * test_batch_size + test_batch_size) % test_total
                    if test_end == 0:
                        test_end = test_total
                    test_idx = test_idx + 1

                    loss_val, accuracy_val = sess.run(
                        [loss, accuracy],
                        feed_dict={self.images: x_test[test_start:test_end], self.labels: y_test[test_start:test_end]}
                    )

                    saver.save(sess, "./model/{}/{}".format(self.model_name, self.model_name))
                    print(accuracy_val)
                    print(i, '       loss_val={:.5f}'.format(loss_val)
                          , '        accuracy_val=', accuracy_val)

                    self.loss_vals.append(loss_val)
                    self.accuracy_vals.append(accuracy_val)

                    self._draw_line()

    def predict(self,datas):
        _, _, y_pred = self.build()
        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            saver = tf.train.Saver()

            if os.path.exists('./model/{}/{}.index'.format(self.model_name, self.model_name)):
                saver.restore(sess, './model/{}/{}'.format(self.model_name, self.model_name))

            batch_size=10
            total=datas.shape[0]
            page_no=int(total/batch_size)

            result=[]
            for i in range(page_no):
                start=i*batch_size
                end=i*batch_size+batch_size

                y_pred_val=sess.run([y_pred],feed_dict={self.images:datas[start:end]})
                ndata=np.array(y_pred_val).reshape((-1,))

                result.append(self.transfer(ndata))
            return np.array(result).flatten()
    def predict_url(self):
        url = 'https://passport.99fund.com/cif/login/loginVerifyCode.htm?time=1528702116568'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = urllib.request.urlopen(req).read()

        with open('0000.png', 'wb') as f:
            f.write(data)

        y_pred, labels = self.predict_image("0000.png")
        print(y_pred)

    def predict_make(self):

        idx=0

        url = 'https://passport.99fund.com/cif/login/loginVerifyCode.htm?time=1528702116568'

        while idx<10000:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req).read()

            file='./newcode/'+str(idx).zfill(4)+'.png'
            with open(file, 'wb') as f:
                f.write(data)

            idx=idx+1
            if idx%8==0:
                time.sleep(1)

    def predict_image_rename(self,files,path):
        images = []

        for file in files:
            if not file.endswith('png'):
                continue
            img = cv2.imread(path+file)
            img = img[:, :, 0]
            img2 = img.flatten()
            abc = pd.value_counts(img2)
            for x in abc.index:
                if abc[x] > 30:
                    abc = abc.drop(x)
            for x in abc.index:
                img[img == x] = 0

            img[img != 0] = 1

            img = img[:, :, np.newaxis]
            images.append(img[:, 7:27, :])
            images.append(img[:, 27:47, :])
            images.append(img[:, 47:67, :])
            images.append(img[:, 67:87, :])


        y_pred=self.predict(np.array(images)).tolist()
        for i in range(int(len(y_pred)/4)):
            shutil.move(path+files[i],'./rename_newcode/'+''.join(y_pred[i*4:i*4+4])+'_'+str(random.randint(0,10000))+'.png')



    def predict_image(self,file):
        img = cv2.imread(file)
        img = img[:, :, 0]
        img2 = img.flatten()
        abc = pd.value_counts(img2)
        for x in abc.index:
            if abc[x] > 30:
                abc = abc.drop(x)
        for x in abc.index:
            img[img == x] = 0

        img[img != 0] = 1

        img = img[:, :, np.newaxis]

        images=[]
        images.append(img[:, 7:27, :])
        images.append(img[:, 27:47, :])
        images.append(img[:, 47:67, :])
        images.append(img[:, 67:87, :])

        label = []
        filename = file[0:4]
        for no in filename:
            xx = [0] * 10
            xx[int(no)] = 1
            label.append(xx)

        labels=[label]
        return self.predict(images),label

    def build(self):
        conv1_1 = tf.layers.conv2d(self.images, 32, kernel_size=(3, 3), activation=tf.nn.relu)
        conv1_2 = tf.layers.conv2d(conv1_1, 32, kernel_size=(3, 3), activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1_2, (3, 3), (2, 2))

        dropout0 = tf.layers.dropout(pool1, rate=0.5)

        conv2_1 = tf.layers.conv2d(dropout0, 128, kernel_size=(3, 3), activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(conv2_1, 128, kernel_size=(3, 3), activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2_2, (3, 3), (2, 2))

        # flatten1=tf.reshape(pool2,(None,-1))
        flatten1 = tf.contrib.layers.flatten(pool2)
        dropout1 = tf.layers.dropout(flatten1, rate=0.5)
        dense2 = tf.layers.dense(dropout1, 10)
        score = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=dense2))
        acc = tf.metrics.accuracy(labels=tf.argmax(self.labels, axis=1),predictions=tf.argmax(dense2, axis=1))
        return score,acc,tf.argmax(dense2, axis=1)
    def _draw_line(self):
        if self.show_val:
            plt.subplot(2, 1, 1)
            plt.title('accuracy')
            plt.plot(range(len(self.accuracy_vals)), self.accuracy_vals, color='b')

            plt.subplot(2, 1, 2)
            plt.title('loss')
            plt.plot(range(len(self.loss_vals)), self.loss_vals, color='r')
            plt.pause(0.1)

        self._save_vals()

    def _save_vals(self):
        np.savez_compressed('./model/{}/{}.npz'.format(self.model_name, self.model_name),
                            loss_vals=np.array(self.loss_vals),accuracy_vals=np.array(self.accuracy_vals))

    def _loss(self, start_p, end_p, no):
        conv1_1 = tf.layers.conv2d(self.images[:, :, start_p:end_p, :], 32, kernel_size=(3, 3), activation=tf.nn.relu)
        conv1_2 = tf.layers.conv2d(conv1_1, 32, kernel_size=(3, 3), activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1_2, (3, 3), (2, 2))

        conv2_1 = tf.layers.conv2d(pool1, 128, kernel_size=(3, 3), activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(conv2_1, 128, kernel_size=(3, 3), activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2_2, (3, 3), (2, 2))

        # flatten1=tf.reshape(pool2,(None,-1))
        flatten1 = tf.contrib.layers.flatten(pool2)
        dropout1 = tf.layers.dropout(flatten1, rate=0.5)
        dense2 = tf.layers.dense(dropout1, 10)
        score = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels[:, no, :], logits=dense2))
        acc = tf.metrics.accuracy(labels=tf.argmax(self.labels[:, no, :], axis=1),
                                  predictions=tf.argmax(dense2, axis=1))

        return score, acc, tf.argmax(dense2, axis=1)

    def get_file(self):
        n_split=3800
        return images[:n_split],labels[:n_split],images[n_split:],labels[n_split:]

    def transfer(self,datas):
        if type(datas)==type([]):
            datas=np.array(datas)

        shp=datas.shape
        lst=datas.flatten().tolist()
        lst=list(map(lambda x:self.label_str[x],lst))
        return np.array(lst).reshape(shp)

model = V2Model('v2', False)

# model.predict_make()
model.predict_image_rename(os.listdir('./newcode/'),'./newcode/')
#
# y_pred,labels=model.predict_image("/Users/xmx/Desktop/pyspace/wutong_code/9427.jpg")
# print(y_pred)
# print(model.transfer(np.argmax(labels,axis=1)))
# model.train()

# y_pred,labels=model.predict_image("zuK5D.jpg","/Users/xmx/Desktop/app/datasets/5code/test/")
#
# print(y_pred)
# print(model.transfer(np.argmax(labels,axis=2)))

# model.train()

# _,_,x_test,y_test=model.get_file()
#
# y_pred=model.predict(x_test[0:10])
# print(y_pred)
#
# y_true=model.transfer(np.argmax(y_test[0:10],axis=2))
# print(y_true)

