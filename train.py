import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import yaml

import process_data as pro
import batch
import cnn

# np.random.seed(1919114)
# tf.set_random_seed(1919114)

# yaml形式の設定を読み込む
f = open("settings.yml", encoding='UTF-8')
settings = yaml.load(f)

image, ratio = pro.load_cnn_data(1)

print (image.shape, ratio.shape)

print(len(image))
num_train = settings["num_train"] ##訓練用データ数
num_validate = settings["num_validate"]
num_test = settings["num_test"]
train_x = image[:num_train]
val_x = image[num_train:num_train + num_validate]
test_x = image[num_train + num_validate:num_train + num_validate + num_test]
train_t = ratio[:num_train]
val_t = ratio[num_train:num_train + num_validate]
test_t = ratio[num_train + num_validate:num_train + num_validate + num_test]

print (train_x.shape, val_x.shape, test_x.shape)

cnn = cnn.CNN(settings["input_sizex"], settings["input_sizey"], settings["num_class"])

Batch_x = batch.Batch(train_x)
Batch_t = batch.Batch(train_t)
Batch_num = settings["Batch_num"]

i = 0
for _ in range(settings["learning_times"]):
    i += 1
    batch_x = Batch_x.next_batch(Batch_num)
    batch_t = Batch_t.next_batch(Batch_num)
    cnn.sess.run(cnn.train_step,
             feed_dict={cnn.x:batch_x, cnn.t:batch_t, cnn.keep_prob:settings["keep_prob"]})
    if i % 10 == 0:
        summary, loss_val, t, p = cnn.sess.run([cnn.summary, cnn.loss, cnn.t, cnn.p],
                feed_dict={cnn.x:val_x,
                           cnn.t:val_t,
                           cnn.keep_prob:1.0})
        print ('Step: %d, Loss: %.12f'
               % (i, loss_val))
        print (np.array(p[:10]).reshape(10))
        print (np.array(t[:10]).reshape(10))
        cnn.saver.save(cnn.sess, os.path.join(os.getcwd(), 'saver/tmp/cnn_session'), global_step=i) 
        cnn.writer.add_summary(summary, i)

correct = np.zeros(num_test)
prediction = np.zeros(num_test)
loss_val, t, p = cnn.sess.run([cnn.loss, cnn.t, cnn.p], feed_dict={cnn.x:test_x, cnn.t:test_t, cnn.keep_prob:1.0})

print ('Final Loss: %.12f' % (loss_val))
print (pro.validate(t.reshape(num_test) * 100, p.reshape(num_test) * 100))

# cnt = 0
# for i in range(num_test):
#     if t[i] - 0.05 <= p[i] and p[i] <= t[i] + 0.05:
#         cnt += 1
#
# print (p.reshape(num_test))
# print (t.reshape(num_test))
# print (cnt, num_test)
