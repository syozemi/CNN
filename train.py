import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import yaml

import batch
import cnn

# np.random.seed(1919114)
# tf.set_random_seed(1919114)

# yaml形式の設定を読み込む
f = open("settings.yml", encoding='UTF-8')
settings = yaml.load(f)

with open('data/image', 'rb') as f:
    image = pickle.load(f)

with open('data/ncratio10', 'rb') as f:
    ratio = pickle.load(f)

image, ratio = batch.shuffle_image(image, ratio)

print(len(image))
num_data = settings["num_data"] ##訓練用データ数
num_test = settings["num_test"]
train_x = image[:num_data]
test_x = image[num_data:num_data + num_test]
train_t = ratio[:num_data]
test_t = ratio[num_data:num_data + num_test]

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
        summary, loss_val, acc_val = cnn.sess.run([cnn.summary, cnn.loss, cnn.accuracy],
                feed_dict={cnn.x:test_x,
                           cnn.t:test_t,
                           cnn.keep_prob:1.0})
        print ('Step: %d, Loss: %.12f, Accuracy: %f'
               % (i, loss_val, acc_val))
        # cnn.saver.save(cnn.sess, os.path.join(os.getcwd(), 'cnn_session'), global_step=i)
        cnn.writer.add_summary(summary, i)
