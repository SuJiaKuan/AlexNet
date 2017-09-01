from __future__ import print_function
import tensorflow as tf
from dataset import DataSet
from alexnet import AlexNet
import numpy as np
import cv2
import time
import os
from tensorflow.python.client import timeline

num_classes = 1000
dropout_prob = 0.9
train_layers = []

MAX_BATCH_SIZE = 128
ITER_NUM = 30
PROCESS_NUM = 1

worker = None
exp_name = 'native'

# dataset initialize
dataset = DataSet("./training_data")

# Fork processes
proc_id = 0
for idx in range(1, PROCESS_NUM):
    new_pid = os.fork()
    if new_pid == 0:
        proc_id = idx
        break

# placeholder
image_ph = tf.placeholder(tf.float32, [None, 256, 256, 3], "x_input_ph")
y_ph = tf.placeholder(tf.float32, [None, num_classes], "y_ph")

# model
model = AlexNet(image_ph, dropout_prob, num_classes, train_layers)

score = model.fc8

# Now, run the network
with tf.Session(worker) as sess:
    tf.global_variables_initializer().run()

    model.load_initial_params(sess)

    batch_size = 1
    while batch_size <= MAX_BATCH_SIZE:
      for i in range(ITER_NUM):
          batch = dataset.next_batch(batch_size)

          print("Start inferring... process={} batch_size={} iter={}".format(proc_id, batch_size, i))
          start_time = time.time()

          result = sess.run([score], feed_dict={image_ph: batch[0]})

          end_time = time.time()
          inference_time = int(round((end_time - start_time) * 1000))
          print("Finish inferring... process={}  batch_size={} iter={} time_consume={}ms".format(proc_id, batch_size, i, inference_time))

          with open("result_{}_{}-{}.txt".format(exp_name, PROCESS_NUM, proc_id), "a") as out:
              if i == 0:
                  out.write("batch_size={}\t".format(batch_size))
              out.write("{}\t".format(inference_time))
              if i == (ITER_NUM - 1):
                  out.write("\n")

      batch_size *= 2
