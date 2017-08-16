from __future__ import print_function
import tensorflow as tf
from dataset import DataSet
from alexnet import AlexNet
import numpy as np
import cv2

num_classes = 5

# hyperparams
learning_rate = 0.001
dropout_prob = 0.9
train_layers = ['fc8', 'fc7']

# placeholder
image_ph = tf.placeholder(tf.float32, [None, 256, 256, 3], "x_input_ph")
y_ph = tf.placeholder(tf.float32, [None, num_classes], "y_ph")

# model
model = AlexNet(image_ph, dropout_prob, num_classes, train_layers)

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
score = model.fc8


# Train op
with tf.name_scope("train"):
    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y_ph))
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))
    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y_ph, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


tf_pred_label = tf.argmax(model.fc8, 1, name="pred_label")
tf_pred_top5 = tf.nn.top_k(model.fc8, k=5, name="pred_top5")

# summary regitration
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Add gradients to summary
for gradient, var in gradients:
    dd = gradient
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter("TensorBoard/", graph = tf.get_default_graph())

# dataset initialize
dataset = DataSet("./training_data")

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    model.load_initial_params(sess)
    print("Start training...")
    for epoch in range(1000):
        batch = dataset.next_batch(50)
        sess.run(train_op, feed_dict={image_ph:batch[0], y_ph:batch[1]})
        if epoch%20 == 0:
            ms, ddd = sess.run([merged_summary, dd], feed_dict={image_ph:batch[0], y_ph:batch[1]})
            print(dd) 
            print("epoch: "+ str(epoch))
            print(ddd)
            writer.add_summary(ms, epoch)

        print("epoch: "+ str(epoch)) 
   
    # 0
    img = cv2.imread("training_data/n01484850/n01484850_17294.JPEG")
    image_resized = cv2.resize(img, (256, 256))
    image_resized = np.reshape(image_resized, [1, 256, 256, 3])
    pred_label, top5 = sess.run([tf_pred_label, tf_pred_top5], feed_dict={image_ph: image_resized})
    print(pred_label)
    print(top5)
    print("\n")

    # 3
    img = cv2.imread("training_data/n01491361/n01491361_264.JPEG")
    image_resized = cv2.resize(img, (256, 256))
    image_resized = np.reshape(image_resized, [1, 256, 256, 3])
    pred_label, top5 = sess.run([tf_pred_label, tf_pred_top5], feed_dict={image_ph: image_resized})
    print(pred_label)
    print(top5)
    print("\n")

    # 2
    img = cv2.imread("training_data/n01494475/n01494475_2041.JPEG")
    image_resized = cv2.resize(img, (256, 256))
    image_resized = np.reshape(image_resized, [1, 256, 256, 3])
    pred_label, top5 = sess.run([tf_pred_label, tf_pred_top5], feed_dict={image_ph: image_resized})
    print(pred_label)
    print(top5)
    print("\n")
 
    # 4
    img = cv2.imread("training_data/n01496331/n01496331_1233.JPEG")
    image_resized = cv2.resize(img, (256, 256))
    image_resized = np.reshape(image_resized, [1, 256, 256, 3])
    pred_label, top5 = sess.run([tf_pred_label, tf_pred_top5], feed_dict={image_ph: image_resized})
    print(pred_label)
    print(top5)
    print("\n")
      
print(dataset.mapping)
