from __future__ import print_function

import os
import glob
import numpy as np
import cv2

class DataSet(object):
    """
    It is inspired by tensorflow dataset example(MNIST)
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
    """

    def __init__(self, train_dir):
        self._train_dir = train_dir
        # dataset preprocessing
        self._train_x, self._train_y, self._num_classes, self.mapping = self._gen_train_file_list()
        self._train_x_shuffled, self._train_y_shuffled = self._train_x, self._train_y
        self._num_examples = len(self._train_x)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def _gen_train_file_list(self):
        train_classes = os.listdir(self._train_dir)
        train_list_x = np.array([])
        train_list_y = np.empty((0, 5), dtype=int)
        num_classes = len(train_classes)
        label_WNID_map = []
        for idx, train_class in enumerate(train_classes):
            regex = self._train_dir+"/"+train_class+"/*.JPEG"
            extracted_train_files = glob.glob(regex)
            label_WNID_map.append({"WNID": train_class, "label": idx})
            for extracted_file in extracted_train_files:
                train_list_x = np.concatenate((train_list_x, [extracted_file]))
                # one-hot encoding
                one_hot_arr = np.zeros([1, num_classes])
                one_hot_arr[0][idx] = 1
                train_list_y = np.append(train_list_y, one_hot_arr, axis=0)
        print (train_list_y.shape)
        return train_list_x, train_list_y, num_classes, label_WNID_map

    def next_batch(self, batch_size, shuffle=True):
        # Shuffle for the first epoch
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._train_x_shuffled = self._train_x[perm0]
            self._train_y_shuffled = self._train_y[perm0]
        # Go to the next epoch

        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            train_x_rest = self._train_x_shuffled[start:self._num_examples]
            train_y_rest = self._train_y_shuffled[start:self._num_examples]
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._train_x_shuffled = self._train_x[perm]
                self._train_y_shuffled = self._train_y[perm]
             # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            train_x_new = self._train_x_shuffled[start:end]
            train_y_new = self._train_y_shuffled[start:end]

            train_x = np.concatenate((train_x_rest, train_x_new), axis=0)
            train_y = np.concatenate((train_y_rest, train_y_new), axis=0)
            
            train_img_x = np.empty([len(train_x), 256, 256, 3])
            for _idx, _file_x in enumerate(train_x):
                train_img_x[_idx] = cv2.resize(cv2.imread(_file_x,), (256, 256))
            return train_img_x, \
                        np.concatenate((train_y_rest, train_y_new), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            train_x = self._train_x_shuffled[start:end]
            train_img_x = np.empty([len(train_x), 256, 256, 3])
            for _idx, _file_x in enumerate(train_x):
                train_img_x[_idx] = cv2.resize(cv2.imread(_file_x,), (256, 256))

            return train_img_x, self._train_y_shuffled[start:end]


