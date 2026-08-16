import csv
import numpy as np
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):

    def __init__(self, file_path, model_number, learning_type="CATEGORY"):
        super().__init__()

        self.model_number = int(model_number) -1
        self.file_path = file_path
        self.learning_type = learning_type

        self.loss_list = []
        self.accuracy_list = []

    def on_test_begin(self, logs=None):
        # open csv file
        self.loss_list = []
        self.accuracy_list = []

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):

        if self.learning_type == "CATEGORY":
            self.loss_list.append(logs["loss"])
            self.accuracy_list.append(logs["accuracy"])
        else:
            self.loss_list.append(logs["loss"])

        #print((batch, logs))

    def on_test_end(self, logs=None):
        avg_loss = np.average(np.array(self.loss_list))
        avg_acc = np.average(np.array(self.accuracy_list))

        with open(self.file_path, 'a') as f:
            writer = csv.writer(f)
            if self.learning_type == "CATEGORY":
                writer.writerow([self.model_number, avg_acc, avg_loss])
            else:
                writer.writerow([self.model_number, avg_loss])


