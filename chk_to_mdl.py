import os
from lstm_generator2 import create_model_lstm, create_model_normal
import conf_class

"""
チェックポイントで保存した重みからモデルを改めて保存する
"""


def chk(c, file=None):
    if file == None:
        file = c.FILE_PREFIX

    learning_num = int(c.LEARNING_NUM)
    loading_num = int(c.LOADING_NUM)
    suffixs = []

    if c.LOAD_TYPE == 0:
        for i in range(learning_num):
            if i !=0:
                if i < 10:
                    key = "000" + str(i)
                elif i < 100:
                    key = "00" + str(i)
                else:
                    key = "0" + str(i)
                suffixs.append([key, str(i)])
    elif c.LOAD_TYPE == 1:
        for i in range(learning_num - loading_num):
            if i !=0:
                if i < 10:
                    key = "000" + str(i)
                elif i < 100:
                    key = "00" + str(i)
                else:
                    key = "0" + str(i)
                suffixs.append([key, str(loading_num + i)])

    for sf in suffixs:
        print(sf)
    #exit(1)
    #suffixs = [     ["0009", "130*64"],]

    for suffix in suffixs:
        chk_path = os.path.join(c.CHK_DIR, suffix[0])
        save_path = "/app/model/bin_op/" + file + "-" + suffix[1]
        print("chk_path",chk_path)

        if os.path.isdir(save_path):
            print("ERROR!! SAVE_DIR Already Exists ", save_path)
            exit(1)

        model = None

        if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or \
                c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or c.METHOD == "TCN" or c.METHOD == "TCN7":

            model = create_model_lstm(c)
        elif c.METHOD == "NORMAL":
            model = create_model_normal(c)

        model.load_weights(chk_path)

        # SavedModel形式で保存
        model.save(save_path)

    print("END!!")

if __name__ == '__main__':
    conf = conf_class.ConfClass()
    conf.FILE_PREFIX = "MN886"
    conf.CHK_DIR = "/app/chk/bin_op/" + conf.FILE_PREFIX + "-" + conf.LEARNING_NUM
    chk(conf)