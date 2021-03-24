
import tensorflow as tf

"""
tensorflow2.xで作成したモデルをtensorflow1.xで使用するため
h5形式で保存して呼出側で後で重みのみロードする
"""


if __name__ == '__main__':
    model_file = "/app/model/bin_op/" + \
        "GBPJPY_CATEGORY_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT9-6_DROP0.0_L-K0_L-R0_DIVIDEMAX10_SPREAD2-90*10"

    save_file = "/app/bin_op/model/" + \
                "GBPJPY_CATEGORY_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT9-6_DROP0.0_L-K0_L-R0_DIVIDEMAX10_SPREAD2-90*10.h5"

    model = tf.keras.models.load_model(model_file)
    model.save(save_file)

    print("END!!")
