import tensorflow as tf
import numpy
from DataSequence_lgbm import DataSequence_lgbm
from DataSequence_lgbm_test import DataSequence_lgbm_test
from readConf2 import *
from datetime import datetime

class tmp():
    def __init__(self):
        self.fearute = numpy.arange(100)
        self.cnt = [0]

    def _generator(self):
        print("Called!!! ")
        while True:
            self.cnt[0] = (self.cnt[0] + 1)
            #print("cnt:", self.cnt)
            feats  = self.fearute[self.cnt[0]:self.cnt[0]+2]
            print(feats)
            labels = numpy.random.rand(1)

            yield feats, labels


@tf.function
def input_func_gen():
    shapes = ((TOTAL_INPUT_LEN),(1))
    #from_generatorだと1件ずつしか読みこめない
    #よってDataSequenceみたいなクラスを作ってcreate_db1メソッドでバッチ分取ってくるようにした方がおそらく速い

    dataset = tf.data.Dataset.from_generator(lambda :dsl.generator(),
                                             ({"x": tf.float32}, tf.float32),
                                             ({"x": (2,)}, (1))
                                        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.repeat(1)
    #iterator = iter(dataset)

    #features_tensors, labels = iterator.get_next()

    #features = {'x': features_tensors}
    #return features, labels
    return dataset

def input_fn_eval():
    shapes = ((TOTAL_INPUT_LEN), (1))
    dataset = tf.data.Dataset.from_generator(lambda: dsl_eval.generator(),
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=shapes)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.repeat(1)
    iterator = iter(dataset)

    features_tensors, labels = iterator.get_next()

    features = {'x': features_tensors}
    return features, labels

def input_fn_predict():
  # Returns tf.data.Dataset of (x, None) tuple.
    pass

if __name__ == '__main__':

    # startからendへ戻ってrec_num分のデータを学習用とする
    rec_num = 1000000 + (INPUT_LEN[0]) + (PRED_TERM) + 1
    #rec_num = 100000 + (INPUT_LEN[0]) + (PRED_TERM) + 1

    #start = datetime(2018, 1, 1)
    start = datetime(2020, 1, 1)

    #end = datetime(2009, 12, 18) #2018,1,1開始で90000000件の場合
    #end = datetime(2007, 1, 1) #2018,1,1開始で1300000000件の場合

    end = datetime(2019, 11, 28) #2020,1,1開始で1000000件の場合
    #end = datetime(2019, 7, 20) #2020,1,1開始で5000000件の場合
    #end = datetime(2017, 10, 6) #2020,1,1開始で25000000件の場合
    #end = datetime(2017, 4, 27) #2020,1,1開始で30000000件の場合
    #end = datetime(2011, 12, 22) #2020,1,1開始で90000000件の場合
    #end = datetime(2010, 3, 15)  #2020,1,1開始で110000000件の場合
    #end = datetime(2009, 4, 12)  #2020,1,1開始で120000000件の場合
    #end = datetime(2007, 7, 17)  #2020,1,1開始で140000000件の場合
    #end = datetime(2007, 1, 1)   #2020,1,1開始で150000000件の場合

    start_eval = datetime(2020, 1, 1, 22)
    end_eval = datetime(2021, 3, 31, 22)

    end_eval = datetime(2020, 1, 31, 22)

    #dsl_eval = DataSequence_lgbm(0, start_eval, end_eval, True, True)

    #dsl = DataSequence_lgbm(rec_num, start, end, False, False)
    dsl = DataSequence_lgbm_test()

    x_col = tf.feature_column.numeric_column(key='x', shape=(TOTAL_INPUT_LEN))


    #see:https://www.tensorflow.org/tutorials/estimator/boosted_trees?hl=ja


    distribution = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    config = tf.estimator.RunConfig(train_distribute=distribution)
    """
    es = tf.estimator.BoostedTreesRegressor(feature_columns=[x_col],
                                            # n_batches_per_layer see:https://stackoverflow.com/questions/55475456/figuring-out-tensorflows-boostedtrees-layer-by-layer-approach/55579136
                                            n_batches_per_layer=dsl.get_steps_per_epoch(),
                                            model_dir=MODEL_DIR,
                                            config=config)
    """
    es = tf.estimator.LinearRegressor(
        feature_columns=[x_col],
        optimizer='SGD',
        config=config)

    es.train(input_fn=input_func_gen, steps=dsl.get_steps_per_epoch())

    print("cnt:", dsl.get_cnt())
    #metrics = es.evaluate(input_fn=input_fn_eval, steps=dsl_eval.get_steps_per_epoch())
    #print(metrics)

    #predictions = es.predict(input_fn=input_fn_predict)