import tensorflow as tf
import numpy


class tmp():
    def __init__(self):
        self.fearute = numpy.arange(100)
        self.cnt = [0]

    def _generator(self):
        print("Called!!! ")
        while True:
            self.cnt[0] = (self.cnt[0] + 1)
            #print("cnt:", cnt)
            feats  = self.fearute[self.cnt[0]:self.cnt[0]+2]
            labels = numpy.random.rand(1)

            yield feats, labels


@tf.function
def input_func_gen():
    print("input Called""")
    shapes = ((2),(1))
    #from_generatorだと1件ずつしか読みこめない
    #よってDataSequenceみたなクラスを作ってcreate_db1メソッドでバッチ分取ってくるようにした方がおそらく速い
    dataset = tf.data.Dataset.from_generator(lambda :tmpC._generator(),
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=shapes)
    dataset = dataset.batch(8)
    dataset = dataset.repeat(1)
    #iterator = dataset.make_one_shot_iterator()
    #features_tensors, labels = iterator.get_next()
    iterator = iter(dataset)
    #print(iterator.get_next())
    #print(dataset)
    features_tensors, labels = iterator.get_next()
    #features_tensors, labels = dataset
    print(features_tensors)
    features = {'x': features_tensors}
    print("1 func finished")
    return features, labels


if __name__ == '__main__':
    tmpC = tmp()

    x_col = tf.feature_column.numeric_column(key='x', shape=(2))
    es = tf.estimator.LinearRegressor(feature_columns=[x_col])
    es = es.train(input_fn=input_func_gen, steps=4)

