import tensorflow as tf
import numpy
from DataSequence_lgbm import DataSequence_lgbm

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

class tmp2():

    def __init__(self, batch_len):
        #コンストラクタ
        self.xs = []
        self.ys = []
        self.cnt = 0
        self.batch_len = batch_len

        for i in range(101):
            tmp_list = (i,i+1)
            self.xs.append(tmp_list)
            self.ys.append(i)
        #print(self.xs)

    def getxy(self):
        start = self.cnt * self.batch_len
        print("start", start)
        end = start + self.batch_len

        batch_len =  self.batch_len

        if end > len(self.xs):
            end = len(self.xs)
            batch_len = len(self.xs) - start

        print("end", end)

        for item1 in self.xs[start : end]:
            print(item1)

        self.cnt += 1

        return self.xs[start : end],self.ys[start : end]

    def getx(self):
        start = self.cnt * self.batch_len
        print("start", start)
        end = start + self.batch_len

        batch_len =  self.batch_len

        if end > len(self.xs):
            end = len(self.xs)
            batch_len = len(self.xs) - start

        print("end", end)

        for item1 in self.xs[start : end]:
            print(item1)

        #self.cnt += 1

        return self.xs[start : end]

    def gety(self):
        start = self.cnt * self.batch_len
        print("start", start)
        end = start + self.batch_len

        batch_len =  self.batch_len

        if end > len(self.xs):
            end = len(self.xs)
            batch_len = len(self.xs) - start

        print("end", end)

        for item1 in self.ys[start : end]:
            print(item1)
        self.cnt += 1

        return self.ys[start : end]

#@tf.function
def input_func_gen():
    print("input Called""")
    shapes = ((2),(1))
    #from_generatorだと1件ずつしか読みこめない
    #よってDataSequenceみたいなクラスを作ってcreate_db1メソッドでバッチ分取ってくるようにした方がおそらく速い
    """
    dataset = tf.data.Dataset.from_generator(lambda :tmpC._generator(),
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=shapes)
    """
    #dataset = tf.data.Dataset.from_tensor_slices(tmpC2.getxy())


    #dataset = dataset.batch(8)
    """
    for x,y in dataset:
        print(x)
        print(y)
    """
    #dataset = dataset.repeat(1)
    #iterator = dataset.make_one_shot_iterator()
    #features_tensors, labels = iterator.get_next()
    #iterator = iter(dataset)
    #print(iterator.get_next())
    #print(dataset)
    #features_tensors, labels = iterator.get_next()
    #print(tmpC2.getx())
    features_tensors = tmpC2.getx()
    labels = tmpC2.gety()

    #print(features_tensors)
    features = {'x': features_tensors}
    print("1 func finished")
    return features, labels



if __name__ == '__main__':
    dsl = DataSequence_lgbm(5)

    #tmpC = tmp()
    tmpC2 = tmp2(8)


    #step_no =dsl.__len__()
    #print("step_no", step_no)

    #for i in range(3):
    #    print(i)
    #    print(input_func_gen())


    x_col = tf.feature_column.numeric_column(key='x', shape=(2))
    es = tf.estimator.LinearRegressor(feature_columns=[x_col])
    es = es.train(input_fn=input_func_gen, steps=4)
    print(input_func_gen())
    #es = es.train(input_fn=dsl.__getitem__, steps=step_no)

