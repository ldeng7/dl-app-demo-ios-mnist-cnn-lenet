### train project

[dl-tensorflow-study: mnist_cnn_lenet.py](https://github.com/ldeng7/dl-tensorflow-study/blob/master/train/model/mnist_cnn_lenet.py)

### script of fetching model variables

```python
# path = r"/path/to/dl-tensorflow-study/train/record/mnist-cnn-lenet/"
# out_path = r"/path/to/cnn_matrix.dat"

def fetch_matrix(path, out_path):
    import tensorflow as tf
    cp = tf.train.get_checkpoint_state(path)
    saver = tf.train.import_meta_graph(cp.model_checkpoint_path + ".meta")
    with tf.Session() as sess:
        saver.restore(sess, cp.model_checkpoint_path)
        f1, bf1, f2, bf2, w1, b1, w2, b2 = sess.run(tf.global_variables()[1:9])
    f1 = f1.transpose(3,2,0,1).flatten()
    f2 = f2.transpose(3,2,0,1).flatten()
    w1 = w1.reshape([7,7,64,512]).transpose([3,2,0,1]).flatten()
    w2 = w2.transpose().flatten()
  
    f = open(out_path, "wb")
    f.write(f1.data)
    f.write(bf1.data)
    f.write(f2.data)
    f.write(bf2.data)
    f.write(w1.data)
    f.write(b1.data)
    f.write(w2.data)
    f.write(b2.data)
    f.flush()
    f.close()
    # file size should be 6,653,480 bytes
```

### Screenshots on simulator

![0](http://ouaoc2fl1.bkt.clouddn.com/20180214000.png)

![1](http://ouaoc2fl1.bkt.clouddn.com/20180214001.png)

![2](http://ouaoc2fl1.bkt.clouddn.com/20180214002.png)

![3](http://ouaoc2fl1.bkt.clouddn.com/20180214003.png)

![4](http://ouaoc2fl1.bkt.clouddn.com/20180214004.png)

![5](http://ouaoc2fl1.bkt.clouddn.com/20180214005.png)

![6](http://ouaoc2fl1.bkt.clouddn.com/20180214006.png)

![7](http://ouaoc2fl1.bkt.clouddn.com/20180214007.png)

![8](http://ouaoc2fl1.bkt.clouddn.com/20180214008.png)

![9](http://ouaoc2fl1.bkt.clouddn.com/20180214009.png)
