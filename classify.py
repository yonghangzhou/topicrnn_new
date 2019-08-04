import tensorflow as tf
import pickle as pkl
import numpy as np

def load_data(filename):
  with open(filename, "rb") as f:
    data = pkl.load(f)

  train_data, valid_data, test_data = data
  train_data = train_data[1:]
  valid_data = valid_data[1:]
  test_data = test_data[1:]
  
  print(train_data[0].shape)
  dim_theta = train_data[0].shape[1]
  dim_repre = train_data[1].shape[1]

  # fix data format
  if len(train_data[2].shape) == 2:
    train_data[2] = train_data[2].flatten()
    valid_data[2] = valid_data[2].flatten()
    test_data[2] = test_data[2].flatten()
  
  num_labels = len(set(test_data[2].tolist()))
  train_data = [train_data[0][0 <= train_data[2], :], \
      train_data[1][0 <= train_data[2], :], \
      train_data[2][0 <= train_data[2]]]

  return train_data, valid_data, test_data, dim_theta, dim_repre, num_labels

def iterator(dataset, batch_size):
  def batchify():
    shuffle_idx = np.random.permutation(dataset[0].shape[0])
    for i in range(dataset[0].shape[0] // batch_size):
      start, end = i * batch_size, i * batch_size + batch_size
      batch = {"theta": dataset[0][start: end, :],
          "repre": dataset[1][start: end, :],
          "label": dataset[2][start: end],
          }
      output = {"x": np.concatenate([batch["theta"], batch["repre"]], axis=1),
          "y": batch["label"]
          }
      yield output
  return batchify

def main():
  filename = "results/results.pkl"
  train_data, valid_data, test_data, dim_theta, dim_repre, num_labels = load_data(filename)
  x = tf.placeholder(tf.float32, shape=[None, dim_theta + dim_repre], name="x")
  y = tf.placeholder(tf.int64, shape=[None], name="y")

  logits = tf.layers.dense(x, 50, activation=tf.sigmoid)
  logits = tf.layers.dense(logits, 2, activation=tf.nn.softmax)
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y, num_labels))
  loss = tf.reduce_mean(loss)
  pred = tf.argmax(logits, axis=1)
  acc = tf.reduce_mean(tf.to_float(tf.equal(pred, y)))

  train_op = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(loss)
  
  configproto = tf.ConfigProto()
  configproto.gpu_options.allow_growth = True
  configproto.allow_soft_placement = True
  with tf.Session(config=configproto) as sess:
    tf.global_variables_initializer().run()
    train_it = iterator(train_data, 32)
    valid_it = iterator(valid_data, 32)
    test_it = iterator(test_data, 32)

    best_valid_acc = 0
    for i in range(100):
      train_acc, valid_acc, test_acc = [], [], []
      for batch in train_it():
        outs = sess.run([train_op, loss, pred, acc], feed_dict={x: batch["x"], y: batch["y"]})
        train_acc.append(outs[3])
      
      for batch in valid_it():
        outs = sess.run([loss, pred, acc], feed_dict={x: batch["x"], y: batch["y"]})
        valid_acc.append(outs[2])

      for batch in test_it():
        outs = sess.run([loss, pred, acc], feed_dict={x: batch["x"], y: batch["y"]})
        test_acc.append(outs[2])

      train_acc = np.mean(train_acc)
      valid_acc = np.mean(valid_acc)
      test_acc = np.mean(test_acc)

      print("train_acc: {}. valid_acc: {}, test_acc: {}".format(train_acc, valid_acc, test_acc))

      if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        print("Best model found at epoch {}".format(i))

if __name__ == "__main__":
  main()

