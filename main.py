import argparse
import os
import pickle as pkl
import numpy as np
import sys
# import model
import vsTopicModel
import tensorflow as tf
import collections


EOS = "<EOS>"
UNK = "<UNK>"
EOS_ID = 0
UNK_ID = 1

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="vist", help="dataset")
parser.add_argument("--batch_size", type=int, default=200, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--frequency_limit", type=int, default=5, help="limit of repeat for vocabulary")
parser.add_argument("--max_seqlen", type=int, default=100, help="maximum sequence length")
parser.add_argument("--num_units", type=int, default=200, help="num of units")
parser.add_argument("--num_hidden", type=int, default=500, help="hidden units of inference network")
parser.add_argument("--dim_emb", type=int, default=200, help="dimension of embedding")
parser.add_argument("--num_topics", type=int, default=5, help="number of topics")
parser.add_argument("--num_layers", type=int, default=1, help="number of layers")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.7, help="dropout")
parser.add_argument("--lambda", type=float, default=1.0, help="coefficient for beta")

parser.add_argument("--beta_batch", type=int, default=0, help="batch norm for beta ")
parser.add_argument("--phi_batch", type=int, default=0, help="batch norm for phi ")
parser.add_argument("--theta_batch", type=int, default=0, help="batch norm for theta ")
parser.add_argument("--init_from", type=str, default=None, help="init_from")
parser.add_argument("--save_dir", type=str, default="results", help="dir for saving the model")


def load_dataset(params,frequency_limit):
    with open("stop_words.txt", "r") as f:
    	stop_words = [line.strip() for line in f.readlines() if line.strip()]
    	stop_words.append(UNK)
    	stop_words.append(EOS)
    with open("datasets/VIST_max_dataset/train_data_dii_sis.txt", "r") as f:
        words = f.read().replace("\n", "").split()         

    word_counter = collections.Counter(words).most_common()
    vocab = dict()
    vocab_wo_stop = dict()    
    for word, frequency in word_counter:
        if frequency>frequency_limit:
            vocab[word] = len(vocab)
            if word not in stop_words:
                vocab_wo_stop[word]=len(vocab_wo_stop)	


    vocab[EOS] = EOS_ID
    vocab[UNK] = UNK_ID
    vocab_wo_stop=vocab
    params.vocab_size=len(vocab)
    params.vocab_wo_size=len(vocab_wo_stop)
    def get_data(filename, vocab,vocab_size):
    	EOS = "<EOS>"
    	UNK = "<UNK>"
    	EOS_ID = 0
    	UNK_ID = 1      
    	with open(filename, "r") as f:
    		lines = f.readlines()
    		data = list(map(lambda s: s.strip().split(), lines))
    		# data=[[vocab.get(x,UNK_ID) if vocab[x] < vocab_size else UNK_ID for x in line if x in vocab.keys()] for line in data]      
    		data=[[vocab.get(x,UNK_ID) for x in line if x in vocab.keys()] for line in data]      

    		return data
    # vocab = {k: vocab[k] for k in vocab if vocab[k] < params.vocab_size}
    train_x = get_data("datasets/VIST_max_dataset/train_data_dii_sis.txt",vocab,params.vocab_size)
    valid_x = get_data("datasets/VIST_max_dataset/val_data_dii_sis.txt",vocab,params.vocab_size)
    test_x = get_data("datasets/VIST_max_dataset/test_data_dii_sis.txt",vocab,params.vocab_size)
    stop_words_ids = set([vocab[k] for k in stop_words if k in vocab])
    # print (stop_words_ids)
    train = train_x
    valid = valid_x
    test = test_x
    return train, valid, test, vocab, stop_words_ids,vocab_wo_stop



def iterator(data, stop_words_ids, params,vocab_wo_stop,dropout):
  def batchify():
    x = data
    batch_size = params.batch_size
    max_seqlen = params.max_seqlen
    shuffle_idx = np.random.permutation(len(x))
    num_batches_per_epoch=len(x) // batch_size    
    # pbar=tqdm(range(num_batches_per_epoch))

    for i in range(num_batches_per_epoch):
      samples = [x[shuffle_idx[j]] for j in range(i*batch_size, i*batch_size + batch_size)]
      samples = [sample[:max_seqlen - 1] for sample in samples]
      length = [l + 1 for l in list(map(len, samples))]
      width = max(length)
      # width = max_seqlen+1

      # indicators = [[1 if token in stop_words_ids else 0 for token in sample] for sample in samples]

      tokens = [[0] + sample + [0] * (width - 1 - len(sample)) for sample in samples]
      targets = [sample + [0] * (width - len(sample)) for sample in samples]

      indicators = [[1 if token in stop_words_ids else 0 for token in sample] for sample in targets]      
      indicators = [indicator + [1] * (width - len(indicator)) for indicator in indicators]

      feature = np.zeros([batch_size,params.vocab_wo_size], dtype='float32')
      for i in range(batch_size):
        for token in samples[i]:
          if token not in stop_words_ids and token in vocab_wo_stop.values() :
            feature[i, token] += 1.


      # feature = feature / (0.1 + np.sum(feature, axis=1, keepdims=True))
      
      """
      for i in range(params.vocab_size):
        if feature[0, i] != 0:
          print(i, feature[0, i])
      """
      
      output = {"tokens": np.asarray(tokens, dtype='int32'),
          "targets": np.asarray(targets, dtype='int32'),
          "indicators": np.asarray(indicators, dtype='int32'),
          "length": np.asarray(length, dtype='int32'),
          "frequency": feature,
          "dropout":dropout
          }
      """
      for v in output.values():
        print(v.shape)
      """
      yield output
      
  return batchify

def main():
  params = parser.parse_args()
  data_train, data_valid, data_test, vocab, stop_words_ids,vocab_wo_stop = load_dataset(params,frequency_limit=params.frequency_limit)

  train_num_batches=len(data_train) // params.batch_size
  data_train = iterator(data_train, stop_words_ids, params,vocab_wo_stop,params.dropout)
  data_valid = iterator(data_valid, stop_words_ids, params,vocab_wo_stop,1.)
  data_test = iterator(data_test, stop_words_ids, params,vocab_wo_stop,1.)
  params.stop_words = np.asarray([1 if i in stop_words_ids else 0 for i in range(params.vocab_size)])
  
  configproto = tf.ConfigProto()
  configproto.gpu_options.allow_growth = True
  configproto.allow_soft_placement = True
  with tf.Session(config=configproto) as sess:
    train =  vsTopicModel.Train(vars(params))
    train.build_graph()

    if params.init_from:
      train.saver.restore(sess, params.init_from)
      print('Model restored from {0}'.format(params.init_from))
    else:
      tf.global_variables_initializer().run()

    train.run(sess, (data_train, data_valid, data_test),train_num_batches,vocab)

if __name__ == "__main__":
  main()
