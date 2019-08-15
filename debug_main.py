import argparse
import os
import pickle as pkl
import numpy as np
import sys
# import model
# import vsTopicModel
import debug_lda
import tensorflow as tf
import collections

dir_path = os.path.dirname(os.path.realpath(__file__))


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
parser.add_argument("--dim_emb", type=int, default=300, help="dimension of embedding")
parser.add_argument("--num_topics", type=int, default=5, help="number of topics")
parser.add_argument("--num_layers", type=int, default=1, help="number of layers")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.7, help="dropout")
parser.add_argument("--lambda", type=float, default=1.0, help="coefficient for beta")

parser.add_argument("--beta_batch", type=int, default=0, help="batch norm for beta ")
parser.add_argument("--phi_batch", type=int, default=0, help="batch norm for phi ")
parser.add_argument("--theta_batch", type=int, default=0, help="batch norm for theta ")
parser.add_argument("--lstm_norm",type=int,default=0,help="Using LayerNormBasicLSTMCell instead of LSTMCell")
parser.add_argument("--beta_sftmx",type=int,default=0,help="Adding Softmax to Beta matrix; (phi_batch flag should be 0 in this case)")
parser.add_argument("--rnn_lim", type=int, default=0, help="adding coefficient for rnn ")
parser.add_argument("--mixture_lambda",type=float,default=0.5,help="mixture paramater for combining h and beta")



parser.add_argument("--init_from", type=str, default=None, help="init_from")
parser.add_argument("--save_dir", type=str, default="results", help="dir for saving the model")


def load_dataset(params,frequency_limit):
    with open(dir_path+"/stop_words.txt", "r") as f:
    	stop_words = [line.strip() for line in f.readlines() if line.strip()]
    	stop_words.append(UNK)
    	stop_words.append(EOS)
    with open(dir_path+"/datasets/VIST_max_dataset/train_data_dii_sis.txt", "r") as f:
        words = f.read().replace("\n", "").split()         

    word_counter = collections.Counter(words).most_common()
    # vocab = dict()
    # vocab_wo_stop = dict()    
    vocab_list=[]
    for word, frequency in word_counter:
        if frequency>frequency_limit:
            if word not in stop_words:
                vocab_list.insert(0,word)
            else:
                vocab_list.insert(-1,word)


                # vocab_wo_stop[word]=len(vocab_wo_stop)	

    vocab=dict(zip(vocab_list,list(np.arange(len(vocab_list)))))

    vocab[EOS] = len(vocab)
    vocab[UNK] = len(vocab)

    vocab_wo_stop=vocab
    # vocab_wo_stop[EOS] = EOS_ID
    # vocab_wo_stop[UNK] = UNK_ID

    params.vocab_size=len(vocab)
    params.vocab_wo_size=len(vocab_wo_stop)
    def get_data(filename, vocab,vocab_size):
    	# EOS = "<EOS>"
    	# UNK = "<UNK>"
    	# EOS_ID = 0
    	# UNK_ID = 1      
    	with open(filename, "r") as f:
    		lines = f.readlines()
    		data = list(map(lambda s: s.strip().split(), lines))
    		# data=[[vocab.get(x,UNK_ID) if vocab[x] < vocab_size else UNK_ID for x in line if x in vocab.keys()] for line in data]      
    		data=[[vocab.get(x,vocab[UNK]) for x in line if x in vocab.keys()] for line in data]      

    		return data
    # vocab = {k: vocab[k] for k in vocab if vocab[k] < params.vocab_size}
    train_x = get_data(dir_path+"/datasets/VIST_max_dataset/train_data_dii_sis.txt",vocab,params.vocab_size)
    valid_x = get_data(dir_path+"/datasets/VIST_max_dataset/val_data_dii_sis.txt",vocab,params.vocab_size)
    test_x = get_data(dir_path+"/datasets/VIST_max_dataset/test_data_dii_sis.txt",vocab,params.vocab_size)
    stop_words_ids = set([vocab[k] for k in stop_words if k in vocab])
    # print (stop_words_ids)
    train = train_x
    valid = valid_x
    test = test_x
    return train, valid, test, vocab, stop_words_ids,vocab_wo_stop



def iterator(data, stop_words_ids, params,vocab_wo_stop,dropout,vocab,model="train"):
  reverse_vocab_in=dict(zip(vocab.values(),vocab.keys()))
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
      # width = max(length)
      width = max_seqlen

      # indicators = [[1 if token in stop_words_ids else 0 for token in sample] for sample in samples]
      eos_word=[vocab[EOS]]
      # tokens = [[0] + sample + [0] * (width - 1 - len(sample)) for sample in samples]
      # targets = [sample + [0] * (width - len(sample)) for sample in samples]

      tokens = [eos_word + sample + eos_word * (width - 1 - len(sample)) for sample in samples]
      targets = [sample + eos_word * (width - len(sample)) for sample in samples]

      indicators = [[1 if token in stop_words_ids else 0 for token in sample] for sample in targets]      
      indicators = [indicator + [1] * (width - len(indicator)) for indicator in indicators]

      # feature=list(map(lambda x: target.count(x) for target in targets))
      feature=[[target.count(x) for x in target ]for target in targets]
      feature=np.asarray(feature,dtype='int32')*(1-np.asarray(indicators,dtype='int32'))

      # feature = np.zeros([batch_size,params.vocab_wo_size], dtype='float32')
      # for i in range(batch_size):
      #   for token in samples[i]:
      #     if token not in stop_words_ids and token in vocab_wo_stop.values() :
      #     # if token not in stop_words_ids :

      #       feature[i, token] += 1.

      # targets_wo = [[vocab_wo_stop.get(reverse_vocab_in.get(word,UNK_ID),UNK_ID) for word in sample] for sample in targets]

      # for i in range(batch_size):
      #   for token in samples[i]:
      #     if vocab_wo_stop[token] in vocab_wo_stop.values():
      #     # if token not in stop_words_ids :

      #       feature[i, token] += 1.



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
          "frequency": np.asarray(feature,dtype='int32'),
          "dropout":dropout,
          "model":model,
          # "targets_wo":np.asarray(targets_wo,dtype='int32')
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
  print('vocab_len',len(vocab))
  print('vocab_wo_len',len(vocab_wo_stop))
  reverse_vocab=dict(zip(vocab.values(),vocab.keys()))
  print(min(stop_words_ids))
  # for item in stop_words_ids:
  #   print(item,reverse_vocab[item])


  train_num_batches=len(data_train) // params.batch_size
  data_train = iterator(data_train, stop_words_ids, params,vocab_wo_stop,params.dropout,vocab,model="Train")

  # data_train_sample=next(data_train())
  # data_train_targ=data_train_sample["targets"][0]
  # data_train_indic=  data_train_sample["indicators"][0]
  # data_train_len=  data_train_sample["length"][0]
  # data_train_freq=  data_train_sample["frequency"][0]
  # print(data_train_freq)
  # data_train_targ_wo=data_train_sample["targets_wo"][0]
  # print('data_train_targ_wo',data_train_targ_wo)




  # reverse_vocab=dict(zip(vocab.values(),vocab.keys()))
  # reverse_vocab_wo=dict(zip(vocab_wo_stop.values(),vocab_wo_stop.keys()))
  # for item in range(len(data_train_targ)):
  #   print(item,': ',reverse_vocab[data_train_targ[item]],', non-stop:',1-data_train_indic[item],', ',data_train_freq[item])

  # print("-"*50,'\n',data_train_len)

  data_valid = iterator(data_valid, stop_words_ids, params,vocab_wo_stop,1.,vocab,model="Valid")
  data_test = iterator(data_test, stop_words_ids, params,vocab_wo_stop,1.,vocab,model="Test")
  params_str=str(vars(params))

  params.stop_words = np.asarray([1 if i in stop_words_ids else 0 for i in range(params.vocab_size)])
  # print(np.where(params.stop_words==1)[0][0])

  save_file_name='k_'+str(params.num_topics)+'_dout_'+str(params.dropout)+'_mixture_lambda_'+str(params.mixture_lambda)+'_nunt_'+str(params.num_units)

  save_info=[params_str,save_file_name]

  configproto = tf.ConfigProto()
  configproto.gpu_options.allow_growth = True
  configproto.allow_soft_placement = True
  with tf.Session(config=configproto) as sess:
    train =  debug_lda.Train(vars(params))
    train.build_graph()

    if params.init_from:
      train.saver.restore(sess, params.init_from)
      print('Model restored from {0}'.format(params.init_from))
    else:
      tf.global_variables_initializer().run()

    train.run(sess, (data_train, data_valid, data_test),train_num_batches,vocab,save_info)

if __name__ == "__main__":
  main()
