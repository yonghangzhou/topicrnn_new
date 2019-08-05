import tensorflow as tf
import os
import numpy as np
import pickle as pkl
import tqdm 
from tqdm import tqdm
from tensorflow import distributions as dist
from tensorflow.python.keras.layers import LSTMCell,Dropout,StackedRNNCells,RNN


def print_top_words(beta, feature_names, n_top_words=10,name_beta=" "):
   # name_beta="beta_"+"iter_"+str(iter_count)+'_k_'+str(network_architecture['n_z'])+ \
   #     '_z_'+str(network_architecture['z_batch_flag'])+'_q_'+str(network_architecture['beta_batch_flag'])+'_c_'+str(network_architecture['phi_batch_flag'])
    beta_list=[]
    print ('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        beta_list.append(" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
        print(" ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print ('---------------End of Topics------------------')    


class vsTopic(object):
  def __init__(self, num_units, dim_emb, vocab_size, num_topics, num_hidden, num_layers, stop_words):
    self.num_units = num_units
    self.dim_emb = dim_emb
    self.num_topics = num_topics
    self.num_hidden = num_hidden
    self.num_layers = num_layers
    self.vocab_size = vocab_size
    self.stop_words = stop_words # vocab size of 01, 1 = stop_words

    with tf.name_scope("beta"):    
      self.beta = 10*tf.get_variable(name="beta", shape=[self.num_topics,self.vocab_size])

    with tf.name_scope("embedding"):    
      self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.dim_emb], dtype=tf.float32)
    

  def forward(self, inputs,params, mode="Train"):
    # build inference network
    stop_indicator=tf.to_float(tf.expand_dims(inputs["indicators"],-1))
    seq_mask=tf.to_float(tf.sequence_mask(inputs["length"]))
    infer_logits = tf.layers.dense(inputs["frequency"], units=self.num_hidden, activation=tf.nn.softplus)
    target_to_onehot=tf.expand_dims(tf.to_float(tf.one_hot(inputs["targets"],self.vocab_size)),2)

    with tf.name_scope("theta"):
        '''KL kl_divergence for theta'''
        alpha = tf.abs(tf.layers.dense(infer_logits, units=self.num_topics,activation=tf.nn.softplus))
        gamma = 10*tf.ones_like(alpha)

        pst_dist = tf.distributions.Dirichlet(alpha)
        pri_dist = tf.distributions.Dirichlet(gamma)

        theta_kl_loss=pst_dist.kl_divergence(pri_dist)
        theta_kl_loss=tf.reduce_mean(theta_kl_loss,-1)
        self.theta=pst_dist.sample()        
    
    with tf.name_scope("RNN_CELL"):
      emb = tf.nn.embedding_lookup(self.embedding, inputs["tokens"])
      cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.num_units), output_keep_prob=inputs["dropout"]) for _ in range(self.num_layers)]
      cell = tf.nn.rnn_cell.MultiRNNCell(cells)
      rnn_outputs, final_output = tf.nn.dynamic_rnn(cell, inputs=emb, sequence_length=inputs["length"], dtype=tf.float32)


    with tf.name_scope("Phi"):      
      emb_wo=(1-stop_indicator)*tf.nn.embedding_lookup(self.embedding,inputs["targets"])
      if params["phi_batch"]==0:
        self.phi=tf.nn.softmax(tf.contrib.layers.batch_norm(tf.layers.dense(emb_wo,self.num_topics),-1))
      elif params["phi_batch"]==1:
        self.phi=tf.nn.softmax(tf.layers.dense(emb_wo,self.num_topics),-1)

        
      self.phi=((1-stop_indicator)*self.phi)+((stop_indicator)*(1/self.num_topics))

    with tf.name_scope("token_loss"):     

      if params["beta_batch"]==1:
        token_logits = tf.expand_dims(tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False),2) + params["lambda"]*tf.expand_dims(1-stop_indicator,-1)*tf.contrib.layers.batch_norm(tf.expand_dims(self.beta,0))        
      elif params["beta_batch"]==0:
        token_logits = tf.expand_dims(tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False),2) + params["lambda"]*tf.expand_dims(1-stop_indicator,-1)*tf.expand_dims(self.beta,0)
      token_logits=tf.nn.softmax(token_logits,-1) 
      token_loss=tf.log(tf.reduce_sum(target_to_onehot*token_logits,-1)+1e-6)
      token_loss=seq_mask*tf.reduce_sum(self.phi*token_loss,-1)
      token_ppl = tf.exp(-tf.reduce_sum(token_loss) / (1e-3 + tf.to_float(tf.reduce_sum(inputs["length"]))))
      token_loss = -tf.reduce_mean(tf.reduce_sum(token_loss, axis=-1))
    
    with tf.name_scope("indicator_loss"):         
      indicator_logits=tf.layers.dense(rnn_outputs,2,activation=tf.nn.softplus)      
      # indicator_logits=tf.layers.dense(indicator_logits,2,activation=tf.nn.softplus)
      labels=1-tf.to_float(tf.one_hot(inputs["indicators"],2))
      indicator_loss=tf.nn.softmax_cross_entropy_with_logits(
              _sentinel=None,
              labels=labels,
              logits=indicator_logits,
              dim=-1,
              name="indicator_loss_softmax",
          )      
      indicator_loss=tf.reduce_mean(tf.reduce_sum(seq_mask*indicator_loss,-1))

    with tf.name_scope("Phi_theta_kl"):
      theta=tf.expand_dims(self.theta,1)
      phi_theta_kl_loss=tf.reduce_mean(tf.reduce_sum(seq_mask*tf.reduce_sum((1-stop_indicator)*self.phi*tf.log((self.phi/(theta+1e-10))+1e-6),-1),-1))


    total_loss=token_loss+theta_kl_loss+indicator_loss+phi_theta_kl_loss


    tf.summary.scalar(tensor=token_loss, name=mode+" token_loss")
    tf.summary.scalar(tensor=phi_theta_kl_loss, name=mode+" phi_theta_kl_loss")    
    tf.summary.scalar(tensor=indicator_loss, name=mode+" indicator_loss")
    tf.summary.scalar(tensor=theta_kl_loss, name=mode+" theta_kl_loss")
    tf.summary.scalar(tensor=total_loss, name=mode+" total_loss")
    tf.summary.scalar(tensor=token_ppl, name=mode+" token_ppl")

    outputs = {
        "token_loss": token_loss,
        "token_ppl": token_ppl,
        "indicator_loss": indicator_loss,
        "theta_kl_loss": theta_kl_loss,
        "loss": total_loss,
        "theta": self.theta,
        "repre": final_output[-1][1],
        "beta":self.beta,
        }
    return outputs


class Train(object):
  def __init__(self, params):
    self.params = params
  
  def _create_placeholder(self):
    self.inputs = {
        "tokens": tf.placeholder(tf.int32, shape=[None, None], name="tokens"),
        "indicators": tf.placeholder(tf.int32, shape=[None, None], name="indicators"),
        "length": tf.placeholder(tf.int32, shape=[None], name="length"),
        "frequency": tf.placeholder(tf.float32, shape=[None, self.params["vocab_size"]], name="frequency"),
        # "frequency": tf.placeholder(tf.float32, shape=[None, None], name="frequency"),
        "targets": tf.placeholder(tf.int32, shape=[None, None], name="targets"),
        "dropout":tf.placeholder(tf.float32,shape=None,name="dropout")
        }

  def build_graph(self):
    self._create_placeholder()
    self.global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
    # with tf.device('/cpu:0'):

    model = vsTopic(num_units = self.params["num_units"],
        dim_emb = self.params["dim_emb"],
        # vocab_size = self.params["vocab_size"],
        vocab_size = self.params["vocab_size"],        
        num_topics = self.params["num_topics"],
        num_layers = self.params["num_layers"],
        num_hidden = self.params["num_hidden"],
        stop_words = self.params["stop_words"],
        )

    # train output
    with tf.variable_scope('topicrnn'):
      self.outputs_train = model.forward(self.inputs,self.params,mode="Train")
      self.outputs_test  = self.outputs_train #same here
      # self.outputs_test  = model.forward(self.inputs,self.params,1.,mode="Train")


    self.summary = tf.summary.merge_all()
    for item in tf.trainable_variables():
      print(item)
    # print('tf.trainable_variables',tf.trainable_variables())
    print('-'*100)
    grads = tf.gradients(self.outputs_train["loss"], tf.trainable_variables())
    grads = [tf.clip_by_value(g, -10.0, 10.0) for g in grads]
    grads, _ = tf.clip_by_global_norm(grads, 20.0)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.params["learning_rate"])
    self.train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()), global_step=self.global_step)
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

  def batch_train(self, sess, inputs):
    keys = list(self.outputs_train.keys())
    outputs = [self.outputs_train[key] for key in keys]
    outputs = sess.run([self.train_op, self.global_step, self.summary] + outputs, feed_dict={self.inputs[k]: inputs[k] for k in self.inputs.keys()})
    ret = {keys[i]: outputs[i+3] for i in range(len(keys))}
    ret["global_step"] = outputs[1]
    ret["summary"] = outputs[2]

    return ret

  def batch_test(self, sess, inputs):
    keys = list(self.outputs_test.keys())
    outputs = [self.outputs_test[key] for key in keys]
    outputs = sess.run(outputs, feed_dict={self.inputs[k]: inputs[k] for k in self.inputs.keys()})
    return {keys[i]: outputs[i] for i in range(len(keys))}

  def run_epoch(self, sess, datasets,train_num_batches,vocab):
    valid_ppl_vec=[]
    train_indic=[]
    train_loss, valid_loss, test_loss = [], [], []
    train_theta, valid_theta, test_theta = [], [], []
    train_repre, valid_repre, test_repre = [], [], []
    # train_label, valid_label, test_label = [], [], []

    dataset_train, dataset_dev, dataset_test = datasets
    # print('dataset_train_len',len(dataset_train))
    pbar=tqdm(range(train_num_batches))
    for _ in pbar:
      batch=next(dataset_train())
      train_outputs = self.batch_train(sess, batch)
      train_loss.append(train_outputs["loss"])
      train_indic.append(train_outputs["indicator_loss"])
      train_theta.append(train_outputs["theta"])
      train_repre.append(train_outputs["repre"])
      beta=train_outputs["beta"]
      theta=train_outputs["theta"]      
      # print('theta_to_beta',theta_to_beta.shape)
      # print('theta',theta.shape)


      pbar.set_description("token_loss: %f, theta_kl_loss: %f, indicator_loss: %f" %(train_outputs["token_loss"],train_outputs["theta_kl_loss"],train_outputs["indicator_loss"]))      
      # train_label.append(batch["label"])
      self.writer.add_summary(train_outputs["summary"], train_outputs["global_step"])
      #print(train_outputs)
    # print_top_words(beta, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0],name_beta="")            
    for batch in dataset_dev():
      valid_outputs = self.batch_test(sess, batch)
      valid_loss.append(valid_outputs["loss"])
      valid_theta.append(valid_outputs["theta"])
      valid_repre.append(valid_outputs["repre"])
      valid_ppl_vec.append(valid_outputs["token_ppl"])
      # valid_label.append(batch["label"])
      #print(valid_outputs)

    for batch in dataset_test():
      test_outputs = self.batch_test(sess, batch)
      test_loss.append(test_outputs["loss"])
      test_theta.append(test_outputs["theta"])
      test_repre.append(test_outputs["repre"])
      # test_label.append(batch["label"])
      #print(test_outputs)

    train_loss = np.mean(train_loss)
    valid_loss = np.mean(valid_loss)
    test_loss = np.mean(test_loss)
    valid_ppl=np.mean(valid_ppl_vec)

    train_theta, valid_theta, test_theta = np.vstack(train_theta), np.vstack(valid_theta), np.vstack(test_theta)
    train_repre, valid_repre, test_repre = np.vstack(train_repre), np.vstack(valid_repre), np.vstack(test_repre)
    # train_label, valid_label, test_label = np.vstack(train_label), np.vstack(valid_label), np.vstack(test_label)

    train_res = [train_loss, train_theta, train_repre]
    valid_res = [valid_loss, valid_theta, valid_repre]
    test_res = [test_loss, test_theta, test_repre]

    print("train_loss: {:.4f}, valid_loss: {:.4f}, valid_ppl: {:.4f}, train_indicator: {:.45}".format(train_loss, valid_loss, valid_ppl,np.mean(train_indic)))

    return train_res, valid_res, test_res,beta

  def run(self, sess, datasets,train_num_batches,vocab):
    best_valid_loss = 1e10
    self.writer = tf.summary.FileWriter(os.path.join(self.params["save_dir"], "train"), sess.graph)
    for i in range(self.params["num_epochs"]):
      train_res, valid_res, test_res,beta = self.run_epoch(sess, datasets,train_num_batches,vocab)
      if i%4==0:
        print_top_words(beta, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0],name_beta="")            
      if best_valid_loss > valid_res[0]:
        # print("Best model found at epoch {}".format(i))
        best_valid_loss = valid_res[0]
        with open(os.path.join(self.params["save_dir"], "results.pkl"), "wb") as f:
          pkl.dump([train_res, valid_res, test_res], f)
        self.saver.save(sess, os.path.join(self.params["save_dir"], "model"), global_step = i)



