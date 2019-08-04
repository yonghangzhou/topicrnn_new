import tensorflow as tf
import os
import numpy as np
import pickle as pkl
import tqdm 
from tqdm import tqdm

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

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

class TopicRNN(object):
  def __init__(self, num_units, dim_emb, vocab_size, num_topics, num_hidden, num_layers, stop_words,max_seqlen):
    self.num_units = num_units
    self.dim_emb = dim_emb
    self.num_topics = num_topics
    self.num_hidden = num_hidden
    self.num_layers = num_layers
    self.vocab_size = vocab_size
    self.stop_words = stop_words # vocab size of 01, 1 = stop_words
    self.beta = tf.get_variable(name="beta", shape=[self.num_topics,self.vocab_size])

    # beta_init_embeddings = tf.random_uniform([self.num_topics,self.vocab_size])
    # self.beta = tf.get_variable(name="beta", initial_value=tf.random_normal([self.num_topics,self.vocab_size]),trainable=True)

  def forward(self, inputs, mode="Train"):
    # build inference network
    self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.dim_emb], dtype=tf.float32)
    # emb_theta=tf.to_float(tf.expand_dims(1-inputs["indicators"],-1))*tf.nn.embedding_lookup(self.embedding, inputs["targets"])

    # infer_mean=tf.squeeze(tf.layers.dense(emb_theta,units=1,activation=tf.nn.softmax),-1)
    # infer_mean=tf.layers.dense(infer_mean,units=self.num_topics,activation=tf.nn.softmax)
    # # print('infer_mean',infer_mean.get_shape())

    # infer_logvar=tf.squeeze(tf.layers.dense(emb_theta,units=1,activation=tf.nn.softmax),-1)
    # infer_logvar=tf.layers.dense(infer_mean,units=self.num_topics,activation=tf.nn.softmax)

    infer_logits = tf.layers.dense(inputs["frequency"], units=self.num_hidden, activation=tf.nn.relu)
    infer_logits = tf.layers.dense(infer_logits, units=self.num_hidden, activation=tf.nn.relu)

    infer_mean = tf.layers.dense(infer_logits, units=self.num_topics,activation=tf.nn.relu)
    infer_logvar = tf.layers.dense(infer_logits, units=self.num_topics,activation=tf.nn.relu)
    
    pst_dist = tf.distributions.Normal(loc=infer_mean, scale=tf.exp(infer_logvar))
    pri_dist = tf.distributions.Normal(loc=tf.zeros_like(infer_mean), scale=tf.ones_like(infer_logvar))
    theta = pst_dist.sample()
    # self.theta=theta
    print('-'*50)
    print('theta',theta.get_shape())
    print('stop_words',self.stop_words.shape)    

    # build generative model
    emb = tf.nn.embedding_lookup(self.embedding, inputs["tokens"])
    cells = [tf.nn.rnn_cell.LSTMCell(self.num_units) for _ in range(self.num_layers)]
    # cells = [tf.nn.rnn_cell.BasicRNNCell(self.num_units) for _ in range(self.num_layers)]

    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    rnn_outputs, final_output = tf.nn.dynamic_rnn(cell, inputs=emb, sequence_length=inputs["length"], dtype=tf.float32)


    # theta_to_beta=tf.layers.dense(tf.expand_dims(theta, 1), units=self.vocab_size, use_bias=False)
    theta_to_beta=tf.expand_dims( tf.matmul(theta,self.beta),1)
    # beta_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(theta_to_beta.name)[0] + '/kernel:0')

    # print('weights',beta_weights.get_shape())
    '''Original'''
    # token_logits = tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False) + theta_to_beta* tf.to_float(1 - self.stop_words)
    # token_logits = tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False) + theta_to_beta* tf.to_float(1 - self.stop_words)
    # token_logits = tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False) + theta_to_beta* tf.to_float(1 - self.stop_words)


    '''Edited '''
    token_logits = tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False) + theta_to_beta* tf.expand_dims(tf.to_float(1 - inputs["indicators"]),-1)
    token_loss =-tf.to_float(tf.sequence_mask(inputs["length"]))*tf.reduce_sum(tf.to_float(tf.one_hot(inputs["targets"],self.vocab_size))*tf.log(tf.nn.softmax(token_logits,axis=-1)+1e-10),-1)
    # token_loss =tf.reduce_sum(tf.to_float(tf.one_hot(inputs["targets"],self.vocab_size))*tf.log(tf.nn.softmax(token_logits,axis=-1)+1e-10),-1)

    # token_loss=tf.reduce_mean(tf.to_float(tf.sequence_mask(inputs["length"]))*token_prob)


    # token_logits = tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False) + \
    #     tf.matmul(theta,self.beta) * \
    #     tf.to_float(1 - self.stop_words)

    print('token_logits',token_logits.shape)                
    # print('my_token_logits',my_token_logits.shape)        
    print('-'*50)

    # token_loss = tf.contrib.seq2seq.sequence_loss(logits=token_logits,
    #     targets=inputs["targets"],
    #     weights=tf.to_float(tf.sequence_mask(inputs["length"])),
    #     average_across_timesteps=False,
    #     average_across_batch=False,
    #     name="token_loss")
    # token_loss = token_loss * tf.to_float(tf.sequence_mask(inputs["length"]))
    token_ppl = tf.exp(tf.reduce_sum(token_loss) / (1e-3 + tf.to_float(tf.reduce_sum(inputs["length"]))))
    token_loss = tf.reduce_mean(tf.reduce_sum(token_loss, axis=-1))
    
    # indicator_logits = tf.squeeze(tf.layers.dense(tf.nn.softmax(rnn_outputs,-1),  units=1), axis=2)
    # indicator_logits = tf.layers.dense(rnn_outputs,2,activation=tf.nn.relu)
    # indicator_logits = tf.nn.softmax(tf.layers.dense(rnn_outputs,2,activation=tf.nn.softplus),axis=-1)
    # indicator_logits = tf.nn.softmax(tf.layers.dense(rnn_outputs,2,activation=tf.nn.softplus),axis=-1)
    indicator_logits = tf.layers.dense(rnn_outputs,2,activation=tf.nn.relu)


    # true_labels=1-tf.to_float(tf.one_hot(inputs['indicators'],2))
    # indicator_loss=tf.reduce_sum(true_labels*tf.log(indicator_logits+1e-6),-1)
    # indicator_loss=tf.reduce_sum(tf.to_float(tf.sequence_mask(inputs["length"]))*indicator_loss,-1)
    # indicator_loss=-tf.reduce_mean(indicator_loss)
    labels=1-tf.to_float(tf.one_hot(inputs["indicators"],2))
    indicator_loss=tf.nn.softmax_cross_entropy_with_logits(
            _sentinel=None,
            labels=labels,
            logits=indicator_logits,
            dim=-1,
            name="indicator_loss_softmax",
        )
    # indicator_logits = tf.sigmoid(indicator_logits)
    # indicator_loss=tf.to_float(tf.expand_dims(inputs['indicators'],-1))*tf.log(indicator_logits+1e-6)+(1-tf.expand_dims(tf.to_float(inputs['indicators']),-1))*tf.log(1-indicator_logits+1e-6)
    # indicator_loss=tf.reduce_sum(indicator_loss,-1)
    indicator_loss=tf.reduce_mean(tf.reduce_sum(tf.to_float(tf.sequence_mask(inputs["length"]))*indicator_loss,-1))


    # indicator_loss=-tf.reduce_mean(tf.reduce_sum(tf.to_float(tf.sequence_mask(inputs["length"],self.max_seqlen))*indicator_loss,-1))



    # indicator_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(inputs["indicators"]),
    #     logits=indicator_logits,
    #     name="indicator_loss")
    # indicator_loss = indicator_loss * tf.to_float(tf.sequence_mask(inputs["length"]))
    # indicator_loss = tf.reduce_mean(tf.reduce_sum(indicator_loss, axis=1))


    # indicator_loss = indicator_loss * tf.to_float(tf.sequence_mask(inputs["length"]))
    # indicator_loss = tf.reduce_mean(tf.reduce_sum(indicator_loss, axis=1))

    kl_loss = tf.contrib.distributions.kl_divergence(pst_dist, pri_dist)
    # print('kl_loss',kl_loss.get_shape())
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1), axis=0)

    total_loss = token_loss + indicator_loss + 1 * kl_loss
    # total_loss = token_prob + indicator_loss + 1 * kl_loss


    tf.summary.scalar(tensor=token_loss, name="token_loss")
    tf.summary.scalar(tensor=indicator_loss, name="indicator_loss")
    # tf.summary.scalar(tensor=kl_loss, name="kl_loss")
    tf.summary.scalar(tensor=total_loss, name="total_loss")
    tf.summary.scalar(tensor=token_ppl, name="ppl")

    outputs = {
        "token_loss": token_loss,
        "token_ppl": token_ppl,
        "indicator_loss": indicator_loss,
        "kl_loss": kl_loss,
        "loss": total_loss,
        "theta": theta,
        "repre": final_output[-1][1],
        "beta":self.beta,
        "token_ppl":token_ppl,
        }


    return outputs

class Train(object):
  def __init__(self, params):
    self.params = params
  
  def _create_placeholder(self):
    self.inputs = {
        # "tokens": tf.place`holder(tf.int32, shape=[None, None], name="tokens"),
        "tokens": tf.placeholder(tf.int32, shape=[None, None], name="tokens"),

        # "indicators": tf.placeholder(tf.int32, shape=[None, None], name="indicators"),
        "indicators": tf.placeholder(tf.int32, shape=[None, None], name="indicators"),

        "length": tf.placeholder(tf.int32, shape=[None], name="length"),
        # "frequency": tf.placeholder(tf.float32, shape=[None, self.params["vocab_size"]], name="frequency"),
        "frequency": tf.placeholder(tf.float32, shape=[None, self.params["vocab_wo_size"]], name="frequency"),
        "targets": tf.placeholder(tf.int32, shape=[None, None], name="targets"),
        }

  def build_graph(self):
    self._create_placeholder()
    self.global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
    # with tf.device('/cpu:0'):

    model = TopicRNN(num_units = self.params["num_units"],
        dim_emb = self.params["dim_emb"],
        # vocab_size = self.params["vocab_size"],
        vocab_size = self.params["vocab_size"],        
        num_topics = self.params["num_topics"],
        num_layers = self.params["num_layers"],
        num_hidden = self.params["num_hidden"],
        stop_words = self.params["stop_words"],
        max_seqlen = self.params["max_seqlen"],
        )

    # train output
    with tf.variable_scope('topicrnn'):
      self.outputs_train = model.forward(self.inputs, mode="Train")
      self.outputs_test  = self.outputs_train #same here

    self.summary = tf.summary.merge_all()
    print('-'*100)
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


      pbar.set_description("token_loss: %f, kl_loss: %f, indicator_loss: %f" %(train_outputs["token_loss"],train_outputs["kl_loss"],train_outputs["indicator_loss"]))      
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
