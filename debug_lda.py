import tensorflow as tf
import os
import numpy as np
import pickle as pkl
import tqdm 
from tqdm import tqdm
from tensorflow import distributions as dist
from tensorflow.python.keras.layers import LSTMCell,Dropout,StackedRNNCells,RNN


def print_top_words(beta, feature_names, n_top_words=20,name_beta=" "):
  beta_list=[]
  beta_values=[]
  print ('---------------Printing the Topics------------------')
  for i in range(len(beta)):  
    beta_values.append(" ".join([" ".join([feature_names[j],':',str(beta[i][j]),', ']) for j in beta[i].argsort()[:-n_top_words - 1:-1]]))    
    beta_list.append(" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print(" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
  print ('---------------End of Topics------------------')    
  return(beta_list,beta_values)


class vsTopic(object):
  def __init__(self, num_units, dim_emb, vocab_size, num_topics, num_hidden, num_layers, stop_words,max_seqlen):
    self.num_units = num_units
    self.dim_emb = dim_emb
    self.num_topics = num_topics
    self.num_hidden = num_hidden
    self.num_layers = num_layers
    self.vocab_size = vocab_size
    self.stop_words = stop_words # vocab size of 01, 1 = stop_words
    self.max_seqlen=max_seqlen
    self.non_stop_len=int(np.where(stop_words==1)[0][0])
    print('self.non_stop_len',self.non_stop_len)
    self.theta_weight=tf.get_variable(shape=[self.dim_emb,self.max_seqlen,self.num_topics],name="theta_weight")
    self.paddings=tf.constant([[0,0],[0,self.vocab_size-self.non_stop_len]])

    with tf.name_scope("beta"):    
      # self.beta = tf.get_variable(name="beta", shape=[self.num_topics,self.vocab_size],)
      self.beta = tf.get_variable(name="beta",shape=([self.num_topics,self.non_stop_len]))
      # self.stop_collector=tf.get_variable(name="stop_collector",shape=([1,self.vocab_size]))

      # self.beta=(1-stop_words)*self.beta

    with tf.name_scope("embedding"):    
      self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.dim_emb], dtype=tf.float32)


  def forward(self, inputs,params, mode="Train"):
    # build inference network
    # if params["beta_sftmx"]==1:
    #   self.beta=tf.nn.softmax(self.beta)
    #   params["beta_batch"]=0

    stop_indicator=tf.to_float(tf.expand_dims(inputs["indicators"],-1))
    seq_mask=tf.to_float(tf.sequence_mask(inputs["length"]))
    # infer_logits = tf.layers.dense(inputs["frequency"], units=self.num_hidden, activation=tf.nn.softplus)
    target_to_onehot=tf.expand_dims(tf.to_float(tf.one_hot(inputs["targets"],self.vocab_size)),2)
    print(inputs.keys())

    # with tf.name_scope("theta"):
    #     '''KL kl_divergence for theta'''
    #     if params["theta_batch"]==0:
    #       alpha = tf.abs(tf.layers.dense(infer_logits, units=1,activation=tf.nn.softplus))
    #       # alpha = tf.abs(tf.layers.dense(infer_logits, units=self.num_topics,activation=tf.nn.softplus))

    #     elif params["theta_batch"]==1:
    #       alpha = tf.abs(tf.contrib.layers.batch_norm(tf.layers.dense(infer_logits, units=self.num_topics,activation=tf.nn.softplus)))

    #     gamma = 10*tf.ones_like(alpha)

    #     pst_dist = tf.distributions.Dirichlet(alpha)
    #     pri_dist = tf.distributions.Dirichlet(gamma)

    #     theta_kl_loss=pst_dist.kl_divergence(pri_dist)
    #     theta_kl_loss=tf.reduce_mean(theta_kl_loss,-1)
    #     self.theta=pst_dist.sample()        
    
    with tf.name_scope("RNN_CELL"):
      emb = tf.nn.embedding_lookup(self.embedding, inputs["tokens"])
      final_output=[[0,1]]
      # if params["lstm_norm"]==0:
      #   # cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.num_units), output_keep_prob=inputs["dropout"]) for _ in range(self.num_layers)]
      #   # cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.num_units), output_keep_prob=inputs["dropout"]) for _ in range(self.num_layers)]
      #   cells = [tf.nn.rnn_cell.GRUCell(self.num_units) for _ in range(self.num_layers)]


        # cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        # rnn_outputs, final_output = tf.nn.dynamic_rnn(cell, inputs=emb, sequence_length=inputs["length"], dtype=tf.float32)
      # elif params["lstm_norm"]==1:
      #   cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.num_units, dropout_keep_prob=inputs["dropout"]) for _ in range(self.num_layers)]
      # cell = tf.nn.rnn_cell.MultiRNNCell(cells)
      # rnn_outputs, final_output = tf.nn.dynamic_rnn(cell, inputs=emb, sequence_length=inputs["length"], dtype=tf.float32)

    with tf.name_scope("theta"):
        '''KL kl_divergence for theta'''
        # emb_wo=(1-stop_indicator)*tf.nn.embedding_lookup(self.embedding,inputs["targets"])          
        emb_wo=(1-stop_indicator)*tf.nn.embedding_lookup(self.embedding,inputs["targets"])          

        if params["theta_batch"]==0:
          # alpha = tf.nn.dropout(tf.abs(tf.layers.dense(tf.squeeze(tf.layers.dense(emb_wo, units=1,activation=tf.nn.softplus),-1),units=self.num_topics,activation=tf.nn.softplus)+1e-10),inputs["dropout"])
          # alpha = tf.abs(tf.layers.dense(tf.squeeze(tf.layers.dense(emb_wo, units=1,activation=tf.nn.softplus),-1),units=self.num_topics,activation=tf.nn.softplus))

          # alpha = tf.layers.dense(tf.squeeze(tf.layers.dense(emb_wo, units=1,activation=tf.nn.softplus),-1),units=self.num_topics,activation=tf.nn.softplus)
          # alpha = tf.layers.dense(tf.squeeze(tf.layers.dense(emb_wo, units=1,activation=tf.nn.softplus),-1),units=self.num_topics,activation=tf.nn.softplus)
          alpha = tf.tensordot(emb_wo,self.theta_weight,[[1,2],[0,1]])




          print("alpha",alpha.get_shape())
          # alpha = tf.abs(tf.layers.dense(infer_logits, units=self.num_topics,activation=tf.nn.softplus))

        elif params["theta_batch"]==1:
          alpha = tf.abs(tf.contrib.layers.batch_norm(tf.layers.dense(tf.squeeze(tf.layers.dense(emb, units=1,activation=tf.nn.softplus),-1),units=self.num_topics,activation=tf.nn.softplus)))

          # alpha = tf.abs(tf.contrib.layers.batch_norm(tf.layers.dense(infer_logits, units=self.num_topics,activation=tf.nn.softplus)))

        gamma = tf.ones_like(alpha)

        pst_dist = tf.distributions.Dirichlet(alpha)
        pri_dist = tf.distributions.Dirichlet(gamma)

        theta_kl_loss=pst_dist.kl_divergence(pri_dist)
        theta_kl_loss=tf.reduce_mean(theta_kl_loss,-1)
        self.theta=pst_dist.sample()        


    with tf.name_scope("Phi"):      
      if params["phi_batch"]==0:
        self.phi=tf.nn.softmax(tf.layers.dense(emb_wo,self.num_topics),-1)
      elif params["phi_batch"]==1:
        self.phi=tf.nn.dropout(tf.nn.softmax(tf.contrib.layers.batch_norm(tf.layers.dense(emb_wo,self.num_topics),-1)),inputs["dropout"])

        
      self.phi=((1-stop_indicator)*self.phi)+((stop_indicator)*(1./self.num_topics))

    with tf.name_scope("token_loss"):     

      if params["beta_batch"]==1:        
        # self.beta=(1-*self.stop_words)*self.beta        
        # token_logits = (1-(params["mixture_lambda"]*(1-tf.expand_dims(stop_indicator,-1))))*\
        #               tf.expand_dims(tf.nn.softmax(tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False),-1),2)+\
        #               params["mixture_lambda"]*tf.expand_dims(1-stop_indicator,-1)*tf.expand_dims(tf.nn.softmax(tf.contrib.layers.batch_norm(self.beta),-1),0)                                            
        ''' Debug '''
        # token_logits = (1-(params["mixture_lambda"]*(1-tf.expand_dims(stop_indicator,-1))))*1\
        # +params["mixture_lambda"]*tf.expand_dims(1-stop_indicator,-1)*tf.expand_dims(tf.nn.softmax(tf.contrib.layers.batch_norm(self.beta),-1),0)                                            
        token_logits = (1-(params["mixture_lambda"]*(1-tf.expand_dims(stop_indicator,-1))))*1\
        +params["mixture_lambda"]*tf.expand_dims(1-stop_indicator,-1)*tf.expand_dims(tf.pad(tf.nn.softmax(tf.contrib.layers.batch_norm(self.beta),-1),self.paddings,"CONSTANT"),0)                                            

        # token_logits = (1-(params["mixture_lambda"]*(1-tf.expand_dims(stop_indicator,-1))))*\
        #               tf.expand_dims(tf.nn.softmax(tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False),-1),2)+\
        #               params["mixture_lambda"]*tf.expand_dims(1-stop_indicator,-1)*tf.expand_dims(tf.nn.softmax(tf.contrib.layers.batch_norm(self.beta),-1),0)                                            


                      # params["mixture_lambda"]*tf.expand_dims(1-stop_indicator,-1)*tf.expand_dims(tf.contrib.layers.batch_norm(tf.nn.softmax(self.beta,-1)),0)                      
          # *(tf.expand_dims(stop_indicator,-1))))*tf.expand_dims(tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False),2)) + tf.expand_dims(1-stop_indicator,-1)*tf.contrib.layers.batch_norm(tf.expand_dims(self.beta,0))                                  
        # if params["rnn_lim"]==0:
        # self.stop_collector=self.stop_words*self.stop_collector
        # self.beta=(1-self.stop_words)*self.beta

        # token_logits = tf.expand_dims(stop_indicator,-1)*self.stop_collector+tf.expand_dims(tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False),2) + tf.expand_dims(1-stop_indicator,-1)*tf.contrib.layers.batch_norm(tf.expand_dims(self.beta,0))        
        # token_logits = tf.expand_dims(tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False),2) + (1-self.stop_words)*tf.contrib.layers.batch_norm(tf.expand_dims(self.beta,0))        

        # elif params["rnn_lim"]==1:
        # token_logits = ((params["lambda"]+((1-params["lambda"])*(tf.expand_dims(stop_indicator,-1))))*tf.expand_dims(tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False),2)) + tf.expand_dims(1-stop_indicator,-1)*tf.contrib.layers.batch_norm(tf.expand_dims(self.beta,0))                          
        # token_logits=tf.nn.softmax(token_logits,-1) 
      # elif params["beta_batch"]==0:
      #   token_logits = tf.expand_dims(tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False),2) + params["lambda"]*tf.expand_dims(1-stop_indicator,-1)*tf.expand_dims(self.beta,0)
      token_loss=tf.log(tf.reduce_sum(target_to_onehot*token_logits,-1)+1e-4)
      token_loss=seq_mask*tf.reduce_sum(self.phi*token_loss,-1)
      token_ppl = tf.exp(-tf.reduce_sum(token_loss) / (1e-3 + tf.to_float(tf.reduce_sum(inputs["length"]))))
      token_loss = -tf.reduce_mean(tf.reduce_sum(token_loss, axis=-1))
    
    # with tf.name_scope("indicator_loss"):         
    #   indicator_logits=tf.nn.dropout(tf.layers.dense(rnn_outputs,2,activation=tf.nn.softplus),inputs["dropout"])
    #   labels=1-tf.to_float(tf.one_hot(inputs["indicators"],2))
    #   indicator_loss=tf.nn.softmax_cross_entropy_with_logits(
    #           _sentinel=None,
    #           labels=labels,
    #           logits=indicator_logits,
    #           dim=-1,
    #           name="indicator_loss_softmax",
    #       )      
    #   indicator_loss=tf.reduce_mean(tf.reduce_sum(seq_mask*indicator_loss,-1))

    with tf.name_scope("Phi_theta_kl"):
      theta=tf.expand_dims(self.theta,1)
      # phi_theta_kl_loss=tf.reduce_mean(tf.reduce_sum(seq_mask*tf.reduce_sum((1-stop_indicator)*self.phi*tf.log((self.phi/(theta+1e-10))+1e-6),-1),-1))
      # phi_theta_kl_loss=tf.reduce_mean(tf.reduce_sum(seq_mask*tf.reduce_sum((1-stop_indicator)*self.phi*tf.log(((1-stop_indicator)*self.phi/(theta+1e-10))+1e-6),-1),-1))
      # phi_theta_kl_loss=tf.reduce_mean(tf.reduce_sum(seq_mask*tf.reduce_sum((1-stop_indicator)*self.phi*tf.log((((1-stop_indicator)*self.phi)/(theta+1e-10))+1e-10),-1),-1))
      phi_theta_kl_loss=tf.reduce_mean(tf.reduce_sum(tf.squeeze(1-stop_indicator,-1)*tf.reduce_sum((1-stop_indicator)*self.phi*tf.log((((1-stop_indicator)*self.phi)/(theta+1e-10))+1e-10),-1),-1))



    # total_loss=token_loss+0.1*theta_kl_loss+0*indicator_loss+phi_theta_kl_loss
    total_loss=token_loss+theta_kl_loss+phi_theta_kl_loss

    print(inputs["model"])


    tf.summary.scalar(tensor=token_loss, name=mode+" token_loss")
    tf.summary.scalar(tensor=phi_theta_kl_loss, name=mode+" phi_theta_kl_loss")    
    # tf.summary.scalar(tensor=indicator_loss, name=mode+" indicator_loss")
    tf.summary.scalar(tensor=theta_kl_loss, name=mode+" theta_kl_loss")
    tf.summary.scalar(tensor=total_loss, name=mode+" total_loss")
    tf.summary.scalar(tensor=token_ppl, name=mode+" token_ppl")

    outputs = {
        "token_loss": token_loss,
        "token_ppl": token_ppl,
        "indicator_loss": tf.constant([0]),
        "theta_kl_loss": theta_kl_loss,
        "phi_theta_kl_loss": phi_theta_kl_loss,
        "loss": total_loss,
        "theta": self.theta,
        # "repre": final_output[-1][1],
        "beta":self.beta,
        }
    return outputs


class Train(object):
  def __init__(self, params):
    self.params = params
  
  def _create_placeholder(self):
    self.inputs = {
        "tokens": tf.placeholder(tf.int32, shape=[None, self.params["max_seqlen"]], name="tokens"),
        "indicators": tf.placeholder(tf.int32, shape=[None, self.params["max_seqlen"]], name="indicators"),
        "length": tf.placeholder(tf.int32, shape=[None], name="length"),
        "frequency": tf.placeholder(tf.float32, shape=[None, self.params["vocab_size"]], name="frequency"),
        # "frequency": tf.placeholder(tf.float32, shape=[None, None], name="frequency"),
        "targets": tf.placeholder(tf.int32, shape=[None, self.params["max_seqlen"]], name="targets"),
        "dropout":tf.placeholder(tf.float32,shape=None,name="dropout"),
        "model":" "
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
        max_seqlen = self.params["max_seqlen"],
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
    self.inputs["model"]=inputs["model"]
    outputs = sess.run([self.train_op, self.global_step, self.summary] + outputs, feed_dict={self.inputs[k]: inputs[k] for k in self.inputs.keys() if k!="model"})
    ret = {keys[i]: outputs[i+3] for i in range(len(keys))}
    ret["global_step"] = outputs[1]
    ret["summary"] = outputs[2]

    return ret

  def batch_test(self, sess, inputs):
    keys = list(self.outputs_test.keys())
    outputs = [self.outputs_test[key] for key in keys]
    outputs = sess.run(outputs, feed_dict={self.inputs[k]: inputs[k] for k in self.inputs.keys() if k!="model"})
    return {keys[i]: outputs[i] for i in range(len(keys))}

  def run_epoch(self, sess, datasets,train_num_batches,vocab):
    valid_ppl=[]
    test_ppl=[]

    train_token,train_indic, train_theta_kl,train_phi_theta=[],[],[],[]
    valid_token,valid_indic, valid_theta_kl,valid_phi_theta=[],[],[],[]

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
      train_phi_theta.append(train_outputs["phi_theta_kl_loss"])
      train_theta_kl.append(train_outputs["theta_kl_loss"])      
      train_indic.append(train_outputs["indicator_loss"])
      train_token.append(train_outputs["token_loss"])
      train_theta.append(train_outputs["theta"])
      # train_repre.append(train_outputs["repre"])
      beta=train_outputs["beta"]
      theta=train_outputs["theta"]      
      # print('theta_to_beta',theta_to_beta.shape)
      # print('theta',theta.shape)


      pbar.set_description("token_loss: %f, theta_kl_loss: %f, train_phi_theta: %f" %(train_outputs["token_loss"],train_outputs["theta_kl_loss"],train_outputs["phi_theta_kl_loss"]))      
      # train_label.append(batch["label"])
      self.writer.add_summary(train_outputs["summary"], train_outputs["global_step"])
      #print(train_outputs)
    # print_top_words(beta, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0],name_beta="")            
    for batch in dataset_dev():
      valid_outputs = self.batch_test(sess, batch)
      valid_loss.append(valid_outputs["loss"])      
      valid_theta_kl.append(valid_outputs["theta_kl_loss"])
      valid_phi_theta.append(valid_outputs["phi_theta_kl_loss"])      
      valid_indic.append(valid_outputs["indicator_loss"])
      valid_token.append(valid_outputs["token_loss"])
      valid_theta.append(valid_outputs["theta"])
      # valid_repre.append(valid_outputs["repre"])
      valid_ppl.append(valid_outputs["token_ppl"])
      # self.writer.add_summary(valid_outputs["summary"])

      # valid_label.append(batch["label"])
      #print(valid_outputs)

    for batch in dataset_test():
      test_outputs = self.batch_test(sess, batch)
      test_loss.append(test_outputs["loss"])
      test_theta.append(test_outputs["theta"])
      # test_repre.append(test_outputs["repre"])
      test_ppl.append(test_outputs["token_ppl"])
      # test_label.append(batch["label"])
      #print(test_outputs)

    train_loss = np.mean(train_loss)
    train_token=np.mean(train_token)
    train_indic=np.mean(train_indic)
    train_theta_kl=np.mean(train_theta_kl)
    train_phi_theta=np.mean(train_phi_theta)

    valid_loss = np.mean(valid_loss)
    valid_token=np.mean(valid_token)    
    valid_theta_kl=np.mean(valid_theta_kl)
    valid_phi_theta=np.mean(valid_phi_theta)
    valid_indic=np.mean(valid_indic)
    valid_ppl=np.mean(valid_ppl)

    test_loss = np.mean(test_loss)

    train_theta, valid_theta, test_theta = [],[],[]
    # np.vstack(train_theta), np.vstack(valid_theta), np.vstack(test_theta)
    train_repre, valid_repre, test_repre = [],[],[]
    # np.vstack(train_repre), np.vstack(valid_repre), np.vstack(test_repre)
    # train_theta
    # train_label, valid_label, test_label = np.vstack(train_label), np.vstack(valid_label), np.vstack(test_label)

    # train_res = [train_loss, train_theta, train_repre]
    # valid_res = [valid_loss, valid_theta, valid_repre]
    test_res = [test_loss, test_theta, test_repre]
    # train_res=[train_loss,train_token,train_indic,train_theta_kl,train_phi_theta]
    # valid_res=[valid_loss,valid_token,valid_indic,valid_theta_kl,valid_phi_theta]    
    train_res={"train_loss":train_loss,"train_token":train_token,"train_indic":train_indic,"train_theta_kl":train_theta_kl,"train_phi_theta":train_phi_theta}
    valid_res={"valid_loss":valid_loss,"valid_token":valid_token,"valid_indic":valid_indic,"valid_theta_kl":valid_theta_kl,"valid_phi_theta":valid_phi_theta,"valid_ppl":valid_ppl}

    print('\n')
    print("train_loss: {:.4f}, train_token: {:.4f}, train_indicator: {:.4f}, train_theta_kl: {:.4f}, train_phi_theta: {:.4f}".format(train_loss,train_token,train_indic,train_theta_kl,train_phi_theta))
    print("valid_loss: {:.4f}, valid_token: {:.4f}, valid_indic: {:.4f}    , valid_theta_kl: {:.4f}, valid_phi_theta: {:.4f},  valid_ppl: {:.4f}".format(valid_loss,valid_token,valid_indic,valid_theta_kl,valid_phi_theta,valid_ppl))
    print('\n')

    return train_res, valid_res, test_res,beta

  def run(self, sess, datasets,train_num_batches,vocab,save_info):
    best_valid_loss = 1e10
    self.writer = tf.summary.FileWriter(os.path.join(self.params["save_dir"], "train"), sess.graph)
    train_dict={"train_loss":[],"train_token":[],"train_indic":[],"train_theta_kl":[],"train_phi_theta":[]}
    valid_dict={"valid_loss":[],"valid_token":[],"valid_indic":[],"valid_theta_kl":[],"valid_phi_theta":[],"valid_ppl":[]}
    # valid_loss_all={"valid_loss":[],"valid_token":[],"valid_indic":[],"valid_theta_kl":[],"valid_phi_theta":[]}

    for i in range(self.params["num_epochs"]):
      train_res, valid_res, test_res,beta = self.run_epoch(sess, datasets,train_num_batches,vocab)
      for key in train_dict:
        train_dict[key].append(train_res[key])
      for key in valid_dict:
        valid_dict[key].append(valid_res[key])
      if i%4==0:
        beta_list,beta_values=print_top_words(beta, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0],name_beta="")            
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path+"/"+self.params["save_dir"], save_info[1]+".pkl"), "wb") as f:
      beta_dict={"beta_names":beta_list,"beta_values":beta_values}
      pkl.dump([train_dict, valid_dict,beta_list,save_info[0],beta_dict], f)


      # self.saver.save(sess, os.path.join(self.params["save_dir"], "model"), global_step = i)



