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
    print(i,": "," ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
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
    self.theta_weight=tf.get_variable(shape=[self.dim_emb,self.max_seqlen,self.num_topics],name="theta_weight")
    self.paddings=tf.constant([[0,0],[0,self.vocab_size-self.non_stop_len]])

    with tf.name_scope("beta"):    
      self.beta = tf.get_variable(name="beta",shape=([self.num_topics,self.non_stop_len]))

    with tf.name_scope("embedding"):    
      self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.dim_emb], dtype=tf.float32)


  def forward(self, inputs,params, mode="Train"):

    stop_indicator=tf.to_float(tf.expand_dims(inputs["indicators"],-1))
    seq_mask=tf.to_float(tf.sequence_mask(inputs["length"],self.max_seqlen))
    target_to_onehot=tf.expand_dims(tf.to_float(tf.one_hot(inputs["targets"],self.vocab_size)),2)

    '''RNN Cell'''
    with tf.name_scope("RNN_CELL"):
      emb = tf.nn.embedding_lookup(self.embedding, inputs["tokens"])    
      cells = [tf.nn.rnn_cell.GRUCell(self.num_units) for _ in range(self.num_layers)]
      cell = tf.nn.rnn_cell.MultiRNNCell(cells)
      rnn_outputs, final_output = tf.nn.dynamic_rnn(cell, inputs=emb, sequence_length=inputs["length"], dtype=tf.float32)

    ''' Sampling theta q(theta|w;alpha)'''
    with tf.name_scope("theta"):
        emb_wo=tf.expand_dims(inputs["frequency"],-1)*tf.nn.embedding_lookup(self.embedding,inputs["targets"])          
        alpha = tf.nn.softplus(tf.tensordot(emb_wo,self.theta_weight,[[1,2],[0,1]]))
        gamma =params["prior"]*tf.ones_like(alpha)
        pst_dist = tf.distributions.Dirichlet(alpha)
        pri_dist = tf.distributions.Dirichlet(gamma)
        '''kl_divergence for theta'''
        theta_kl_loss=pst_dist.kl_divergence(pri_dist)
        theta_kl_loss=tf.reduce_mean(theta_kl_loss,-1)
        self.theta=pst_dist.sample()        


    ''' Phi Matrix '''   
    with tf.name_scope("Phi"):   
      # self.phi=tf.nn.dropout(tf.nn.softmax(tf.contrib.layers.batch_norm(tf.layers.dense(emb_wo,self.num_topics),-1)),inputs["dropout"])
      self.phi=tf.nn.softmax(tf.contrib.layers.batch_norm(tf.layers.dense(emb_wo,self.num_topics),-1))
      self.phi=((1-stop_indicator)*self.phi)+((stop_indicator)*(1./self.num_topics))

        
    '''Token loss (Reconstruction Loss)'''
    with tf.name_scope("token_loss"):     
      self.h_to_vocab=tf.layers.dense(rnn_outputs, units=self.vocab_size, use_bias=False)
      self.b_to_vocab=tf.pad(tf.contrib.layers.batch_norm(self.beta),self.paddings,"CONSTANT")
      self.token_all_probs=tf.nn.softmax(tf.expand_dims(self.h_to_vocab,2)+((1-tf.expand_dims(stop_indicator,-1))*self.b_to_vocab),-1)
      self.token_ppx_non_prob=tf.nn.softmax(tf.expand_dims(self.h_to_vocab,2)+self.b_to_vocab,-1)

      target_prob=tf.reduce_sum(target_to_onehot*self.token_all_probs,-1)
      target_non_prob=tf.reduce_sum(target_to_onehot*self.token_ppx_non_prob,-1)

      token_loss=tf.log(target_prob+1e-4)
      token_loss=seq_mask*tf.reduce_sum(self.phi*token_loss,-1)
      token_loss = -tf.reduce_mean(tf.reduce_sum(token_loss, axis=-1))


    with tf.name_scope("indicator_loss"):         
      self.indicator_logits = tf.squeeze(tf.contrib.layers.batch_norm(tf.layers.dense(tf.layers.dense(rnn_outputs,  units=5,activation=tf.nn.softplus),units=1,activation=tf.nn.softplus)), axis=2)
      indicator_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(inputs["indicators"]),logits=self.indicator_logits,name="indicator_loss")
      indicator_loss=tf.reduce_mean(tf.reduce_sum(seq_mask*indicator_loss,-1))
      indicator_acc=tf.reduce_mean(tf.to_float(tf.equal(tf.round(tf.nn.sigmoid(self.indicator_logits)),tf.to_float(inputs["indicators"]))),-1)
      indicator_acc=tf.reduce_mean(indicator_acc)





    with tf.name_scope("Perplexity"):
        self.stop_prob=tf.nn.sigmoid(self.indicator_logits)        
        self.h_part=tf.expand_dims(self.stop_prob,-1)*tf.nn.softmax(self.h_to_vocab,-1)
        self.theta_part=tf.reduce_sum(tf.expand_dims(tf.expand_dims(1-self.stop_prob,-1)*tf.expand_dims(self.theta,1),-1)*self.token_ppx_non_prob,2)
        token_ppl_log=tf.log(tf.reduce_sum(tf.squeeze(target_to_onehot,2)*(self.h_part+self.theta_part),-1)+1e-10)
        token_ppl=tf.exp(-tf.reduce_sum(seq_mask*token_ppl_log)/(1e-10+tf.to_float(tf.reduce_sum(inputs["length"]))))



    # with tf.name_scope("TextGenerate"):
    #   self.k_text=tf.expand_dims(tf.nn.sigmoid(self.indicator_logits),-1)*tf.nn.softmax(self.h_to_vocab,-1)

    ''' KL between Phi and theta '''
    with tf.name_scope("Phi_theta_kl"):
      theta=tf.expand_dims(self.theta,1)
      phi_theta_kl_loss=tf.reduce_mean(tf.reduce_sum(tf.squeeze(1-stop_indicator,-1)*tf.reduce_sum((1-stop_indicator)*self.phi*tf.log((((1-stop_indicator)*self.phi)/(theta+1e-10))+1e-10),-1),-1))      

    total_loss=token_loss+theta_kl_loss+indicator_loss+phi_theta_kl_loss

    with tf.name_scope("SwitchP"):
      all_topics=tf.argmax(self.phi,-1)
      # cat_topic=dist.Categorical(probs=self.phi)
      # cat_topic=dist.Categorical(probs=self.theta)
      # all_topics=tf.transpose(cat_topic.sample(sample_shape=[self.phi.get_shape()[1]]))
      print('-'*100)
      print('all_topics',all_topics.get_shape())
      print('-'*100)

      # all_topics=tf.self.phi
    with tf.name_scope("Entropies"):
      # phi_entropy=tf.reduce_mean(tf.reduce_sum(tf.to_float(1-inputs["indicators"])*tf.reduce_sum(-self.phi*tf.log(self.phi+1e-10),-1),-1)/tf.reduce_sum(tf.to_float(1-inputs["indicators"])),-1)      
      phi_entropy=tf.reduce_sum(-(1-stop_indicator)*self.phi*tf.log(self.phi+1e-10))/tf.reduce_sum(tf.to_float(1-inputs["indicators"]))
      theta_entropy=tf.reduce_mean(tf.reduce_sum(-self.theta*tf.log(self.theta+1e-10),-1))      


    # tf.summary.scalar(tensor=token_loss, name=mode+" token_loss")
    # tf.summary.scalar(tensor=phi_theta_kl_loss, name=mode+" phi_theta_kl_loss")    
    # tf.summary.scalar(tensor=indicator_loss, name=mode+" indicator_loss")
    # tf.summary.scalar(tensor=theta_kl_loss, name=mode+" theta_kl_loss")
    # tf.summary.scalar(tensor=total_loss, name=mode+" total_loss")
    # tf.summary.scalar(tensor=token_ppl, name=mode+" token_ppl")

    outputs = {
        "token_loss": token_loss,
        "token_ppl": token_ppl,
        "indicator_loss": indicator_loss,
        "theta_kl_loss": theta_kl_loss,
        "phi_theta_kl_loss": phi_theta_kl_loss,
        "loss": total_loss,
        "theta": self.theta,
        "repre": final_output[-1][1],
        "beta":self.beta,
        "all_topics": all_topics,
        "non_stop_indic":1-inputs["indicators"],
        "phi":self.phi,
        "accuracy":indicator_acc,
        "theta_entropy":theta_entropy,
        "phi_entropy":phi_entropy
        }
    return outputs

  def textGenerate(self):
      # self.theta_part=tf.reduce_sum(tf.expand_dims(tf.expand_dims(1-self.stop_prob,-1)*tf.expand_dims(theta_gen,1),-1)*self.token_ppx_non_prob,2)      
      pred_next_token_theta=dist.Categorical(probs=self.h_part+self.theta_part).sample()
      return pred_next_token_theta





class Train(object):
  def __init__(self, params):
    self.params = params
  
  def _create_placeholder(self):
    self.inputs = {
        "tokens": tf.placeholder(tf.int32, shape=[None, self.params["max_seqlen"]], name="tokens"),
        "indicators": tf.placeholder(tf.int32, shape=[None, self.params["max_seqlen"]], name="indicators"),
        "length": tf.placeholder(tf.int32, shape=[None], name="length"),
        "frequency": tf.placeholder(tf.float32, shape=[None, self.params["max_seqlen"]], name="frequency"),
        "targets": tf.placeholder(tf.int32, shape=[None, self.params["max_seqlen"]], name="targets"),
        "dropout":tf.placeholder(tf.float32,shape=None,name="dropout"),
        "model":" "    
        }
    # self.theta_gen=tf.placeholder(tf.float32, shape=[None, self.params["num_topics"]], name="theta_gen")    

  def build_graph(self):
    self._create_placeholder()
    self.global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
    # with tf.device('/cpu:0'):

    self.model = vsTopic(num_units = self.params["num_units"],
        dim_emb = self.params["dim_emb"],
        vocab_size = self.params["vocab_size"],        
        num_topics = self.params["num_topics"],
        num_layers = self.params["num_layers"],
        num_hidden = self.params["num_hidden"],
        stop_words = self.params["stop_words"],
        max_seqlen = self.params["max_seqlen"],
        )

    # train output
    with tf.variable_scope('VSTM'):
      self.outputs_train = self.model.forward(self.inputs,self.params,mode="Train")
      self.outputs_test  = self.outputs_train #same here
      # self.outputs_gen= self.model.textGenerate()
      # self.outputs_test  = model.forward(self.inputs,self.params,1.,mode="Train")


    # self.summary = tf.summary.merge_all()
    for item in tf.trainable_variables():
      print(item)
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
    # outputs = sess.run([self.train_op, self.global_step, self.summary] + outputs, feed_dict={self.inputs[k]: inputs[k] for k in self.inputs.keys() if k!="model"})
    outputs = sess.run([self.train_op, self.global_step] + outputs, feed_dict={self.inputs[k]: inputs[k] for k in self.inputs.keys() if k!="model"})
    ret = {keys[i]: outputs[i+2] for i in range(len(keys))}
    ret["global_step"] = outputs[1]
    # ret["summary"] = outputs[2]

    return ret

  def batch_test(self, sess, inputs):
    keys = list(self.outputs_test.keys())
    outputs = [self.outputs_test[key] for key in keys]
    outputs = sess.run(outputs, feed_dict={self.inputs[k]: inputs[k] for k in self.inputs.keys() if k!="model"})
    return {keys[i]: outputs[i] for i in range(len(keys))}

  def freq_calc(self,sample_input):
    sample_input_list=sample_input.tolist()
    return([[sample_input_list[0].count(word)*(1-self.params["stop_words"][word]) for word in sample_input_list[0]]])

  def test_textGen(self, sess):
    sample_input_list=[[self.vocab['<EOS>'] for _ in range(self.params["max_seqlen"])]]          
    sample_input=np.array(sample_input_list)
    sample_frequency=self.freq_calc(sample_input)
    # sample_input=input_idx
    seq_len=self.params["generate_len"]
    for k in range(seq_len):
      # feed_dict_text={self.inputs["tokens"]:sample_input,self.inputs["targets"]:sample_input,self.inputs["length"]:[k+1],self.theta_gen:1./self.params["num_topics"]*np.ones((1,self.params["num_topics"]))}    
      feed_dict_text={self.inputs["tokens"]:sample_input,self.inputs["targets"]:sample_input,self.inputs["frequency"]:sample_frequency,self.inputs["length"]:[k+1]}          
      generated_idx = sess.run(self.model.textGenerate(), feed_dict=feed_dict_text)    
      revised_text=" ".join([self.reverse_vocab[sample_input[0][idx]] for idx in range(k+1)])
      generated_text=" ".join([self.reverse_vocab[item] for item in generated_idx[0]])
      sample_input[0][k+1]=generated_idx[0][k]
      sample_frequency=self.freq_calc(sample_input)
    return revised_text


  def run_epoch(self, sess, datasets,train_num_batches,vocab,epoch_num):
    self.vocab=vocab
    self.reverse_vocab=dict(zip(vocab.values(),vocab.keys()))


    def switch_calc(topics_all,topics_non_idx):
      non_topics=[[item[0][item[1]>0] for item in list(zip(topics_all,topics_non_idx))]][0]
      topics_roll=[np.roll(item,shift=-1) for item in non_topics]
      next_compare=[ x==y for (x,y) in zip(non_topics, topics_roll)]
      next_compare=[item[:-1] for item in next_compare]
      Switch_P=np.mean([np.mean(item) for item in next_compare])
      return Switch_P

    train_ppl=[]
    valid_ppl=[]
    test_ppl=[]

    train_token,train_indic, train_theta_kl,train_phi_theta,train_switch,train_acc,train_theta_ent,train_phi_ent=[],[],[],[],[],[],[],[]
    valid_token,valid_indic, valid_theta_kl,valid_phi_theta,valid_switch,valid_acc,valid_theta_ent,valid_phi_ent=[],[],[],[],[],[],[],[]

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
      train_repre.append(train_outputs["repre"])
      train_ppl.append(train_outputs["token_ppl"])
      train_acc.append(train_outputs["accuracy"])
      train_theta_ent.append(train_outputs["theta_entropy"])
      train_phi_ent.append(train_outputs["phi_entropy"])



      beta=train_outputs["beta"]
      theta=train_outputs["theta"]      

      topics_all=train_outputs["all_topics"]
      topics_non_idx=train_outputs["non_stop_indic"]


      train_mini_switch=switch_calc(topics_all,topics_non_idx)
      train_switch.append(train_mini_switch)


      pbar.set_description("token: %f, theta_kl: %f, indicator: %f, phi_theta: %f, ppx: %f, theta_entropy: %f, phi_entropy: %f, SwitchP: %f" %(train_outputs["token_loss"],train_outputs["theta_kl_loss"],train_outputs["indicator_loss"],train_outputs["phi_theta_kl_loss"],train_outputs["token_ppl"],train_outputs["theta_entropy"],train_outputs["phi_entropy"],train_mini_switch))      
      # self.writer.add_summary(train_outputs["summary"], train_outputs["global_step"])
    self.non_topics=[]
    self.original_text=[]
    print('epoch_num: ',epoch_num)
    if epoch_num==(self.params["num_epochs"]-1):
      self.non_topics=[[([self.reverse_vocab[word] for word in item[2][item[1]>0]],item[0][item[1]>0]) for item in list(zip(topics_all[15:25],topics_non_idx[15:25],batch["targets"][15:25]))]][0]
      self.original_text=[" ".join([self.reverse_vocab[word] for word in sentence]) for sentence in batch["targets"][15:25]]
      # phi_vec=train_outputs["phi"][15][topics_non_idx[15]>0]
      # phi_entropy=[np.sum(-item*np.log(item)) for item in phi_vec]
      # print('theta_vec: ',train_outputs["theta"][15:18])
      # print('theta_entropy: ',[-np.sum(np.array(theta)*np.log(np.array(theta)+1e-10)) for theta in train_outputs["theta"][15:18]])
      # print('phi_entropy',phi_entropy)
      # for item in non_topics:
      #   print(list(zip(item[0],item[1])))
      #   print('\n')

    for batch in dataset_dev():
      valid_outputs = self.batch_test(sess, batch)
      valid_loss.append(valid_outputs["loss"])      
      valid_theta_kl.append(valid_outputs["theta_kl_loss"])
      valid_phi_theta.append(valid_outputs["phi_theta_kl_loss"])      
      valid_indic.append(valid_outputs["indicator_loss"])
      valid_token.append(valid_outputs["token_loss"])
      valid_theta.append(valid_outputs["theta"])
      valid_repre.append(valid_outputs["repre"])
      valid_ppl.append(valid_outputs["token_ppl"])
      valid_acc.append(valid_outputs["accuracy"])
      valid_theta_ent.append(valid_outputs["theta_entropy"])
      valid_phi_ent.append(valid_outputs["phi_entropy"])


      valid_topics_all=valid_outputs["all_topics"]
      valid_topics_non_idx=valid_outputs["non_stop_indic"]
      valid_switch.append(switch_calc(valid_topics_all,valid_topics_non_idx))

    self.sample_text=[]
    if epoch_num==(self.params["num_epochs"]-1):
      self.sample_text=self.test_textGen(sess)    
      print('sample_text: ',self.sample_text,'\n')

    train_loss = np.mean(train_loss)
    train_token=np.mean(train_token)
    train_indic=np.mean(train_indic)
    train_theta_kl=np.mean(train_theta_kl)
    train_phi_theta=np.mean(train_phi_theta)
    train_switch=np.mean(train_switch)
    train_ppl=np.mean(train_ppl)
    train_acc=np.mean(train_acc)
    train_theta_ent=np.mean(train_theta_ent)
    train_phi_ent=np.mean(train_phi_ent)

    valid_loss = np.mean(valid_loss)
    valid_token=np.mean(valid_token)    
    valid_theta_kl=np.mean(valid_theta_kl)
    valid_phi_theta=np.mean(valid_phi_theta)
    valid_indic=np.mean(valid_indic)
    valid_switch=np.mean(valid_switch)
    valid_ppl=np.mean(valid_ppl)
    valid_acc=np.mean(valid_acc)
    valid_theta_ent=np.mean(valid_theta_ent)
    valid_phi_ent=np.mean(valid_phi_ent)    

    # test_loss = np.mean(test_loss)

    train_theta, valid_theta, test_theta = np.vstack(train_theta), np.vstack(valid_theta), []
    train_repre, valid_repre, test_repre = np.vstack(train_repre), np.vstack(valid_repre), []
    # train_label, valid_label, test_label = np.vstack(train_label), np.vstack(valid_label), np.vstack(test_label)

    # train_res = [train_loss, train_theta, train_repre]
    # valid_res = [valid_loss, valid_theta, valid_repre]
    # test_res = [test_loss, test_theta, test_repre]
    # train_res=[train_loss,train_token,train_indic,train_theta_kl,train_phi_theta]
    # valid_res=[valid_loss,valid_token,valid_indic,valid_theta_kl,valid_phi_theta]    
    test_res=[[]]
    train_res={"train_loss":train_loss,"train_token":train_token,"train_indic":train_indic,"train_theta_kl":train_theta_kl,"train_phi_theta":train_phi_theta,"train_ppl":train_ppl,"train_acc":train_acc,"train_theta_ent":train_theta_ent,"train_phi_ent":train_phi_ent}
    valid_res={"valid_loss":valid_loss,"valid_token":valid_token,"valid_indic":valid_indic,"valid_theta_kl":valid_theta_kl,"valid_phi_theta":valid_phi_theta,"valid_ppl":valid_ppl,"valid_switch":valid_switch,"valid_acc":valid_acc,"valid_theta_ent":valid_theta_ent,"valid_phi_ent":valid_phi_ent}

    print('\n')
    print("train ==> loss: {:.4f}, token: {:.4f}, indicator: {:.4f} , theta_kl: {:.4f}, phi_theta: {:.4f}, switch:{:.4f}, ppl: {:.4f}, theta_ent: {:.4f}, phi_ent: {:.4f}, acc: {:.4f}".format(train_loss,train_token,train_indic,train_theta_kl,train_phi_theta,train_switch,train_ppl,train_theta_ent,train_phi_ent,train_acc))
    print("valid ==> loss: {:.4f}, token: {:.4f}, indicator: {:.4f} , theta_kl: {:.4f}, phi_theta: {:.4f}, switch:{:.4f}, ppl: {:.4f}, theta_ent: {:.4f}, phi_ent: {:.4f}, acc: {:.4f}".format(valid_loss,valid_token,valid_indic,valid_theta_kl,valid_phi_theta,valid_switch,valid_ppl,valid_theta_ent,valid_phi_ent,valid_acc))
    print('\n')

    return train_res, valid_res, test_res,beta,self.sample_text,self.non_topics,self.original_text

  def run(self, sess, datasets,train_num_batches,vocab,save_info):
    best_valid_loss = 1e10
    # self.writer = tf.summary.FileWriter(os.path.join(self.params["save_dir"], "train"), sess.graph)
    train_dict={"train_loss":[],"train_token":[],"train_indic":[],"train_theta_kl":[],"train_phi_theta":[],"train_acc":[],"train_theta_ent":[],"train_phi_ent":[]}
    valid_dict={"valid_loss":[],"valid_token":[],"valid_indic":[],"valid_theta_kl":[],"valid_phi_theta":[],"valid_ppl":[],"valid_switch":[],"valid_acc":[],"valid_theta_ent":[],"valid_phi_ent":[]}
    # valid_loss_all={"valid_loss":[],"valid_token":[],"valid_indic":[],"valid_theta_kl":[],"valid_phi_theta":[]}

    for i in range(self.params["num_epochs"]):
      train_res, valid_res, test_res,beta,output_text,non_topics,original_text = self.run_epoch(sess, datasets,train_num_batches,vocab,i)
      for key in train_dict:
        train_dict[key].append(train_res[key])
      for key in valid_dict:
        valid_dict[key].append(valid_res[key])
      if i==(self.params["num_epochs"]-1):
        beta_list,beta_values=print_top_words(beta, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0],name_beta="")            
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path+"/"+self.params["save_dir"], save_info[1]+".pkl"), "wb") as f:
      beta_dict={"beta_names":beta_list,"beta_values":beta_values}
      generated={"gen_text":output_text}
      assigned_topics={"non_topics":non_topics}
      original_text={"original_text":original_text}
      pkl.dump([train_dict, valid_dict,beta_list,save_info[0],beta_dict,generated,assigned_topics,original_text], f)





