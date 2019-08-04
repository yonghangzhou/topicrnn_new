import tensorflow as tf
from tensorflow.python.keras.layers import Layer,LSTMCell,StackedRNNCells,RNN
from tensorflow import distributions as dist


class cmSimpleRNNCell(Layer):

    def __init__(self, units,beta,vocabulary_size,topics,**kwargs):
        self.units = units
        self.beta=beta
        self.state_size = units
        self.topics=topics
        self.vocabulary_size=vocabulary_size
        self.output_size =[topics,vocabulary_size]
        self.counter=0

        super(cmSimpleRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')

        self.bernoli_kernel=self.add_weight(shape=(self.units, 1),
                                      initializer='uniform',
                                      name='bernoli_kernel')        
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')

        self.vocab_kernel = self.add_weight(
            shape=(self.units,self.vocabulary_size),
            initializer='uniform',
            name='vocab_kernel')        
        self.built = True

    def call(self, inputs,indicators,indices,states):
        self.counter+=1
        ''' h_{t-1}: previous hidden state '''        
        prev_state = states[0]


        ''' Transformation of x to h dimension (this happens one by one for each word) W_{xh} * x '''
        x_h = tf.keras.backend.dot(inputs, self.kernel)


        ''' Next state h_t=W_{xh} '''
        next_state = x_h + tf.keras.backend.dot(prev_state, self.recurrent_kernel)


        ''' Bernoli sampling for stop words'''
        bern_p_params=tf.sigmoid(tf.nn.softplus(tf.keras.backend.dot(next_state,self.bernoli_kernel)))
        bern_p_params=tf.expand_dims(bern_p_params,-1)
        bern_dist=dist.Bernoulli(probs=bern_p_params)        
        bern_samples=tf.to_float(bern_dist.sample())

        ''' The output of each cell'''
        word_prob_out=tf.nn.softmax(tf.expand_dims(tf.keras.backend.dot(next_state, self.vocab_kernel),1)+(1-bern_samples)*self.beta,axis=-1)
        # word_prob_out=tf.nn.softmax(tf.expand_dims(tf.keras.backend.dot(next_state, self.vocab_kernel),1)+self.beta,axis=-1)

        return word_prob_out, [next_state]

