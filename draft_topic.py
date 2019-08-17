import numpy as np
import argparse
import os
import pickle as pkl
import numpy as np
import sys
import collections
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))

sample_topic=np.array([[2,0,0,2,0,2,5,0,1],[8,0,8,0,0,14,7,14,5],[0,4,4,0,0,5,0,5,0]])
print('sample_topic',sample_topic.shape)
sample_idx=  np.array([[1,0,0,1,0,1,0,0,1],[1,0,1,0,0,1,0,1,1],  [0,1,1,0,0,1,0,1,0]])


sample_out=[[item[0][item[1]>0] for item in list(zip(sample_topic,sample_idx))]][0]
sample_roll=[np.roll(item,shift=-1) for item in sample_out]

print('sample_out',sample_out)
print('sample_roll',sample_roll)
compare=[ x==y for (x,y) in zip(sample_out, sample_roll)]
compare=[item[:-1] for item in compare]
Switch_P=[np.mean(item) for item in compare]
print('Switch_P',Switch_P)

# print(list(zip(sample_out,sample_roll)))


# print('sample_out',sample_out)
# print('compare',compare)
# for item in sample_out[0]:
# 	print('item, ',item)
	# print(np.roll(item,shift=-1))
# print('sample_roll',sample_roll)


# print(list(zip(sample_topic,sample_idx)))

# out_idx=[[sample_item[doc_idx][idx] for idx in range(len(sample_item[doc_idx]))if sample_idx[doc_idx][idx] >0] for doc_idx in range(len(sample_topic))]
# print(out_idx)
# sample_array=np.array([[0.1,0.5,0.8,0.2],[0.3,0.7,0.4,0.2],[0.1,0.1,0.8,0.4],[0.5,0.6,0.3,0.2]])
# indicator=np.array([[1.,0.,1.,0.]])

# a=tf.constant(sample_array,dtype=tf.float64)
# indic=tf.constant(indicator,dtype=tf.float64)

# # a_idx_max=tf.where(indicator>0,tf.to_float(tf.argmax(a,axis=-1)),y=tf.zeros_like(indicator))
# a_idx_max=tf.expand_dims(tf.argmax(a,axis=-1),0)
# a_idx_max=tf.where(indicator>0,a_idx_max,-1*tf.ones_like(a_idx_max))
# tf.keras.layers.Lambda


# # a_idx_max*=indic
# # a_idx_max=tf.squeeze(a_idx_max,-1)
# # a_max_roll=tf.roll(a_idx_max,shift=-1,axis=-1)
# # a_compare=indic*tf.to_float(tf.equal(a_max_roll,a_idx_max))

# with tf.Session() as sess:
# 	# a,a_idx_max,a_max_roll,a_compare=sess.run((	a,a_idx_max,a_max_roll,a_compare))
# 	a,a_idx_max=sess.run((	a,a_idx_max))

# 	print("original:",a,'\n')
# 	print("max:",a_idx_max,'\n')
# 	# print("roll:",a_max_roll,'\n')
# 	# print("compare",a_compare,'\n')



