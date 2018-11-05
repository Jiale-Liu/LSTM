# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:41:38 2018

@author: liujiale
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:41:28 2018

@author: liujiale
"""

import numpy as np


#from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import adam
from keras import optimizers,regularizers
import scipy as sp
from scipy import io
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_json
import theano
from keras.layers import LSTM, Dense
import h5py
import math
from sklearn import metrics

#import seq2seq
#from seq2seq.models import AttentionSeq2Seq

from keras.layers import merge
regularizers

data_train_18f=np.random.random((5263365,2,18))
label_train_1f=np.random.random((5263365,2))
data_test_18f=np.random.random((1327815,2,18))
label_test_1f=np.random.random((1327815,2))
data_val_18f=np.random.random((2724890,2,18))
label_val_1f=np.random.random((2724890,2))

TIME_STEPS = 2
INPUT_DIM = 18
lstm_units = 8
nb_classes = 1

data_flag=sp.io.matlab.mio.loadmat('surrounding_3.mat')
exclude=['_globals_','_header_','_version_']
for obj in data_flag.keys():
    if obj not in exclude:
        exec(obj+'=data_flag["'+obj+'"]')  
train_18f=np.asarray(data_flag['train_18f'],dtype="float32") 
train_1flag=np.asarray(data_flag['train_1flag'],dtype="float32")
test_18f=np.asarray(data_flag['test_18f'],dtype="float32") 
test_1flag=np.asarray(data_flag['test_1flag'],dtype="float32") 
val_18f=np.asarray(data_flag['val_18f'],dtype="float32")
val_1flag=np.asarray(data_flag['val_1flag'],dtype="float32") 

for i in range(0,10526730):
    data_train_18f[math.floor(i/2),i%2,:]=train_18f[i,:]

for i in range(0,2655630):
    data_test_18f[math.floor(i/2),i%2,:]=test_18f[i,:]

for i in range(0,5449780):
    data_val_18f[math.floor(i/2),i%2,:]=val_18f[i,:]

for i in range(0,10526730):
    label_train_1f[math.floor(i/2),i%2]=train_1flag[i,0]

for i in range(0,2655630):
    label_test_1f[math.floor(i/2),i%2]=test_1flag[i,0]

for i in range(0,5449780):
    label_val_1f[math.floor(i/2),i%2]=val_1flag[i,0]
    


# first way attention
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul



# build RNN model with attention
inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
drop1 = Dropout(0.15)(inputs)
lstm_out = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.6,return_sequences=True,), name='bilstm')(drop1)
lstm_out1 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm1')(lstm_out )
lstm_out2 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm2')(lstm_out1 )

lstm_out2 = merge([lstm_out2,lstm_out],mode='sum')

lstm_out3 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm3')(lstm_out2 )
lstm_out4 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm4')(lstm_out3 )
lstm_out4 = merge([lstm_out2,lstm_out4],mode='sum')

lstm_out5 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm5')(lstm_out4 )
lstm_out6 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm6')(lstm_out4 )
lstm_out6 = merge([lstm_out6,lstm_out4],mode='sum')

lstm_out7 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm7')(lstm_out6 )
lstm_out8 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm8')(lstm_out7 )
lstm_out8 = merge([lstm_out6,lstm_out8],mode='sum')

lstm_out9 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm9')(lstm_out8 )
lstm_out10 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm10')(lstm_out9 )
lstm_out10 = merge([lstm_out10,lstm_out8],mode='sum')

lstm_out11 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm11')(lstm_out10 )
lstm_out12 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm12')(lstm_out11 )
lstm_out12 = merge([lstm_out10,lstm_out12],mode='sum')


lstm_out13 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm13')(lstm_out12 )
lstm_out14 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm14')(lstm_out13 )
lstm_out14 = merge([lstm_out14,lstm_out12],mode='sum')

lstm_out15 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm15')(lstm_out14 )
lstm_out16 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm16')(lstm_out15 )
lstm_out16 = merge([lstm_out14,lstm_out16],mode='sum')

lstm_out17 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm17')(lstm_out16 )
lstm_out18 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm18')(lstm_out17 )
lstm_out18 = merge([lstm_out18,lstm_out16],mode='sum')

lstm_out19 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm19')(lstm_out18 )
lstm_out20 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm20')(lstm_out19 )
lstm_out20 = merge([lstm_out20,lstm_out18],mode='sum')

lstm_out21 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm21')(lstm_out20 )
lstm_out22 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm22')(lstm_out21 )
lstm_out22 = merge([lstm_out20,lstm_out22],mode='sum')

lstm_out23 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm23')(lstm_out22 )
lstm_out24 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm24')(lstm_out23 )
lstm_out24 = merge([lstm_out24,lstm_out22],mode='sum')


lstm_out25 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm25')(lstm_out24 )
lstm_out26 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm26')(lstm_out25 )
lstm_out26 = merge([lstm_out24,lstm_out26],mode='sum')

lstm_out27 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm27')(lstm_out26 )
lstm_out28 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm28')(lstm_out27 )
lstm_out28 = merge([lstm_out28,lstm_out26],mode='sum')

lstm_out29 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm29')(lstm_out28 )
lstm_out30 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm30')(lstm_out29 )
lstm_out30 = merge([lstm_out28,lstm_out30],mode='sum')

lstm_out31 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm31')(lstm_out30 )
lstm_out32 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm32')(lstm_out31 )
lstm_out32 = merge([lstm_out32,lstm_out30],mode='sum')

lstm_out33 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm33')(lstm_out32 )
lstm_out34 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm34')(lstm_out33 )
lstm_out34 = merge([lstm_out34,lstm_out32],mode='sum')

lstm_out35 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm35')(lstm_out34 )
lstm_out36 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm36')(lstm_out35 )
lstm_out36 = merge([lstm_out34,lstm_out36],mode='sum')

lstm_out37 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm37')(lstm_out36 )
lstm_out38 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm38')(lstm_out37 )
lstm_out38 = merge([lstm_out38,lstm_out36],mode='sum')

lstm_out39 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm39')(lstm_out38 )
lstm_out40 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm40')(lstm_out39 )
lstm_out40 = merge([lstm_out38,lstm_out40],mode='sum')


lstm_out41 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm41')(lstm_out40 )
lstm_out42 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm42')(lstm_out41 )
lstm_out42 = merge([lstm_out42,lstm_out40],mode='sum')

lstm_out43 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm43')(lstm_out42 )
lstm_out44 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm44')(lstm_out43 )
lstm_out44 = merge([lstm_out44,lstm_out42],mode='sum')

lstm_out45 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm45')(lstm_out44 )
lstm_out46 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm46')(lstm_out45 )
lstm_out46 = merge([lstm_out44,lstm_out46],mode='sum')

lstm_out47 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm47')(lstm_out46 )
lstm_out48 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm48')(lstm_out47 )
lstm_out48 = merge([lstm_out48,lstm_out46],mode='sum')

lstm_out49 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm49')(lstm_out48 )
lstm_out50 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm50')(lstm_out49 )
lstm_out50 = merge([lstm_out48,lstm_out50],mode='sum')

lstm_out51 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm51')(lstm_out50 )
lstm_out52 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm52')(lstm_out51 )
lstm_out52 = merge([lstm_out52,lstm_out50],mode='sum')

lstm_out53 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm53')(lstm_out52 )
lstm_out54 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm54')(lstm_out53 )
lstm_out54 = merge([lstm_out54,lstm_out52],mode='sum')

lstm_out55 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm55')(lstm_out54 )
lstm_out56 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm56')(lstm_out55 )
lstm_out56 = merge([lstm_out54,lstm_out56],mode='sum')

lstm_out57 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm57')(lstm_out56 )
lstm_out58 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm58')(lstm_out57 )
lstm_out58 = merge([lstm_out58,lstm_out56],mode='sum')

lstm_out59 = Bidirectional(LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm59')(lstm_out58 )
lstm_out60 = Bidirectional(LSTM(lstm_units,activation='relu',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),dropout=0.6,return_sequences=True,), name='bilstm60')(lstm_out59 )
lstm_out60 = merge([lstm_out58,lstm_out60],mode='sum')


attention_mul = attention_3d_block(lstm_out60)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.15)(attention_flatten)
output = Dense(2, activation='softmax')(drop2)
model = Model(inputs=inputs, outputs=output)


# second way attention
#inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
#units = 32
#activations = LSTM(units, activation='sigmoid',recurrent_activation='tanh',return_sequences=True,kernel_initializer='TruncatedNormal',recurrent_initializer='orthogonal', dropout=0.7, recurrent_dropout=0.6,use_bias=True,name='lstm_layer')(inputs)
#activations1 = LSTM(units,activation='tanh',recurrent_activation='relu', return_sequences=True,kernel_initializer='TruncatedNormal',recurrent_initializer='orthogonal',dropout=0.7, recurrent_dropout=0.6, name='lstm_layer1')(activations)
#activations2 = LSTM(units,activation='tanh',recurrent_activation='relu', return_sequences=True,kernel_initializer='TruncatedNormal',recurrent_initializer='orthogonal',dropout=0.7, recurrent_dropout=0.6, name='lstm_layer2')(activations1)
#activations3 = LSTM(units,activation='tanh',recurrent_activation='relu', return_sequences=True,kernel_initializer='TruncatedNormal',recurrent_initializer='orthogonal',dropout=0.6, recurrent_dropout=0.6, name='lstm_layer3')(activations2)


#attention = Dense(1, activation='tanh')(activations3)
#attention = Flatten()(attention)
#attention = Activation('softmax')(attention)
#attention = RepeatVector(units)(attention)
#attention = Permute([2, 1], name='attention_vec')(attention)
#attention_mul = merge([activations, attention], mode='mul', name='attention_mul')
#out_attention_mul = Flatten()(attention_mul)
#output = Dense(2, activation='sigmoid')(out_attention_mul)
#model = Model(inputs=inputs, outputs=output)
    

# second way attention
#inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
#units = 32

#activations = LSTM(units,activation='relu',recurrent_activation='tanh',return_sequences=True,kernel_initializer='uniform', dropout=0.6,name='lstm_layer')(inputs)
#activations1 = LSTM(units,activation='tanh',recurrent_activation='relu', return_sequences=True,kernel_initializer='uniform',dropout=0.6,name='lstm_layer1')(activations)
#activations2 = LSTM(units,activation='relu',recurrent_activation='relu', return_sequences=True,kernel_initializer='uniform', name='lstm_layer2')(activations1)
#activations3 = LSTM(units,activation='relu',recurrent_activation='relu', return_sequences=True,kernel_initializer='uniform', name='lstm_layer3')(activations2)

#attention = Dense(1, activation='tanh')(activations3)
#attention = Flatten()(attention)
#attention = Activation('softmax')(attention)
#attention = RepeatVector(units)(attention)
#attention = Permute([2, 1], name='attention_vec')(attention)
#attention_mul = merge([activations, attention], mode='mul', name='attention_mul')
#out_attention_mul = Flatten()(attention_mul)
#output = Dense(2, activation='sigmoid')(out_attention_mul)
#model = Model(inputs=inputs, outputs=output)




#learning_rate = 0.1 
#adam = Adam(lr=learning_rate)

#adam = Adam(lr=learning_rate, decay=learning_rate/nb_epoch, momentum=0.9, nesterov=True)

#sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
#decay=lr/nb_epoch, momentum=0.9,clipnorm=1.)  





model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


x_train = data_train_18f
y_train = label_train_1f



x_val = data_val_18f
y_val = label_val_1f



model.fit(x_train, y_train,
          batch_size=500, nb_epoch=5,
          validation_data=(x_val, y_val))






x_test = data_test_18f
y_test = label_test_1f
print("",y_test.shape)

print(len(y_test))



           
score = model.evaluate(x_test, y_test, batch_size=100)
print(score)
                                
proba = model.predict(x_test, batch_size=100)

print("",proba.shape)


json_string = model.to_json()  
open('lstm_space_control.json','w').write(json_string)  
model.save_weights('lstm_space_weights.h5')  
np.savetxt("lstm_space_proba_control.txt",proba)

#y_test=y_test[:,1]
#proba=proba[:,1]
combine=np.hstack((y_test,proba))
print("",combine.shape)



np.savetxt("combine.txt",combine)

