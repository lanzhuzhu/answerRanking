from __future__ import print_function
from __future__ import absolute_import
from datetime import datetime
import pandas as pd
import subprocess
from keras.callbacks import EarlyStopping

from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Lambda,Flatten
from keras.layers import Embedding,Merge,Input,merge
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
from keras import backend as K
from keras.utils import np_utils
from nnet_layer import similarityMatrix,AttentionLSTM
from keras.optimizers import Adadelta
from keras.layers.merge import concatenate
import numpy as np
import os

# np.random.seed(1337)  # for reproducibility
"""

"""

# set parameters:
batch_size = 50  #32
nb_filter = 150 #250
filter_length = 5 #3
nb_epoch = 70
n_outs = 1
lstm_output_size = 55
activation = 'tanh'



if __name__ == '__main__':
    # loading data
    data_dir ="engdata"  #"trainall" #
    model_name = "biLSTM_CNN_SM_tensor"#Merge_tensor
    version = 3
    ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')

    q_train = np.load(os.path.join(data_dir, '{}.questions.npy'.format("train")))
    a_train = np.load(os.path.join(data_dir, 'train.answers.npy'))
    q_overlap_train = np.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
    a_overlap_train = np.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
    Y_train = np.load(os.path.join(data_dir, 'train.labels.npy'))

    q_test = np.load(os.path.join(data_dir, 'test.questions.npy'))
    a_test = np.load(os.path.join(data_dir, 'test.answers.npy'))
    q_overlap_test = np.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
    a_overlap_test = np.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
    y_test = np.load(os.path.join(data_dir, 'test.labels.npy'))
    qids_test = np.load(os.path.join(data_dir, 'test.qids.npy'))
    # y_test = np_utils.to_categorical(y_test,2)

    q_dev = np.load(os.path.join(data_dir, 'dev.questions.npy'))
    a_dev = np.load(os.path.join(data_dir, 'dev.answers.npy'))
    q_overlap_dev = np.load(os.path.join(data_dir, 'dev.q_overlap_indices.npy'))
    a_overlap_dev = np.load(os.path.join(data_dir, 'dev.a_overlap_indices.npy'))
    y_dev = np.load(os.path.join(data_dir, 'dev.labels.npy'))
    # y_dev = np_utils.to_categorical(y_dev,2)
    print('Loading data...')
    print('question shape', q_train.shape)
    print('answer shape', a_train.shape)
    # print('TEST question shape', q_test.shape)
    # print('TEST answer shape', a_test.shape)
    # print('DEV question shape', q_dev.shape)
    # print('DEV answer shape', a_dev.shape)


    numpy_rng = np.random.RandomState(123)
    q_max_sent_size = q_train.shape[1]
    a_max_sent_size = a_train.shape[1]

    overlap_ndim = 5
    print ("Generating random vocabulary for word overlap indicator features with dim:", overlap_ndim)
    dummy_word_id = np.max(a_overlap_train)
    print ("Gaussian")
    vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, overlap_ndim) * 0.25
    vocab_emb_overlap[-1] = 0

    # Load word2vec embeddings
    fname = os.path.join(data_dir, 'emb_vocab_embedding.npy')

    print("Loading word embeddings from", fname)
    vocab_emb = np.load(fname)
    vocab_ndim = vocab_emb.shape[1]
    print("Word embedding matrix size:", vocab_emb.shape)

    #############
    ndim = vocab_emb.shape[1] + vocab_emb_overlap.shape[1]
    print('Build model...')


    ####### shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(Y_train)))
    q_shuffled = q_train[shuffle_indices]
    a_shuffled = a_train[shuffle_indices]
    q_overlap_shuffled =q_overlap_train[shuffle_indices]
    a_overlap_shuffled =a_overlap_train[shuffle_indices]
    Y_train = Y_train[shuffle_indices]
    # Y_train=np_utils.to_categorical(Y_train,2)

    def mean_1d(X):
        return K.mean(X,axis=1)
    # attention_vec_a = mean_1d(attention_vec_q)
    def max_1d(X):
        return K.max(X, axis=1)
    meanpool = Lambda(mean_1d,output_shape=lambda x:(x[0],x[2]))
    maxpool = Lambda(max_1d, output_shape=lambda x:(x[0],x[2]))


    ###########question
    q_left = Input(shape=(q_max_sent_size,),dtype='int32',name = 'q_left')
    ql_emb = Embedding(vocab_emb.shape[0],
                              vocab_emb.shape[1],
                              weights = [vocab_emb],
                              input_length=q_max_sent_size)(q_left)

    q_right = Input(shape=(q_max_sent_size,),dtype='int32',name = 'q_right')
    qr_emb = Embedding(vocab_emb_overlap.shape[0],
                              vocab_emb_overlap.shape[1],
                              weights = [vocab_emb_overlap],
                              input_length=q_max_sent_size)(q_right)
    # print ("layer.output_shape",qr_emb.output_shape)
    q_input = concatenate([ql_emb,qr_emb], axis=2)
    # print ("layer.output_shape",q_input.shape.eval())

    # we add a Convolution1D, which will learn nb_filter word group filters of size filter_length:
    q_forwards = LSTM(lstm_output_size,return_sequences=True,dropout=0.25)(q_input) #,consume_less = "mem"
    q_backwards = LSTM(lstm_output_size,return_sequences=True,go_backwards=True,dropout=0.25)(q_input)
    q_after_LSTM = concatenate([q_forwards,q_backwards],axis=2)
    q_after_LSTM_dp = Dropout(0.2)(q_after_LSTM)

    q_CNN = Conv1D(filters=nb_filter,
                            kernel_size=filter_length,
                            padding='valid',
                            activation=activation,
                            strides=1)(q_after_LSTM_dp)

    q_after_CNN = Dropout(0.5)(q_CNN)
    nnet_q = Lambda(max_1d, output_shape=(nb_filter,))(q_after_CNN)

      ######

    ########### answer
    a_left = Input(shape=(a_max_sent_size,),dtype='int32',name = 'a_left')
    al_emb = Embedding(vocab_emb.shape[0],
                              vocab_emb.shape[1],
                              weights = [vocab_emb],
                              input_length=a_max_sent_size)(a_left)

    a_right = Input(shape=(a_max_sent_size,),dtype='int32',name = 'a_right')
    ar_emb = Embedding(vocab_emb_overlap.shape[0],
                              vocab_emb_overlap.shape[1],
                              weights = [vocab_emb_overlap],
                              input_length=a_max_sent_size)(a_right)
    # print ("layer.output_shape",qr_emb.output_shape)
    a_input = concatenate([al_emb,ar_emb],axis=2)
    # print ("layer.output_shape",a_input.shape.eval())

    a_forwards = LSTM(lstm_output_size,return_sequences=True,dropout=0.25)(a_input)
    a_backwards = LSTM(lstm_output_size,return_sequences=True,go_backwards=True,dropout=0.25)(a_input)


    a_after_LSTM = concatenate([a_forwards,a_backwards],axis=2)
    a_after_LSTM_dp = Dropout(0.2)(a_after_LSTM)


    a_CNN = Conv1D(filters=nb_filter,
                            kernel_size=filter_length,
                            padding='valid',
                            activation=activation,
                            strides=1)(a_after_LSTM_dp)

    a_after_CNN = Dropout(0.5)(a_CNN)
    nnet_a = Lambda(max_1d, output_shape=(nb_filter,))(a_after_CNN)



    merged = similarityMatrix([nnet_q,nnet_a])
    dropout1 = Dropout(0.3)(merged)

    joint = Dense(nb_filter*2+1,activation=activation)(dropout1)
    after_dp = Dropout(0.5)(joint)

    output = Dense(1,activation='sigmoid')(after_dp)
    # print ("layer.output_shape",output.output_shape)
    model = Model(inputs=[q_left,q_right,a_left,a_right],outputs = output)
    model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])  #negative_log_likelihood

    print ('write the test result!')
    outdir = os.path.join(data_dir,'exp.out/model={};version={};{}'.format(model_name, version,ts))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

    model.fit([q_shuffled,q_overlap_shuffled,a_shuffled,a_overlap_shuffled], Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              # validation_split= 0.1
              validation_data=([q_dev,q_overlap_dev,a_dev,a_overlap_dev], y_dev),
              callbacks=[early_stopping],
              )
    y_pred_test = model.predict([q_test,q_overlap_test,a_test,a_overlap_test],batch_size=batch_size)
    N=len(y_pred_test)
    df_submission = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
    df_submission['qid'] = qids_test
    df_submission['iter'] = 0
    df_submission['docno'] = np.arange(N)
    df_submission['rank'] = 0
    df_submission['sim'] = y_pred_test
    df_submission['run_id'] = 'lab'
    df_submission.to_csv(os.path.join(outdir, 'submission.txt'), header=False, index=False, sep=' ')

    df_gold = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
    df_gold['qid'] = qids_test
    df_gold['iter'] = 0
    df_gold['docno'] = np.arange(N)
    df_gold['rel'] = y_test
    df_gold.to_csv(os.path.join(outdir, 'gold.txt'), header=False, index=False, sep=' ')

    subprocess.call("/bin/sh run_eval.sh '{}'".format(outdir), shell=True)


    score = model.evaluate([q_test,q_overlap_test,a_test,a_overlap_test], y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    #
    json_string = model.to_json()
    open(outdir+"biLSTM_CNN.json","w").write(json_string)
    model.save_weights(outdir+"biLSTM_CNN_weights.h5")
    #

