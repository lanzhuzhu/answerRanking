from keras.models import model_from_json
import numpy as np
import os
from  keras.utils import np_utils
from collections import defaultdict
# from nnet_layer import AttentionLSTM
from nnet_layer import SimilarityMatrixLayer



def map_score(qids, labels, preds):
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    average_precs = []
    for qid, candidates in qid2cand.iteritems():
        average_prec = 0
        running_correct_count = 0
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            if label > 0:
                running_correct_count += 1
                average_prec += float(running_correct_count) / i
        average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score


if __name__ == '__main__':

    data_dir = "engdata/" #"trainall/"

    print('Loading data...')
    q_test = np.load(os.path.join(data_dir, 'train.questions.npy'))
    a_test = np.load(os.path.join(data_dir, 'train.answers.npy'))
    q_overlap_test = np.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
    a_overlap_test = np.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
    Y_test = np.load(os.path.join(data_dir, 'train.labels.npy'))
    # y_test = np_utils.to_categorical(Y_test, 2)
    print('TEST question shape', q_test.shape)
    print('TEST answer shape', a_test.shape)

    mpath = os.path.join(data_dir, "engdatabiLSTM_SM.json")
    json_string=open(mpath).read()
    # print json_string
    # model = model_from_json(json_string,custom_objects={'SimilarityMatrixLayer':SimilarityMatrixLayer})

    model = model_from_json(json_string)
    model.load_weights(data_dir + "engdatabiLSTM_SM_weights.h5")
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])  # negative_log_likelihood
    print ("begining test!")
    # score = model.evaluate([q_test,q_overlap_test,a_test,a_overlap_test], y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    right = 0
    count = 0
    i = 0
    batch_size = 2

    predict = model.predict([q_test[i:i + batch_size], q_overlap_test[i:i + batch_size], a_test[i:i + batch_size],
                             a_overlap_test[i:i + batch_size]], batch_size=batch_size, verbose=1)
    print ("the prediction is:", predict)

    for m in xrange(10):

        predict = model.predict([q_test[i:i + batch_size], q_overlap_test[i:i + batch_size], a_test[i:i + batch_size],
                                 a_overlap_test[i:i + batch_size]], batch_size=batch_size, verbose=1)
        print ("the lable is:", Y_test[i:i + batch_size], predict)
        # lable = [int(p>0.5) for p in predict]
        lable = [int(p >0.5) for p in predict]
        right += sum([int(lable[k] == Y_test[m * batch_size + k]) for k in xrange(len(lable))])
        print ("the predict lable is:", lable, right)
        i += batch_size
        count += batch_size
    print ("the accuracy is:", right / float(count))
