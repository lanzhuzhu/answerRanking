# coding=utf-8
from keras.preprocessing.text import text_to_word_sequence
from alphabet import Alphabet
from collections import Counter
import numpy as np
import itertools
import cPickle
import gensim
import os

UNKNOWN_WORD_IDX = 0
np.random.seed(123)


# attention the code type and the dimension can be changed!
# data can be a file contains several sentences
def add_to_vocab(data, alphabet, n=None):
    sortedData = Counter(itertools.chain(*data)).most_common(n)
    words = [x[0] for x in sortedData]
    for token in words:
        alphabet.add(token)


def load_data_and_labels(outdir):
    with open(outdir + "question", "r") as querry:
        q_file = querry.readlines()
    q_train = [q.strip() for q in q_file]
    question_train = [text_to_word_sequence(s, split=" ") for s in q_train]

    with open(outdir + "answer", "r") as answers:
        a_file = answers.readlines()
    a_train = [a.strip() for a in a_file]
    answer_train = [text_to_word_sequence(s, split=" ") for s in a_train]

    with open(outdir + "labels", "r") as lable:
        lableFile = lable.readlines()
    Y_train = [lb.strip() for lb in lableFile]
    return [question_train, answer_train, Y_train]


# data can be multi files,the vocab can be extend by counter all the "UNKNOWN_WORD_IDX"words
def build_vocab(data, outdir=None):
    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    alphabet.add('<PAD/>')  # pad_index = 1
    for file in data:
        add_to_vocab(file, alphabet)
    print("build vocabulary：", len(alphabet))

    if outdir:
        filePath = os.path.join(outdir, 'vocab.pickle')
        cPickle.dump(alphabet, open(filePath, 'w+'))
        print("save alphabet", len(alphabet))
    # print "alphabet0",alphabet.keys()[:2]
    return alphabet


def update_vocab(alphabet,infile,vecFile,outdir=None):
    wFile = open(infile,"r")
    wordsAll = wFile.read().split("\n")
    words = []
    model = gensim.models.Word2Vec.load_word2vec_format(vecFile, binary=True)
    for w in wordsAll:
        s= w.strip()
        if s in model:
            words.append(s)
    print ("all words are %d ,useful words %d:"%(len(wordsAll),len(words)))
    for w in words:
        alphabet.add(w)
    print ("update vocabulary", len(alphabet))
    wFile.close()
    if outdir:
        filePath = os.path.join(outdir, 'vocab.pickle')
        cPickle.dump(alphabet, open(filePath, 'w'))
        print ("save alphabet", len(alphabet))
    return alphabet


def build_vocabVector(vocab, vecFile):
    model = gensim.models.KeyedVectors.load_word2vec_format(vecFile, binary=True)
    # print "the dimension of the vector is:",len(model.values()[0])
    # vocab = cPickle.load(open(vobFile))
    ndim = 50  # 50 #300 #400
    random_words_count = 0
    print ("words in vocab are:", len(vocab))
    vocab_emb = np.zeros((len(vocab), ndim))
    for word, idx in vocab.iteritems():
        if word in model:  # .decode('utf-8')
            word_vec = model[word]
            # print word_vec
        else:
            word_vec = np.random.uniform(-0.25, 0.25, ndim)
            random_words_count += 1
        vocab_emb[idx] = word_vec

    print ("Using zero vector as random")
    print ('random_words_count', random_words_count)
    print ("<PAD/>", vocab.get("<PAD/>"))
    print ("UNKNOWN_WORD_IDX", vocab.get("UNKNOWN_WORD_IDX"))
    print ("word2vec", vocab_emb.shape)

    vocab_embedding = np.array(vocab_emb)
    return vocab_embedding


# padding and convert to indices
def convert2indices(data, alphabet, max_sent_length=40):
    data_idx = []
    count = 0
    uknown = set()
    for sentence in data:
        ex = np.ones(max_sent_length)
        for i, token in enumerate(sentence):
            idx = alphabet.get(token, UNKNOWN_WORD_IDX)
            ex[i] = idx
            if idx == 0:
                count += 1
                uknown.add(token)
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('int32')
    print ("the unkonwn words appear %d times" % count)  # 2425)
    # print "unkown words:", uknown
    return data_idx


# compute the overlap words in file, if overlap,value is 1,else 0, if no words, is 2,thus must before the padding!
def compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length):
    stoplist = stoplist if stoplist else []
    q_indices, a_indices = [], []
    count = 0
    for question, answer in zip(questions, answers):
        q_set = set([q for q in question if q not in stoplist])
        a_set = set([a for a in answer if a not in stoplist])
        word_overlap = q_set.intersection(a_set)
        if not len(word_overlap): count += 1

        # print [wo for wo in word_overlap]
        q_idx = np.ones(q_max_sent_length) * 2
        for i, q in enumerate(question):
            value = 0
            if q in word_overlap:
                value = 1
            q_idx[i] = value
        q_indices.append(q_idx)

        a_idx = np.ones(a_max_sent_length) * 2
        for i, a in enumerate(answer):
            value = 0
            if a in word_overlap:
                value = 1
            a_idx[i] = value
        a_indices.append(a_idx)
        # pay attention the matrix : one line represents one sentence
    q_indices = np.vstack(q_indices).astype('int32')
    a_indices = np.vstack(a_indices).astype('int32')
    print ("no overlap qa pairs are:", count)
    return q_indices, a_indices


if __name__ == '__main__':
    outdir = "Health/"
    # wordsFile = "extraWords.txt"
    emb_path = "embeddings/aquaint+wiki.txt.gz.ndim=50.bin"
    [questions, answers, Y_train] = load_data_and_labels(outdir)
    # answers = [ans[:300] for ans in answers]
    alphabet = build_vocab([questions, answers],outdir=outdir)
    # alphabet = update_vocab(alphabet,wordsFile,emb_path,outdir)
    labels = np.array(Y_train).astype('int32')

    # reloead the total vocab and embeddings!
    # vobFile = "embeddings/total_vocab.pickle"
    # alphabet =  cPickle.load(open(vobFile))

    stoplist=None

    # reloead the embeddings!
    # fname_vocab = os.path.join(outdir, 'vocab.pickle')
    # alphabet = cPickle.load(open(fname_vocab))
    # emb_path = "embeddings/aquaint+wiki.txt.gz.ndim=50.bin"  # aquaint+wiki.txt.gz.ndim=50.bin"  #"GoogleNews-vectors-negative300.bin" #wiki.en.text.w2v_vector"

    train_basename = os.path.basename(emb_path)
    name, ext = os.path.splitext(train_basename)
    # print train_basename,name,ext
    vocab_embedding = build_vocabVector(alphabet, emb_path)
    outfile = os.path.join(outdir, 'emb_{}.npy'.format(name))
    print (outfile)
    np.save(outfile, vocab_embedding)

    ####

    q_max_sent_length = max(map(lambda x: len(x), questions))
    a_max_sent_length = max(map(lambda x: len(x), answers))
    print ('q_max_sent_length', q_max_sent_length)
    print ('a_max_sent_length', a_max_sent_length)
    q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length)
    questions_idx = convert2indices(questions, alphabet, q_max_sent_length)
    answers_idx = convert2indices(answers, alphabet, a_max_sent_length)
    print ('questions_idx:', questions_idx.shape)
    print ('answers_idx', answers_idx.shape)
    # outdir += "tocalVocab/"
    # print outdir
    np.save(os.path.join(outdir, '{}.questions.npy'.format("train")), questions_idx)
    np.save(os.path.join(outdir, '{}.answers.npy'.format("train")), answers_idx)
    np.save(os.path.join(outdir, '{}.labels.npy'.format("train")), labels)
    np.save(os.path.join(outdir, '{}.q_overlap_indices.npy'.format("train")), q_overlap_indices)
    np.save(os.path.join(outdir, '{}.a_overlap_indices.npy'.format("train")), a_overlap_indices)
