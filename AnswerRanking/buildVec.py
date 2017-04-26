import numpy
import os
from alphabet import Alphabet
import cPickle

numpy.random.seed(123)
def load_bin_vec(fname):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print fname
    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    alphabet.add('<PAD/>')  # pad_index = 1
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, ndim = map(int, header.split())
        binary_len = numpy.dtype('float32').itemsize * ndim
        print 'vocab_size, layer1_size', vocab_size, ndim
        vocab_emb = numpy.zeros((vocab_size+2, ndim))
        vocab_emb[0] = numpy.random.uniform(-0.25, 0.25, ndim)
        vocab_emb[1] = numpy.random.uniform(-0.25, 0.25, ndim)
        count = 0
        for i, line in enumerate(xrange(vocab_size)):
            if i % 100000 == 0:
                print '.',
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            fid = alphabet.fid
            alphabet.add(word)
            vocab_emb[fid] = numpy.fromstring(f.read(binary_len), dtype='float32')
        vocab_embedding = numpy.array(vocab_emb)
        print "done!the vocab are",alphabet.fid
        print "Words found in  embeddings", vocab_embedding.shape
        return alphabet,vocab_embedding


if __name__ == '__main__':
    fname = "embeddings/aquaint+wiki.txt.gz.ndim=50.bin"
    outdir = "embeddings/"
    alphabet,vocab_embedding = load_bin_vec(fname)
    vec_filePath = os.path.join(outdir,'total_vocab.pickle')
    cPickle.dump(alphabet, open(vec_filePath, 'w'))
    print "alphabet", len(alphabet)
