# answerRanking
answer selection task for QA

## 编译环境：
- python2.7+
- keras2.0
- gensim
- cPickel

预训练的word2vec embeddings 从[这里](https://drive.google.com/open?id=0B-yipfgecoSBb1dTcW5MdVhGNkE)下载 .
放在embedding/目录下。

## 运行：
- python data.py
- python answerRank_bilstm.py
- python answerRank_cnn_bilstm_tensor.py
- python reloadTestModel.py

