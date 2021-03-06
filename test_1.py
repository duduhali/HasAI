from __future__ import print_function
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Concatenate, Permute, Dropout,Add
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import numpy as np
import re
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

'''返回切分后的词
>>>tokenize('Bob dropped the apple. Where is the apple?')
['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
'''
def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        # print(line) # b'1 John went to the bedroom.\n'
        line = line.decode('utf-8').strip()
        # print(line) # 1 John moved to the bathroom.
        nid, line = line.split(' ', 1) #切割，只切第一个空格
        nid = int(nid) #序号
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            # print(q, a, supporting) #Where is Mary?  bathroom 1
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    file = f.readlines()
    data = parse_stories(file, only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data) ##定义一个函数，把字符串列表拼接在一起成功长字符串
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

import tarfile
from keras.utils.data_utils import get_file #加载文件方法
path = get_file('babi_tasks_1-20_v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)


file_train = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt'
file_test = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt'
train_stories = get_stories(tar.extractfile(file_train))
test_stories = get_stories(tar.extractfile(file_test))
print('test_stories',test_stories)

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
print('vocab')
print(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

print('inputs_train[0]')
print(inputs_train[0])
print('queries_train[0]')
print(queries_train[0])
print('answers_train[0]')
print(answers_train[0])

print ('inputs_test[0]')
print(inputs_test[0])
print('queries_test[0]')
print(queries_test[0])
print('answers_test[0]')
print(answers_test[0])


print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')


# ######################################## story - question_encoder ########################################
# # embed the input sequence into a sequence of vectors
# input_encoder_m = Sequential()
# input_encoder_m.add(Embedding(input_dim=vocab_size,
#                               output_dim=64,
#                               input_length=story_maxlen))
# input_encoder_m.add(Dropout(0.3))
# # output: (samples, story_maxlen, embedding_dim)
#
#
# # embed the question into a sequence of vectors
# question_encoder = Sequential()
# question_encoder.add(Embedding(input_dim=vocab_size,
#                                output_dim=64,
#                                input_length=query_maxlen))
# question_encoder.add(Dropout(0.3))
# # output: (samples, query_maxlen, embedding_dim)
# # compute a 'match' between input sequence elements (which are vectors)
# # and the question vector sequence
# from keras.layers import Dot
# d = Dot([input_encoder_m, question_encoder], axes=[2, 2], normalize=True)
# output = Activation('softmax')(d)
# match = Model(inputs=d, outputs=output)
#
# # match = Sequential()
# # match.add(dot)
# # match.add(Activation('softmax'))
# # output: (samples, story_maxlen, query_maxlen)
# ######################################## story - question_encoder ########################################
#
#
# # embed the input into a single vector with size = story_maxlen:
# input_encoder_c = Sequential()
# input_encoder_c.add(Embedding(input_dim=vocab_size,
#                               output_dim=query_maxlen,
#                               input_length=story_maxlen))
# input_encoder_c.add(Dropout(0.3))
# # output: (samples, story_maxlen, query_maxlen)
# # sum the match vector with the input vector:
# response = Sequential()
# response.add(Add([match, input_encoder_c]))
# # output: (samples, story_maxlen, query_maxlen)
# response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)
#
# # concatenate the match vector with the question vector,
# # and do logistic regression on top
# answer = Sequential()
# answer.add(Concatenate([response, question_encoder], concat_axis=-1))
#
#
# # the original paper uses a matrix multiplication for this reduction step.
# # we choose to use a RNN instead.
# answer.add(LSTM(32))
# # one regularization layer -- more would probably be needed.
# answer.add(Dropout(0.3))
# answer.add(Dense(vocab_size))
# # we output a probability distribution over the vocabulary
# answer.add(Activation('softmax'))
#
# # checkpoint
# checkpointer = ModelCheckpoint(filepath="./checkpoint.hdf5", verbose=1)
# # learning rate adjust dynamic
# lrate = ReduceLROnPlateau(min_lr=0.00001)
#
# answer.compile(optimizer='rmsprop', loss='categorical_crossentropy',
#                metrics=['accuracy'])
# # Note: you could use a Graph model to avoid repeat the input twice
# answer.fit(
#     [inputs_train, queries_train, inputs_train], answers_train,
#     batch_size=32,
#     nb_epoch=5000,
#     validation_data=([inputs_test, queries_test, inputs_test], answers_test),
#     callbacks=[checkpointer, lrate]
# )