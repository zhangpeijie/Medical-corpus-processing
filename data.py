#-*-coding:utf-8 -*-
# 第一步：数据处理
# pikle是一个将任意复杂的对象转成对象的文本或二进制表示的过程。
# 同样，必须能够将对象经过序列化后的形式恢复到原有的对象。
# 在 Python 中，这种序列化过程称为 pickle，
# 可以将对象 pickle 成字符串、磁盘上的文件或者任何类似于文件的对象，
# 也可以将这些字符串、文件或任何类似于文件的对象 unpickle 成原来的对象。
import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-bod": 1, "I-bod": 2,
             "B-sym": 3, "I-sym": 4,
             "B-dis": 5, "I-dis": 6,
             "B-tes": 7, "I-tes": 8,
             }


# 输入train_data文件的路径，读取训练集的语料，输出train_data
def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        '''lines的形状为['北\tB-LOC\n','京\tI-LOC\n','的\tO\n','...']总共有2220537个字及对应的tag'''
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            # char 与 label之间有个空格
            # ine.strip()的意思是去掉每句话句首句尾的空格
            # .split()的意思是根据空格来把整句话切割成一片片独立的字符串放到数组中，同时删除句子中的换行符号\n
            [char, label] = line.strip().split()
            # 把一个个的字放进sent_
            sent_.append(char)
            # 把字后面的tag放进tag_
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


# 由train_data来构造一个(统计非重复字)字典{'第一个字':[对应的id,该字出现的次数],'第二个字':[对应的id,该字出现的次数], , ,}
# 去除低频词，生成一个word_id的字典并保存在输入的vocab_path的路径下，
# 保存的方法是pickle模块自带的dump方法，保存后的文件格式是word2id.pkl文件
def vocab_build(vocab_path, corpus_path, min_count):
    """
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    # sent_的形状为['我',在'北','京']，对应的tag_为['O','O','B-LOC','I-LOC']
    for sent_, tag_ in data:
        for word in sent_:
            # 如果字符串只包含数字则返回 True 否则返回 False。
            if word.isdigit():
                word = '<NUM>'
            # A-Z：(\u0041-\u005a)    a-z ：\u0061-\u007a
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                # [len(word2id)+1, 1]用来统计[位置标签，出现次数]，第一次出现定为1
                word2id[word] = [len(word2id) + 1, 1]
            else:
                # word2id[word][1]实现对词频的统计，出现次数累加1
                word2id[word][1] += 1
    # 用来统计低频词
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        # 寻找低于某个数字的低频词
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        # 把这些低频词从字典中删除
        del word2id[word]
    # 删除低频词后为每个字重新建立id，而不再统计词频
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        # 序列化到名字为word2id.pkl文件
        pickle.dump(word2id, fw)


# 通过pickle模块自带的load方法(反序列化方法)加载输出word2id
def read_dictionary(vocab_path):
    """
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        # 反序列化方法加载输出
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id





def sentence2id(sent, word2id):
    """
    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        # 如果sent中的词在word2id找不到，用<UNK>--->3905来表示
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


# 输入vocab，vocab就是前面得到的word2id，embedding_dim=300
def random_embedding(vocab, embedding_dim):
    """
    :param vocab:
    :param embedding_dim:
    :return:
    """
    # 返回一个len(vocab)*embedding_dim=3905*300的矩阵(每个字投射到300维)作为初始值
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


# padding,输入一句话，不够标准的样本用pad_mark来补齐
''' 
输入：seqs的形状为二维矩阵，形状为[[33,12,17,88,50]-第一句话
                                 [52,19,14,48,66,31,89]-第二句话
                                                    ] 
输出：seq_list为seqs经过padding后的序列
      seq_len_list保留了padding之前每条样本的真实长度
      seq_list和seq_len_list用来喂给feed_dict
'''


def pad_sequences(sequences, pad_mark=0):
    '''
    :param sequences:
    :param pad_mark:
    :return:
    '''
    # 返回一个序列中长度最长的那条样本的长度
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        # 由元组格式()转化为列表格式[]
        seq = list(seq)
        # 不够最大长度的样本用0补上放到列表seq_list
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        # seq_len_list用来统计每个样本的真实长度
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


# 生成batch
''' seqs的形状为二维矩阵，形状为[[33,12,17,88,50....]...第一句话
                                [52,19,14,48,66....]...第二句话
                                                    ] 
   labels的形状为二维矩阵，形状为[[0, 0, 3, 4]....第一句话
                                 [0, 0, 3, 4]...第二句话
                                             ]
'''


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        # sent_的形状为[33,12,17,88,50....]句中的字在Wordid对应的位置标签
        # 如果tag_形状为['O','O','B-LOC','I-LOC']，对应的label_形状为[0, 0, 3, 4]
        # 返回tag2label字典中每个tag对应的value值
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        # 保证了seqs的长度为batch_size
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels
