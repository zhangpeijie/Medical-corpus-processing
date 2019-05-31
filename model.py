#-*-coding:utf-8 -*-
import codecs
import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger
from eval import conlleval

class HMM(object):

    def __init__(self):
        self.state_list = ['B','M','E','S']
        self.start_p = {}
        self.trans_p = {}
        self.emit_p = {}

        self.model_file = 'hmm_model.pkl'
        self.trained = False

    def train(self,datas,model_path=None):
        if model_path == None:
            model_path = self.model_file
        #统计状态频数
        state_dict = {}

        def init_parameters():
            for state in self.state_list:
                self.start_p[state] = 0.0
                self.trans_p[state] = {s:0.0 for s in self.state_list}
                self.emit_p[state] = {}
                state_dict[state] = 0

        def make_label(text):
            out_text = []
            if len(text) == 1:
                out_text = ['S']
            else :
                out_text += ['B']+['M']*(len(text)-2)+['E']
            return out_text

        init_parameters()
        line_nb = 0

        #监督学习方法求解参数，详情见统计学习方法10.3.1节
        for line in datas:
            line = line.strip()
            if not line:
                continue
            line_nb += 1

            word_list = [w for w in line if w != ' ']
            line_list = line.split()
            line_state = []
            for w in line_list:
                line_state.extend(make_label(w))

            assert len(line_state) == len(word_list)

            for i,v in enumerate(line_state):
                state_dict[v] += 1

                if i == 0:
                    self.start_p[v] += 1
                else :
                    self.trans_p[line_state[i-1]][v] += 1
                    self.emit_p[line_state[i]][word_list[i]] = self.emit_p[line_state[i]].get(word_list[i],0)+1.0

        self.start_p = {k: v*1.0/line_nb for k,v in self.start_p.items()}
        self.trans_p = {k:{k1: v1/state_dict[k1] for k1,v1 in v0.items()} for k,v0 in self.trans_p.items()}
        self.emit_p = {k:{k1: (v1+1)/state_dict.get(k1,1.0) for k1,v1 in v0.items()} for k,v0 in self.emit_p.items()}

        with open(model_path,'wb') as f:
            import pickle
            pickle.dump(self.start_p,f)
            pickle.dump(self.trans_p,f)
            pickle.dump(self.emit_p,f)
        self.trained = True
        print('model train done,parameters save to ',model_path)

    #读取参数模型
    def load_model(self,path):
        import pickle
        with open(path,'rb') as f:
            self.start_p = pickle.load(f)
            self.trans_p = pickle.load(f)
            self.emit_p = pickle.load(f)
        self.trained = True
        print('model parameters load done!')

    #维特比算法求解最优路径 ，详情见统计学方法10.4.2节
    def __viterbi(self,text,states,start_p,trans_p,emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y]*emit_p[y].get(text[0],1.0)
            path[y] = [y]

        for t in range(1,len(text)):
            V.append({})
            new_path = {}

            for y in states:
                emitp = emit_p[y].get(text[t],1.0)

                (prob , state) = max([(V[t - 1][y0] * trans_p[y0].get(y, 0) * emitp, y0) \
                                      for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = prob
                new_path[y] = path[state]+[y]
            path = new_path

        if emit_p['M'].get(text[-1],0) > emit_p['S'].get(text[-1],0):
            (prob,state) = max([(V[len(text)-1][y],y) for y in ('E',"M")])
        else :
            (prob,state) = max([(V[len(text)-1][y],y) for y in states])

        return (prob,path[state])

    def cut(self,text):
        if not self.trained:
            print('Error：please pre train or load model parameters')
            return

        prob,pos_list = self.__viterbi(text,self.state_list,self.start_p,self.trans_p,self.emit_p)
        begin_,next_ = 0,0

        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin_ = i
            elif pos == 'E':
                yield text[begin_:i+1]
                next_ = i+1
            elif pos == 'S':
                yield char
                next_ = i+1
        if next_ < len(text):
            yield text[next_:]


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            # 维持行数不变，后面的行接到前面的行后面
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            # 经过droupput处理
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                # 该函数返回一个用于初始化权重的初始化程序 “Xavier” 。
                                # 这个初始化器是用来保持每一层的梯度大小都差不多相同
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                # tf.zeros_initializer()，也可以简写为tf.Zeros()
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            # output的形状为[batch_size,steps,cell_num]
            s = tf.shape(output)
            # reshape的目的是为了跟w做矩阵乘法
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b
            # s[1]=batch_size
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            # crf_log_likelihood作为损失函数
            # inputs：unary potentials,就是每个标签的预测概率值
            # tag_indices，这个就是真实的标签序列了
            # sequence_lengths,一个样本真实的序列长度，为了对齐长度会做些padding，但是可以把真实的长度放到这个参数里
            # transition_params,转移概率，可以没有，没有的话这个函数也会算出来
            # 输出：log_likelihood:标量;transition_params,转移概率，如果输入没输，它就自己算个给返回

            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            # 交叉熵做损失函数
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        # 添加标量统计结果
        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """
        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """
        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            # epoch_num=40
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        """
        :param sess:
        :param sent:
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        # 计算出多少个batch，计算过程：(50658+64-1)//64=792
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        #num_batches=2000
        # 记录开始训练的时间
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 产生每一个batch
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            # sys.stdout 是标准输出文件，write就是往这个文件写数据
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            # step_num=epoch*792+step+1
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                # 训练的最后一个batch保存模型
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        # seq_len_list用来统计每个样本的真实长度
        # word_ids就是seq_list，padding后的样本序列
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            # labels经过padding后，喂给feed_dict
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        # seq_len_list用来统计每个样本的真实长度
        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """
        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        # seq_len_list用来统计每个样本的真实长度
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            # transition_params代表转移概率，由crf_log_likelihood方法计算出
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            # 打包成元素形式为元组的列表[(logit,seq_len),(logit,seq_len),( ,),]
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """
        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch + 1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)
