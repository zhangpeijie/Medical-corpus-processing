#-*- coding:utf-8 _*-
import codecs
from model  import  HMM
from dataset import  dataset
import os

if __name__ == '__main__':
    #预处理
    dataset=dataset()
    dataset.remove_space("test_cws1.txt","row_cws.txt")

    fin = codecs.open("row_cws.txt" , "r", "utf-8")
    fout = codecs.open("test_cws2.txt", "w", "utf-8")
    train_data = 'train_cws.txt'
    model_file = 'hmm.pkl'
    hmm = HMM()
    #分词
    hmm.train(open(train_data, 'r', encoding='utf-8'), model_file)
    hmm.load_model(model_file)
    for line in fin.readlines():
       fout.writelines(' '.join(hmm.cut(line)))
    fin.close()
    fout.close()
    #分词模型评价
    os.system("python cw_evaluate.py")

    #标注预处理
    dataset.ner_dataset("train_ner.txt", "train_data")
    dataset.ner_dataset("train_ner.txt", "test_data")
    #训练实体标注模型
    os.system("main.py")