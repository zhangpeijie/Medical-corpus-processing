# -*- coding: utf-8 -*-
#4-tags for character tagging: B(Begin),E(End),M(Middle),S(Single)
import  codecs
import re
class dataset(object):

    def remove_space(self,fin1,fout1):
        fin=codecs.open(fin1,"r","utf-8")
        fout=codecs.open(fout1,"w","utf-8")
        for s in fin.readlines():
            fout.writelines(s.replace(' ',''))
        fin.close()
        fout.close()
        return

    def character_tagging(self,input_file, output_file):
        input_data = codecs.open(input_file, 'r', 'utf-8')
        output_data = codecs.open(output_file, 'w', 'utf-8')
        for line in input_data.readlines():
            word_list = line.strip().split()
            for word in word_list:
                if len(word) == 1:
                    output_data.write(word + "\tS\n")
                else:
                    output_data.write(word[0] + "\tB\n")
                    for w in word[1:len(word) - 1]:
                        output_data.write(w + "\tM\n")
                    output_data.write(word[len(word) - 1] + "\tE\n")
            output_data.write("\n")
        input_data.close()
        output_data.close()

    def ner_dataset(self,input_file,output_file):

        input_data = codecs.open(input_file, 'r', 'utf-8')
        output_data = codecs.open(output_file, 'w', 'utf-8')
        for line in input_data.readlines():
            simple=re.compile("[[](.*?)[]]bod{1}",re.S|re.M).findall(line)
            for word in simple:
                word = word.strip().split()
                if len(word) == 1:
                    output_data.writelines(word[0].replace('[','') + "\tB-bod\n")
                else:
                    output_data.writelines(word[-1].replace('[','') + "\tB-bod\n")
            simple = re.compile("[[](.*?)[]]dis{1}", re.S|re.M).findall(line)
            for word in simple:
                if len(word) < 10:
                    word = word.strip().split()
                    if len(word) == 1:
                        output_data.writelines(word[0] + "\tB-dis\n")
                    else:
                        output_data.writelines(word[0] + "\tB-dis\n")
                        for w in range(1, len(word)):
                            output_data.writelines(word[w] + "\tI-dis\n")
            simple = re.compile("[[](.*?)[]]tes{1}", re.S|re.M).findall(line)
            for word in simple:
                if len(word) < 10:
                    word = word.strip().split()
                    if len(word) == 1:
                        output_data.writelines(word[0] + "\tB-tes\n")
                    else:
                        output_data.writelines(word[0] + "\tB-tes\n")
                        for w in range(1, len(word)):
                            output_data.writelines(word[w] + "\tI-tes\n")

            simple = re.compile("[[](.*?)[]]sym{1}", re.S|re.M).findall(line)
            for word in simple:
                if len(word)<10:
                    word = word.strip().split()
                    if len(word) == 1:
                        output_data.writelines(word[0] + "\tB-sym\n")
                    else:
                        output_data.writelines(word[0] + "\tB-sym\n")
                        for w in range(1, len(word)):
                            output_data.writelines(word[w] + "\tI-sym\n")


