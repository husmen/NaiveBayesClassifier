""" DOCSTRING """
import os
import re
import io
# Simport math as calc
#from itertools import izip, islice, tee
from random import randint
import sys
import jpype
from snowballstemmer import stemmer

#import nltk
from nltk.util import ngrams

sys.path.append('./turkish-stemmer-python')
from TurkishStemmer import TurkishStemmer

class DataProcessing():
    """ DOCSTRING """

    def __init__(self, labels, location, codec, split):
        """Constructor method to load dataset."""
        self.split = split
        self.codec = codec
        self.labels = labels
        print("Labels: " + str(self.labels))
        print("Location: " + str(location))
        self.files, self.files_training, self.files_testing = self.get_documents(
            location)
        self.words = self.load_documents(location, self.files_training)
        self.words_clean = self.clean_stopwords(self.words)
        self.words_zemberek = self.zemberek_process(self.words_clean)
        #self.words_snowball = self.snowball_process(self.words_clean)
        #self.words_trstemmer = self.trstemmer_process(self.words_clean)

        #self.words_2 = self.text2ngrams(self.words,2)
        self.words_clean_2 = self.text2ngrams(self.words_clean,2)
        self.words_zemberek_2 = self.text2ngrams(self.words_zemberek,2)
        #self.words_snowball_2 = self.text2ngrams(self.words_snowball,2)
        #self.words_trstemmer_2 = self.text2ngrams(self.words_trstemmer,2)

        #self.words_3 = self.text2ngrams(self.words,3)
        self.words_clean_3 = self.text2ngrams(self.words_clean,3)
        self.words_zemberek_3 = self.text2ngrams(self.words_zemberek,3)
        #self.words_snowball_3 = self.text2ngrams(self.words_snowball,3)
        #self.words_trstemmer_3 = self.text2ngrams(self.words_trstemmer,3)
        # sys.getdefaultencoding()

        #print(self.string2ngrams("_".join(["this","is","some","bullshit"]), 3))
        #print(word_grams('one two three four'.split(' ')))
        #N = 3
        #trigrams = izip(*(islice(seq, index, None) for index, seq in enumerate(tee(s, N))))
        #list(trigrams)
        return

    def get_documents(self, location):
        """Method to retrieve dataset documents."""
        labels = self.labels
        files = dict.fromkeys(labels, [])
        files_train = dict.fromkeys(labels, [])
        files_test = dict.fromkeys(labels, [])

        nbr = []
        test_idx = set()
        #files_testing = dict.fromkeys(labels, [])

        for label in labels:
            labelfile = []
            for file in os.listdir(location+label+'/'):
                labelfile.append(file)
            files[label] = labelfile
            nbr.append(len(labelfile))

        while len(test_idx) < int(min(nbr)*self.split):
            test_idx.add(randint(0, 200))

        for label in labels:
            labelfile = []
            for i in test_idx:
                labelfile.append(files[label][i])
            files_test[label] = labelfile
            files_train[label] = list(files[label])
            for i in sorted(test_idx, reverse=True):
                del files_train[label][i]

        return files, files_train, files_test

    def load_documents(self, loc, files):
        """Method to load labeled data from the training set."""
        #words = dict.fromkeys(self.labels, "")
        words = dict.fromkeys(self.labels, "")
        for label in self.labels:
            for file in files[label]:
                #words[label] = ""
                with io.open(loc+label+'/'+file, 'r', encoding=self.codec) as t_f:
                    for line in t_f:
                        words[label] = words[label] + ' ' + line
        # print(words)
        for label in self.labels:
            words[label] = re.findall(r"<?/?\w+>?", words[label].lower())
        # print(words[self.labels[0]])
        return words

    def clean_stopwords(self, words):
        """Method to remove stopwords"""
        words_clean = dict.fromkeys(self.labels, "")
        stopwords = set()
        with io.open('stopwords/stopwords.txt', 'r', encoding="utf-8") as t_f:
            for line in t_f:
                stopwords.add(line.rstrip())
        with io.open('stopwords/stopwords-tr.txt', 'r', encoding="utf-8") as t_f:
            for line in t_f:
                stopwords.add(line.rstrip())
        with io.open('stopwords/turkce-stop-words', 'r', encoding="utf-8") as t_f:
            for line in t_f:
                stopwords.add(line.rstrip())
        #stopwords2 = set(stopwords)
        # print(stopwords)
        # print(len(stopwords))
        for label in self.labels:
            words_clean[label] = [w for w in words[label]
                                  if w.isalpha() and not w in stopwords and len(w) > 3]
        #words = dict.fromkeys(self.labels, "")
        # for label in self.labels:
            #words[label] = re.findall(r"<?/?\w+>?", words[label].lower())
        return words_clean

    def zemberek_process(self, words):
        """Method to extract root"""
        words_zemberek = dict.fromkeys(self.labels, "")
        # start JVM
        jpype.startJVM("/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so",
                       "-Djava.class.path=zemberek-tum-2.0.jar", "-ea")

        # Load TurkiyeTurkcesi Class and create tr_o Object
        tr_c = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
        tr_o = tr_c()
        # load Zemberek Class and create zemberek Object
        zemberek_c = jpype.JClass("net.zemberek.erisim.Zemberek")
        zemberek = zemberek_c(tr_o)

        for label in self.labels:
            for w in words[label]:
                result =zemberek.kelimeCozumle(w)
                if result:
                    tmp = str(result[0]).split()
                    words_zemberek[label] = words_zemberek[label] + ' ' + tmp[3]
        
        for label in self.labels:
            words_zemberek[label] = re.findall(r"<?/?\w+>?", words_zemberek[label].lower())


        return words_zemberek

    def snowball_process(self, words):
        """Method to extract stems"""
        words_snowball = dict.fromkeys(self.labels, "")

        find_stem = stemmer('turkish')
        #print("######")
        for label in self.labels:
            tmp = find_stem.stemWords(words[label])
            words_snowball[label] = tmp
        #for label in self.labels:
            
            #words_snowball[label] = re.findall(r"<?/?\w+>?", words_snowball[label].lower())

        return words_snowball

    def trstemmer_process(self, words):
        """Method to extract stems"""

        words_trstemmer = dict.fromkeys(self.labels, "")
        find_stem2 = TurkishStemmer()
        for label in self.labels:
            tmp = []
            for w in words[label]:
                tmp.append(find_stem2.stem(w))
            words_trstemmer[label] = tmp

        return words_trstemmer

    def text2ngrams(self, data, n):
        """docstring"""
        #print(data)
        words = dict.fromkeys(self.labels, "")
        for label in self.labels:
            tmp = string2ngrams("_".join(data[label]),n)
            #print(tmp[:10])
            words[label] = tmp
        return words
        
    def string2ngrams(self, string, n):
        """doscstring"""
        tmp = [string[i:i+n] for i in range(len(string)-n+1)]
        return tmp

    def word_grams(self, data, n):
        """docstring"""
        words = dict.fromkeys(self.labels, "")
        for label in self.labels:
            s = []
            for ngram in ngrams(data[label], n):
                s.append(' '.join(str(i) for i in ngram))
            words[label] = s
        return words

def string2ngrams(string, n):
    """doscstring"""
    tmp = [string[i:i+n] for i in range(len(string)-n+1)]
    return tmp
