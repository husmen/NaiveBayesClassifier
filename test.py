""" DOCSTRING """
# coding: utf-8
from texttable import Texttable

from data_processing import DataProcessing
from naivebayesclassifer import NaiveBayesClassifier

# TR_CODECS
# https://docs.python.org/2.4/lib/standard-encodings.html
# 857, IBM857                   cp857
# ibm1026                       cp1026
# windows-1254                  cp1254
# iso-8859-9, latin5, L5        iso8859_9
# macturkish                    mac_turkish

LOCATION = "dataset/"
LABELS = ['ekonomi', 'magazin', 'saglik', 'siyasi', 'spor']
CODEC = "cp1254"

print("\n\n##### PROCESSING DATA #####")
DATASET = DataProcessing(LABELS, LOCATION, CODEC, 0.25)
print("Dataset statistics:")
print("Label\t\tFull dataset\tTraining set\tTesting set")
for label in LABELS:
    print("{}\t\t{}\t\t{}\t\t{}".format(label, len(DATASET.files[label]), len(DATASET.files_training[label]), len(DATASET.files_testing[label])))

print("Training dataset word statistics:")
print("Label\t\tall words\tno stopwords\tzemberek")
for label in LABELS:
    print("{}\t\t{}\t\t{}\t\t{}".format(label, len(DATASET.words[label]), len(DATASET.words_clean[label]),len(DATASET.words_zemberek[label])))


#print(DATASET.words_clean['magazin'][0:50])
#print(DATASET.words_zemberek['magazin'][0:50])
#print(DATASET.words_clean['magazin'][0:10])
#print(DATASET.words_zemberek['magazin'][0:10])


models = []
model_type = [2,3,2,3]
model_labels = ["words_clean_2","words_clean_3","words_zemberek_2","words_zemberek_3"]

print("\n\n##### TRAINING MODELS #####")
print('\n### model with words_clean_2 ###')
models.append(NaiveBayesClassifier(LABELS, DATASET.files_training, DATASET.words_clean_2, CODEC))
#print('\n### model with words_clean_3 ###')
#models.append(NaiveBayesClassifier(LABELS, DATASET.files_training, DATASET.words_clean_3, CODEC))
#print('\n### model with words_zemberek_2 ###')
#models.append(NaiveBayesClassifier(LABELS, DATASET.files_training, DATASET.words_zemberek_2, CODEC))
#print('\n### model with words_zemberek_3 ###')
#models.append(NaiveBayesClassifier(LABELS, DATASET.files_training, DATASET.words_zemberek_3, CODEC))

print("\n\n##### TESTING MODELS #####")
for model in models:
    print("### {} model ###".format(model_labels[models.index(model)]))
    confusion_matrix = []
    for label in LABELS:
        predicted = [0] * 5
        confusion_matrix.append(predicted)
    results_pos = dict.fromkeys(LABELS, 0)
    results_neg = dict.fromkeys(LABELS, 0)
    
    for label in LABELS:
        tmp_correct = 0
        tmp_wrong = 0
        for file in DATASET.files_testing[label]:
            path = LOCATION + label +'/' + file
            #print(path)
            tmp_res = model.classify(path, model_type[models.index(model)])
            #print(tmp_res)
            if tmp_res == label:
                tmp_correct += 1
            else:
                tmp_wrong += 1

            confusion_matrix[LABELS.index(tmp_res)][LABELS.index(label)] += 1

        results_pos[label] = tmp_correct
        results_neg[label] = tmp_wrong

    #print(results_pos)
    #print(results_neg)
    print("# Confusion Matrix #")
    table = []
    table.append([" "] + LABELS)
    for row in confusion_matrix:
        table.append([" "] + row)

    for i in range(5):
        table[i+1][0] = LABELS[i]
    t = Texttable()
    t.add_rows(table)
    print(t.draw())

    print("# Performance Measures #")

    table2 = []
    table2.append([" "] + ["Precision","Recall","F1 Score"])
    for label in LABELS:
        tmp = sum(confusion_matrix[LABELS.index(label)])
        if tmp == 0:
            tmp = 1
        Precision = results_pos[label]/tmp
        #print("Precision for {}: {}".format(label,Precision))

        Recall = results_pos[label]/len(DATASET.files_testing[label])
        #print("Recall for {}: {}".format(label,Recall))
        Fmeasure = 2*(Recall * Precision) / (Recall + Precision)
        #print("Fmeasure for {}: {}".format(label,Fmeasure))
        table2.append([label,Precision,Recall,Fmeasure])

    t2 = Texttable()
    t2.add_rows(table2)
    print(t2.draw())
