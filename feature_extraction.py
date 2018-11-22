import json_lines
import jsonlines

from textblob import TextBlob
import pyrenn  # for rnn

import pandas as pd
import matplotlib.pyplot as mpl

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsRegressor

feature_names = []  # list of all feature names (in order)

training_features = []  # list of extracted features for each social media post in training set
training_classifications = []  # classifications of whether each post is clickbait/not

test_features = []  # list of extracted features for each social media post in training set
test_classifications = []  # classifications of whether each post is clickbait/not
test_ids = []

max_samples = 20000  # of samples to use for training

'''
Definitions:
NNP: Proper noun, singular
DT: Determiner
POS: Possessive ending
IN: Preposition or subordinating conjunction
VBZ: Verb, 3rd person singular present
NNS: Noun, Plural
JJ: Adjective
WRB: Wh-adverb
WDT: Wh-determiner
NN: Noun, singular, or mass
5WIH: What, where, when, which, how
VBD: Verb, past tense
QM: Question Mark
RB: Adverb
PRP: Personal pronoun
RBS: Adverb, superlative
VBN: Verb, past participle
NP: Noun Phrase
VB: Verb, base form
VBP: Verb, non-3rd person singular present
WP: Wh-pronoun
EX: Existential there
'''


def extract_features(file_name, set_type):
    i = 0

    NNP = []
    NP = []
    VB = []
    NN = []
    DT = []
    JJ = []
    IN = []
    VBZ = []
    CD = []
    PRP = []
    RB = []
    POS = []
    WP = []
    TO = []
    VBG = []
    NNS = []
    VBP = []
    WDT = []
    VBD = []
    RBS = []
    VBN = []
    EX = []

    with json_lines.open(file_name) as reader:
        for obj in reader:
            features = []
            post_text = obj['postText'][0]

            blob = TextBlob(post_text)
            for pair in blob.tags:
                tag = pair[1]
                word = pair[0]
                
                if tag == 'NNP':
                    NNP.append(word)
                elif tag == 'NN':
                    NN.append(word)
                elif tag == 'DT':
                    DT.append(word)
                elif tag == 'JJ':
                    JJ.append(word)
                elif tag == 'IN':
                    IN.append(word)
                elif tag == 'VBZ':
                    VBZ.append(word)
                elif tag == 'CD':
                    CD.append(word)
                elif tag == 'PRP':
                    PRP.append(word)
                elif tag == 'RB':
                    RB.append(word)
                elif tag == 'POS':
                    POS.append(word)
                elif tag == 'WP':
                    WP.append(word)
                elif tag == 'TO':
                    TO.append(word)
                elif tag == 'VBG':
                    VBG.append(word)
                elif tag == 'NNS':
                    NNS.append(word)
                elif tag == 'VBP':
                    VBP.append(word)
                elif tag == 'WDT':
                    WDT.append(word)
                elif tag == 'VBD':
                    VBD.append(word)
                elif tag == 'RBS':
                    RBS.append(word)
                elif tag == 'VBN':
                    VBN.append(word)
                elif tag == 'VB':
                    VB.append(word)
                elif tag == 'NP':
                    NP.append(word)
                elif tag == 'EX':
                    EX.append(word)
                else:
                    #print(tag)
                    continue

            """
            # (1) feature – number of NNP in the post
            features.append(len(NNP))

            # (3) feature – number of tokens
            features.append(len(post_text.split()))

            # (4) feature - word length of post text
            features.append(len(post_text))

            # (5) feature – POS 2-gram NNP NNP
            c = 0
            for pair in blob.ngrams(n=2):
                if pair[0] in NNP and pair[1] in NNP:
                    c += 1
            features.append(c)
            """

            # (6) feature - whether the post starts with a number
            if len(post_text) == 0:
                features.append(0)
            elif post_text[0].isdigit():
                features.append(1)
            else:
                features.append(0)

            """
            # (7) feature - average length of words in post
            total_chars = 0
            for word in post_text.split():
                total_chars += len(word)
            if len(post_text) == 0:
                features.append(0)
            else:
                features.append(total_chars / len(post_text.split()))

            # (8) feature — number of IN
            features.append(len(IN)) 
            
            """

            # (9) feature – POS 2-gram NNP VBZ
            c = 0
            for pair in blob.ngrams(n=2):
                if pair[0] in NNP and pair[1] in VBZ:
                    c += 1
            features.append(c)
            
            # (10) feature – POS 2-gram IN NNP
            c = 0
            for pair in blob.ngrams(n=2):
                if pair[0] in IN and pair[1] in NNP:
                    c += 1
            features.append(c)

            """
            
            # (11) feature – length of longest word in the post
            c = 0
            for word in post_text.split():
                if c < len(word):
                    c = len(word)
            features.append(c)

            # (12) feature – number of WRD; wh-adverbs, includes how and which
            wwwwwwhw = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
            c = 0
            for word in post_text.split():
                if word.lower() in wwwwwwhw:
                    c += 1
            features.append(c)

            # (14) feature – number of NN
            features.append(len(NN))
            
            """

            # (16) feature – whether the post starts with the following key words
            wwwwwwhw = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
            if len(post_text) == 0:
                features.append(0)
            elif post_text.split()[0] in wwwwwwhw:
                features.append(1)
            else:
                features.append(0)

            #(17) feature – whether a '?' exists
            if '?' in post_text:
                features.append(1)
            else:
                features.append(0)

            # (19) feature — count POS pattern this/these NN
            if len(post_text) == 0:
                features.append(0)
            elif 'this' in post_text or 'these' in post_text:
                features.append(1)
            else:
                features.append(0)

            """

            # (21) feature — number of PRP
            features.append(len(PRP))

            # (22) feature – number of VBZ
            features.append(len(VBZ))

            # (23) feature – POS 3-gram NNP NNP VBZ
            c = 0
            for triple in blob.ngrams(n=3):
                if triple[0] in NNP and triple[1] in NNP and triple[2] in VBZ:
                    c += 1
            features.append(c)

            # (24) feature – POS 2-gram NN IN
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in NN and triple[1] in IN:
                    c += 1
            features.append(c)

            # (25) feature – POS 3-gram NN IN NNP
            c = 0
            for triple in blob.ngrams(n=3):
                if triple[0] in NN and triple[1] in IN and triple[2] in NNP:
                    c += 1
            features.append(c)

            # (27) feature – POS 2-gram NNP .
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in NNP and triple[1] == '.':
                    c += 1
            features.append(c)

            # (28) feature – POS 2-gram PRP VBP
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in PRP and triple[1] in VBP:
                    c += 1
            features.append(c)

            # (30) feature – Number of WP
            features.append(len(WP))

            # (32) feature – Number of DT
            features.append(len(DT))

            # (33) feature – POS 2-gram NNP IN
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in NNP and triple[1] in IN:
                    c += 1
            features.append(c)

            # (34) feature – POS 3-gram IN NNP NNP
            c = 0
            for triple in blob.ngrams(n=3):
                if triple[0] in IN and triple[1] in NNP and triple[2] in NNP:
                    c += 1
            features.append(c)     
            
            """

            # (35) feature – Number of POS
            features.append(len(POS))

            """

            # (36) feature - POS 2-gram IN NN
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in IN and triple[1] in NN:
                    c += 1
            features.append(c)

            # (38) feature – number of ',' in the post
            c = 0
            for text in post_text.split():
                if ',' in text:
                    c += 1
                    #print(text)
            features.append(c)

            # (39) feature - POS 2-gram NNP NNS
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in NNP and triple[1] in NNS:
                    c += 1
            features.append(c)

            # (40) feature - POS 2-gram IN JJ
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in IN and triple[1] in JJ:
                    c += 1
            features.append(c)

            # (41) feature - POS 2-gram NNP POS
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in NNP and triple[1] in POS:
                    c += 1
            features.append(c)

            # (42) feature – Number of WDT
            features.append(len(WDT))

            # (44) feature - POS 2-gram NN NN
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in NN and triple[1] in NN:
                    c += 1
            features.append(c)

            # (45) feature - POS 2-gram NN NNP
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in NN and triple[1] in NNP:
                    c += 1
            features.append(c)

            # (46) feature - POS 2-gram NNP VBD
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in NNP and triple[1] in VBD:
                    c += 1
            features.append(c)

            # (49) feature — number of RB (adverbs)
            features.append(len(RB))

            # (50) feature – POS 3-gram NNP NNP NNP
            c = 0
            for triple in blob.ngrams(n=3):
                if triple[0] in NNP and triple[1] in NNP and triple[2] in NNP:
                    c += 1
            features.append(c)
            
            # (51) feature – POS 3-gram NNP NNP NN
            c = 0
            for triple in blob.ngrams(n=3):
                if triple[0] in NNP and triple[1] in NNP and triple[2] in NN:
                    c += 1
            features.append(c)
            
            """
            
            # (53) feature — number of RBS
            features.append(len(RBS))

            """
            # (54) feature — number of VBN
            features.append(len(VBN))

            # (55) feature - POS 2-gram VBN IN
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in VBN and triple[1] in IN:
                    c += 1
            features.append(c)

            # (56) feature - whether exists pattern: NUMBER NP VB
            c = 0
            for triple in blob.ngrams(n=2):
                if str.isdigit(triple[0]) and triple[1] in NP and triple[3] in VB:
                    c += 1
            features.append(c)

            # (57) feature - POS 2-gram JJ NNP
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in JJ and triple[1] in NNP:
                    c += 1
            features.append(c)

            # (58) feature – POS 3-gram NNP NN NN
            c = 0
            for triple in blob.ngrams(n=3):
                if triple[0] in NNP and triple[1] in NN and triple[2] in NN:
                    c += 1
            features.append(c)

            # (59) feature - POS 2-gram DT NN
            c = 0
            for pair in blob.ngrams(n=2):
                if triple[0] in DT and triple[1] in NN:
                    c += 1
            features.append(c)

            # (60) feature - whether EX exists (Existential there)
            features.append(len(EX))

            # (own) feature – whether the post is all uppercase
            if post_text.isupper():
                features.append(1)
            else:
                features.append(0)

            # (own) feature – whether the last word ends with a '!' mark; more exclamations, more weight
            c = 0
            index = -1
            if len(post_text) == 0:
                features.append(0)
            else:
                while post_text[index] == '!':
                    c += 1
                    index -= 1
                features.append(c)

            # (own) feature – whether the post ends with more than one consecutive '.'; more periods, more weight
            c = 0
            index = -1
            if len(post_text) == 0:
                features.append(0)
            else:
                while post_text[index] == '.':
                    c += 1
                    index -= 1
                features.append(c)
                
            """

            # (own) feature - whether the post ends with a number
            if len(post_text) == 0:
                features.append(0)
            else:
                if post_text[-1].isdigit():
                    features.append(1)
                else:
                    features.append(0)

            ############ End of features ############
            if set_type == "training":
                training_features.append(features)
            else:
                test_features.append(features)

            i += 1
            if i == max_samples: #set_type == "training" and
                break


def rnn(training_data, test_data):
    net = pyrenn.CreateNN([len(test_features),3,3,1])


    # training data
    P = training_data # input
    Y = 1 # output
    # test data
    Ptest = 1 # input
    Ytest =  1 # output

    # set the number of iterations (epochs) to 100 and the termination error to 1e-5.
    # Training will stop after 100 iterations or when Error <= E_stop.
    net = pyrenn.train_LM(P, Y, net, verbose=True, k_max=100, E_stop=1e-5)
    y = pyrenn.NNOut(P, net)
    ytest = pyrenn.NNOut(test_data, net)


# retrieve classifications for social media posts and insert it into specified list
def extract_classifications(file_path, set_type):
    i = 0
    with json_lines.open(file_path) as reader:
        for obj in reader:
            classification = obj["truthMean"]
            # classification = 1 if obj["truthClass"] == "clickbait" else 0
            if set_type == "training":
                training_classifications.append(classification)
            else:
                test_classifications.append(classification)
                test_ids.append(obj["id"])

            # Only collect features for the number of samples specified (takes too long for all 17,000)
            i += 1
            if set_type == "training" and i == max_samples:
                break


def read_files():
    # extract training set features
    print("Extracting training set features")
    extract_features('training_data/instances.jsonl', "training")
    print("Finished extracting training set features!\n")

    # extract training set classifications
    print("Extracting training set classifications")
    extract_classifications('training_data/truth.jsonl', "training")
    print("Finished extracting training set classifications!\n")

    # extract test set features
    print("Extracting test set features")
    extract_features('test_data/instances.jsonl', "test")
    print("Finished extracting test set features!\n")

    # extract test set classifications
    print("Extracting test set classifications")
    extract_classifications('test_data/truth.jsonl', "test")
    print("Finished extracting test set classifications!\n")


# Put all feature names (in order) in feature_names - used for RFE
# Update this if add more features
def create_feature_names_list():
    feature_names.extend((
        "Number of NNP", "Number of tokens", "Word length of post text", "POS 2-gram NNP NNP",
        "Whether the post starts with a number", "Average length of words in post", "Number of IN", "POS 2-gram NNP VBZ",
        "POS 2-gram IN NNP", "Length of longest word in post text", "Number of WRB", "Number of NN",
        "Whether post starts with 5W1H", "Whether ? exists", "Count POS pattern this/these NN", "Count POS pattern PRP",
        "Number of VBZ", "POS 3-gram NNP NNP VBZ", "POS 2-gram NN IN", "POS-gram NN IN NNP", "POS 2-gram NNP", "POS 2-gram PRP VBP",
        "Number of WP", "Number of DT", "POS 2-gram NNP IN", "POS 3-gram IN NNP NNP", "Number of POS",
        "POS 2-gram IN NN", "Number of ,", "POS 2-gram NNP NNS", "POS 2-gram IN JJ", "POS 2-gram NNP POS",
        "Number of WDT", "POS 2-gram NN NN", "POS 2-gram NN NNP", "POS 2-gram NNP VBD", "Number of RB",
        "POS 3-gram NNP NNP NNP", "POS 3-gram NNP NNP NN", "Number of RBS", "number of VBN", "POS 2-gram VBN IN",
        "Whether exists NUMBER NP VB", "POS 2-gram JJ NNP", "POS e-gram NNP NN NN", "POS 2-gram DT NN", "Whether Exist EX",
        "Whether post is all uppercase", "Whether last word ends with !",
        "Whether post ends with more than one consecutive .", "Whether post ends with a number"
    ))


# Use RFE to find the most important features to use
def rfe_svm(training_set_features, training_set_class, test_set_features, test_set_class):
    max_accuracy = -10000
    best_features = []

    # perform RFE until one feature left
    while len(training_set_features[0]) >= 1:
        # train linear SVM
        clf = LinearSVC(max_iter=100000)
        clf.fit(training_set_features, training_set_class)

        # classify test set
        results = clf.predict(test_set_features)

        # get accuracy
        print("Number of features: ", len(training_set_features[0]))
        print("Features: ", feature_names)
        accuracy = get_accuracy(results, test_set_class)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_features = feature_names.copy()

        # get weights and find minimum
        weights = clf.coef_[0]

        min_index = -1
        min_value = 100000

        for i, weight in enumerate(weights):
            val = abs(weight)
            if val <= min_value:
                min_index = i
                min_value = val

        # remove feature from training set and test set
        for feature_set in training_set_features:
            del feature_set[min_index]

        for feature_set in test_set_features:
            del feature_set[min_index]

        # remove feature from list of feature names
        del feature_names[min_index]

        print("\n")

    # print features that resulted in highest accuracy
    print("Best number of features to use for classification:", len(best_features))
    print(best_features)


# compute SVM based on truth mean
def svm(training_set_features, training_set_class, test_set_features, test_set_class):
    # train SVM
    clf = SVR(gamma="auto")
    clf.fit(training_set_features, training_set_class)

    # classify test set
    results = clf.predict(test_set_features)

    # write predictions to file
    create_predictions_file(results)


def knn(training_set_features, training_set_classifications, test_set_features, test_set_classifications):
    # train the model
    n = 10
    classifier = KNeighborsRegressor(n_neighbors=n, weights='distance')
    classifier.fit(training_set_features, training_set_classifications)

    # predict output
    predictions = classifier.predict(test_set_features)

    # write predictions to file
    create_predictions_file(predictions)


def estimate_to_class(means):
    # 0, 0.33, 0.66, 1
    p1 = 0.33 / 2
    p2 = (0.66 + 0.33) / 2
    p3 = (1 + 0.66) / 2

    for i in range(len(means)):
        val = means[i]
        if p1 < val:
            means[i] = 0
        elif p1 < val < p2:
            means[i] = 0.33
        elif p2 < val < p3:
            means[i] = 0.66
        elif p3 < val:
            means[i] = 1

    return means


def get_truth_means(test_set_features):
    mean = 0
    means = []

    for feature_list in test_set_features:
        for feature in feature_list:
            mean += feature
        mean /= len(feature_list)
        means.append(mean)
        mean = 0

    means = normalize(means)
    means = estimate_to_class(means)
    #print (means)


def normalize(arr):
    max = -99999
    min = 99999

    for m in arr:
        if m > max:
            max = m
        if m < min:
            min = m

    for i in range(len(arr)):
        arr[i] = (arr[i] - min) / (max - min)
    
    return arr


# used to determine accuracy for whether post is clickbait/not clickbait
def get_accuracy(results, actual):
    num_correct = 0
    i = 0
    for result in results:
        if result == actual[i]:
            num_correct += 1
    print("Accuracy is", round(num_correct / len(results) * 100, 4), "%")

    return num_correct / len(results)


# create a .jsonl file containing the predictions - each line includes id and prediction
def create_predictions_file(predictions):
    with jsonlines.open('evaluation/predictions.jsonl', mode='w') as writer:
        i = 0
        for id in test_ids:
            writer.write({"id": id, "clickbaitScore": predictions[i]})
            i += 1
    writer.close()


def main():
    # extract features from training and test set
    create_feature_names_list()
    read_files()

    # classify using linear SVM and perform RFE to determine most important features
    # ['POS 2-gram NNP VBZ', 'POS 2-gram IN NNP', 'Whether ? exists', 'Count POS pattern this/these NN', 'Number of POS', 'POS 2-gram IN JJ', 'Number of RBS']
    # rfe_svm(training_features, training_classifications, test_features, test_classifications)

    #svm(training_features, training_classifications, test_features, test_classifications)

    # classify using K-NN - test on test set (TODO: should also test on training set)
    knn(training_features, training_classifications, test_features, test_classifications)

    # classify using RNN - test on test set
    # rnn(training_features, training_classifications, test_features, test_classifications)
    #create_predictions_file()


if __name__ == '__main__':
    main()

