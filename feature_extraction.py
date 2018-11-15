import json_lines
from textblob import TextBlob

posts_features = []  # list of extracted features for each social media post

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

def extract_features():
    i = 0

    NNP = []
    NN = []
    DT = []
    JJ = []
    IN = []
    VBZ = []
    CD = []
    PRP = []
    RB = []

    with json_lines.open('training_data/instances.jsonl') as reader:
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

            # (1) feature – number of NNP in the post
            features.append(len(NNP))

            # (3) feature – number of tokens
            features.append(len(post_text.split()))

            # (4) feature - word length of post text
            features.append(len(post_text))

            # (5) feature – POS 2-gram NNP NNP (possessive 2-gram nouns)
            c = 0
            for pair in blob.ngrams(n=2):
                if pair[0] in NNP and pair[1] in NNP:
                    if is_possessive(pair[0]):
                        c += 1
            features.append(c)

            # (6) feature - whether the post starts with a number
            if post_text[0].isdigit():
                features.append(1)
            else:
                features.append(0)

            # (7) feature - average length of words in post
            total_chars = 0
            for word in post_text.split():
                total_chars += len(word)
            features.append(total_chars / len(post_text.split()))

            # (8) feature — number of IN
            features.append(len(IN))

            # (9) feature – POS 2-gram NNP VBZ
            c = 0
            for pair in blob.ngrams(n=2):
                if pair[0] in NNP and pair[1] in VBZ:
                    if is_possessive(pair[0]):
                        c += 1
            features.append(c)

            # (10) feature – POS 2-gram IN NNO
            c = 0
            for pair in blob.ngrams(n=2):
                if pair[0] in IN and pair[1] in NNP:
                    if is_possessive(pair[0]):
                        c += 1
            features.append(c)

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

            # (16) feature – whether the post starts with the following key words
            if post_text.split()[0] in wwwwwwhw:
                features.append(1)
            else:
                features.append(0)

            # (17) feature – whether a '?' exists
            if '?' in post_text:
                features.append(1)
            else:
                features.append(0)

            # (19) feature — count POS pattern this/these NN
            if 'this' in post_text or 'these' in post_text:
                features.append(1)
            else:
                features.append(0)

            # (20) feature — count POS pattern PRP


            # (21) feature — number of PRP
            features.append(len(PRP))

            # (38) feature – number of ',' in the post
            c = 0
            for text in post_text.split():
                if ',' in text:
                    c += 1
                    #print(text)
            features.append(c)

            # (49) feature — number of RB (adverbs)
            features.append(len(RB))

            # (own) feature – whether the post is all uppercase
            if post_text.isupper():
                #print(post_text)
                features.append(1)

            # (own) feature – whether the last word ends with a '!' mark; more exclamations, more weight
            c = 0
            index = -1
            while post_text[index] == '!':
                c += 1
                index -= 1
            features.append(c)

            # (own) feature – whether the post ends with more than one consecutive '.'; more periods, more weight
            c = 0
            index = -1
            while post_text[index] == '.':
                c += 1
                index -= 1
            features.append(c)

            # (own) feature - whether the post ends with a number
            if post_text[-1].isdigit():
                features.append(1)
            else:
                features.append(0)

            ############ End of features ############
            posts_features.append(features)

            # used for testing smaller number of files - delete later
            i += 1
            if i > 100:
                break


def is_possessive(word):
    if '\'' in word:
        if word[-1] == 's':
            if word[-2] == 'e' and word[-3] == '\'':
                return True
            elif word[-2] == '\'':
                return True
            else:
                return False
        elif word[-1] == '\'':
            return True
    else:
        return False

def read_file():
    # extract features into post_features
    extract_features()

    for features in posts_features:
        print(features)


def main():
    read_file()

if __name__ == '__main__':
    main()
