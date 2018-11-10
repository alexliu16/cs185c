import json_lines
from textblob import TextBlob
import nltk
from nltk.tag import pos_tag

posts_features = []  # list of extracted features for each social media post


def extract_features():
    i = 0
    with json_lines.open('training_data/instances.jsonl') as reader:
        for obj in reader:
            features = []
            post_text = obj['postText'][0]

            blob = TextBlob(post_text)

            # feature – number of tokens
            # (1) feature - number of proper pronouns
            tagged = pos_tag(post_text.split())
            propernouns = [word for word, pos in tagged if pos == 'NNP']
            features.append(len(propernouns))

            # (3) feature – number of tokens
            features.append(len(post_text.split()))

            # (4) feature - word length of post text
            features.append(len(post_text))

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

            # feature - whether the post ends with a number
            if post_text[-1].isdigit():
                features.append(1)
            else:
                features.append(0)

            # (16) feature – whether the post contains the following key words
            wwwwwwhw = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
            c = 0
            for text in post_text.split():
                if text.lower() in wwwwwwhw:
                     c += 1
                    # print(post_text)
            features.append(c)

            # (17) feature – whether the last word ends with a '?' mark
            if post_text[-1] == '?':
                features.append(1)
            else:
                features.append(0)

            # feature – whether the last word ends with a '!' mark
            if post_text[-1] == '!':
                features.append(1)
            else:
                features.append(0)

            # feature – whether the post contains the following key words
            wwwwwwhw = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
            if post_text.split()[0].lower() in wwwwwwhw:
                features.append(1)
                #print(post_text.split()[0])
            else:
                features.append(0)
                #print(post_text.split()[0])

            # feature – whether the post contains the following key words
            c = 0
            for text in post_text.split():
                if text.lower() in wwwwwwhw:
                    c += 1
                    #print(post_text)
            features.append(c)

            # feature – number of ',' in the post
            c = 0
            for text in post_text.split():
                if ',' in text:
                    c += 1
                    #print(text)
            features.append(c)

            # feature – whether the post is all uppercase
            if post_text.isupper():
                #print(post_text)
                features.append(1)

            # feature – number of noun phrases in the post
            c = 0
            for phrase in blob.noun_phrases:
                c += 1
            features.append(c)

            posts_features.append(features)

            # used for testing smaller number of files - delete later
            i += 1
            if i > 100:
                break


def read_file():
    # extract features into post_features
    extract_features()

    for features in posts_features:
        print(features)


def main():
    read_file()


if __name__ == '__main__':
    main()
