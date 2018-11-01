import json_lines

posts_features = []  # list of extracted features for each social media post


def read_file():
    i = 0
    with json_lines.open('training_data/instances.jsonl') as reader:
        for obj in reader:
            features = []
            post_text = obj['postText'][0]

            # feature - word length of post text
            features.append(len(post_text.split()))

            # feature - whether the post starts with a number
            if post_text[0].isdigit():
                features.append(1)
            else:
                features.append(0)

            posts_features.append(features)

            # used for testing smaller number of files - delete later
            i += 1
            if i > 100:
                break

    for features in posts_features:
        print(features)


def main():
    read_file()


if __name__ == '__main__':
    main()
