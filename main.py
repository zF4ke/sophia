import json
import random
import tflearn
import tensorflow
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
import nltk
import pickle

stemmer = LancasterStemmer()


def create_model():
    # try:
    #    with open("data.pickle", "rb") as f:
    #        words, labels, training, output = pickle.load(f)
    # except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:

            # splitting a large sample of text into words
            w = nltk.word_tokenize(pattern)

            words.extend(w)
            docs_x.append(w)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # stem the words
    # https://www.nltk.org/howto/stem.html#unit-tests-for-snowball-stemmer
    words = [stemmer.stem(w.lower())
             for w in words if w not in "?" and w not in "," and w not in "."]
    # remove duplicates and sort the words
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("./model/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    return words, labels, training, output


def load_model():
    net = tflearn.input_data(shape=[None, len(training[0])])
    # layers
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    # try:
    #    model.load("model.tflearn")
    # except:
    model.fit(training, output, n_epoch=1000,
              batch_size=8, show_metric=True)
    model.save("./model/model.tflearn")

    return model


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    print("O bot está pronto! (Type quit to stop)")

    while True:
        i = input("You: ")
        if i.lower() == "quit":
            break

        # retorna uma array de probabilidades e escolhe a maior
        results = model.predict([bag_of_words(i, words)])[0]
        results_index = np.argmax(results)

        tag = labels[results_index]

        if results[results_index] > 0.7:
            for int in data["intents"]:
                if int["tag"] == tag:
                    res = int["responses"]
            print(random.choice(res))
        else:
            print(tag)
            print("Eu não entendi, tenta outra vez.")


with open("intents.json") as file:
    data = json.load(file)

words, labels, training, output = create_model()
model = load_model()

chat()
