import pickle
import pandas as pd
import pymorphy2
import multiprocessing
import math

train = pd.read_csv('reviews/train.csv',
                    header=None,
                    names=['Review', 'Class'])
test = pd.read_csv('reviews/test.csv',
                   header=None,
                   names=['Review', 'Class'])

# Выделяем данные для обучения
reviews = train['Review']
testreviews = test['Review']


def normalize(reve):
    morph = pymorphy2.MorphAnalyzer()
    reviews2 = []
    for line in reve:
        lst = line.split()
        words = ""
        for word in lst:
            p = morph.parse(word)[0]
            words += p.normal_form + " "
        reviews2.append(words)
    return reviews2


if __name__ == '__main__':
    length = int(math.ceil(reviews.count() / 15))
    length2 = int(math.ceil(testreviews.count() / 15))
    rev = []
    rev2 = []
    test = []
    train = []
    for i in range(14):
        a = reviews[length * i: length * (i + 1)]
        b = testreviews[length2 * i: length2 * (i + 1)]
        rev.append(a)
        rev2.append(b)
    rev.append(reviews[length * 14: len(reviews)])
    rev2.append(testreviews[length2 * 14: len(testreviews)])
    pool = multiprocessing.Pool(processes=15)
    answer = pool.map(normalize, rev)
    for i in range(len(answer)):
        for j in range(len(answer[i])):
            train.append(answer[i][j])
    reviews = pd.Series(train)
    answer = pool.map(normalize, rev2)
    for i in range(len(answer)):
        for j in range(len(answer[i])):
            test.append(answer[i][j])
    testreviews = pd.Series(test)
    reviews.to_csv("trainnorm.csv", index=False)
    testreviews.to_csv("testnorm.csv", index=False)
