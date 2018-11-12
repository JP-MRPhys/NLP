import numpy as np
import pandas as pd
import spacy

nlp = spacy.load('en')


class BatchGenerator:

    def __init__(self, dat, batch_size):
        n = len(dat)
        lst = [i for i in range(n)]
        lst = sorted(lst, key=lambda i: len(dat[i][0].split(' ')))

        self.batch_xs, self.batch_ys, self.reviews = [], [], []

        for i in range(n // batch_size):
            long = len(dat[lst[(i + 1) * batch_size - 1]][0].split(' '))
            batch_x = np.zeros((batch_size, long, 300))
            batch_y = np.zeros((batch_size, 2))
            review = []

            for j in range(batch_size):
                words = dat[lst[i * batch_size + j]][0].split(' ')
                for k in range(len(words)):
                    batch_x[j][k] = nlp(words[k]).vector  # use existing Word2vec model
                for k in range(k, long):
                    batch_x[j][k] = nlp(' ').vector  # padding with ' '

                batch_y[j][dat[lst[i * batch_size + j]][
                    1]] = 1  # represent class as 1-hot vector
                review.append(dat[lst[i * batch_size + j]][0])

            self.batch_xs.append(batch_x)
            self.batch_ys.append(batch_y)
            self.reviews.append(review)

    def get(self, batch_id):
        return self.batch_xs[batch_id], self.batch_ys[batch_id], self.reviews[batch_id]


if __name__ == '__main__':
    train_csv = pd.read_csv('/datasets/train/')
