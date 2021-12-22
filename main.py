import matplotlib.pyplot as plt
import numpy as np
import pandas

df = pandas.read_csv('data_0123.csv', header=None)
print(df)

output = df.iloc[0:125, 5].values
output = np.where(output == 1, -1, 1)
input = df.iloc[0:125, [0, 1, 2, 3]].values

plt.title('2D View', fontsize=16)

plt.scatter(input[:63, 0], input[:63, -1], color='black', marker='o', label='ones')
plt.scatter(input[63:125, 0], input[63:125, 1], color='green', marker='x', label='zeros')
plt.xlabel('sapel length')
plt.ylabel('petal length')
plt.legend(loc='upper left')

plt.show()


class Perceptron(object):
    def _init_(self, ogrenme_orani=0.1, iter_sayisi=10):
        self.ogrenme_orani = ogrenme_orani
        self.iter_sayisi = iter_sayisi

    def learn(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []
        for _ in range(self.iter_sayisi):
            error = 0
            for xi, hedef in zip(X, y):
                variation = self.ogrenme_orani * (hedef - self.tahmin(xi))
                self.w[1:] += variation * xi
                self.w[0] += variation
                error += int(variation != 0.0)
            self.errors.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def tahmin(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


classifier = Perceptron(ogrenme_orani=0.1, iter_sayisi=10)

print(classifier.learn(input, output))

print(classifier.w)

print(classifier.errors)

plt.plot(range(1, len(classifier.errors) + 1), classifier.errors)
plt.xlabel('Test')
plt.ylabel('Yanlış tahmin sayısı')
plt.show()
