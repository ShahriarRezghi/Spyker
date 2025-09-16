import os

import numpy
import torch
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import spyker
from spyker import sparse


def dataset(root, batch):
    trainx, trainy, testx, testy = spyker.read_mnist(
        root + "/train-images-idx3-ubyte",
        root + "/train-labels-idx1-ubyte",
        root + "/t10k-images-idx3-ubyte",
        root + "/t10k-labels-idx1-ubyte",
    )
    trainx, trainy, testx, testy = spyker.to_torch(trainx, trainy, testx, testy)
    trainx = trainx.div(255).to(torch.float32)
    testx = testx.div(255).to(torch.float32)
    train = DataLoader(TensorDataset(trainx, trainy), batch_size=batch)
    test = DataLoader(TensorDataset(testx, testy), batch_size=batch)
    return train, test


def update(config, mult, limit):
    ratio = config.negative / config.positive
    config.positive = min(config.positive * mult, limit)
    config.negative = config.positive * ratio


def total(network, transform, dataset):
    total_data, total_target = [], []
    for data, target in tqdm(dataset):
        data = network(transform(data))
        total_data.append(data.cpu())
        total_target.append(target)
    return torch.cat(total_data), torch.cat(total_target)


class Transform:
    def __init__(self):
        std = [0.471404521, 1.099943882, 2.042752923]
        self.filters = spyker.LoG(3, std, pad=3)

    def __call__(self, array):
        return sparse.code(spyker.threshold(self.filters(array), 0.01), 15)


class Network:
    def __init__(self):
        self.count1, self.count2, self.thresh1, self.thresh2 = 0, 0, 16, 5
        self.conv1 = spyker.Conv(6, 100, 5, pad=2, mean=0.5, std=0.02)
        self.conv2 = spyker.Conv(100, 200, 3, pad=1, mean=0.5, std=0.02)
        self.conv1.stdpconfig = [spyker.STDPConfig(0.0004, -0.0003)]
        self.conv2.stdpconfig = [spyker.STDPConfig(0.0004, -0.0003)]
        self.wta1 = lambda x: sparse.convwta(x, 3, 5)
        self.wta2 = lambda x: sparse.convwta(x, 1, 8)

    def train1(self, array):
        output = sparse.inhibit(self.conv1(array, self.thresh1))
        self.conv1.stdp(array, self.wta1(output))

        self.count1 += array.shape[0]
        if self.count1 > 2000:
            self.count1 -= 2000
            update(self.conv1.stdpconfig[0], 1.5, 1.5)

    def train2(self, array):
        array = sparse.pool(self.conv1(array, self.thresh1), 2)
        output = sparse.inhibit(self.conv2(array, self.thresh2))
        self.conv2.stdp(array, self.wta2(output))

        self.count2 += array.shape[0]
        if self.count2 > 2000:
            self.count2 -= 2000
            update(self.conv2.stdpconfig[0], 1.5, 1.5)

    def __call__(self, array):
        array = sparse.pool(self.conv1(array, self.thresh1), 2)
        array = sparse.pool(self.conv2(array, self.thresh2), 3)
        return (15 - spyker.to_torch(sparse.gather(array))).flatten(1)

    def save(self, path):
        kernel1 = spyker.to_numpy(self.conv1.kernel)
        kernel2 = spyker.to_numpy(self.conv2.kernel)
        numpy.savez(path, conv1_kernel=kernel1, conv2_kernel=kernel2)

    def load(self, path):
        data = numpy.load(path)
        spyker.to_tensor(data["conv1_kernel"]).to(self.conv1.kernel)
        spyker.to_tensor(data["conv2_kernel"]).to(self.conv2.kernel)


if __name__ == "__main__":
    batch_size = 64
    data_root = "./MNIST/"
    model_path = "rezghi_mnist_sparse.npz"

    network = Network()
    transform = Transform()
    trainset, testset = dataset(data_root, batch_size)

    if not os.path.isfile(model_path):
        print(f"Training first layer:")
        for data, target in tqdm(trainset):
            network.train1(transform(data))

        print(f"Training second layer:")
        for data, target in tqdm(trainset):
            network.train2(transform(data))

        print(f"Saving model to: {model_path}")
        network.save(model_path)

    print(f"Loading model from: {model_path}")
    network.load(model_path)

    print("Computing network outputs")
    trainx, trainy = total(network, transform, trainset)
    testx, testy = total(network, transform, testset)

    print("Running PCA reduction")
    pca = PCA(n_components=200).fit(trainx, trainy)
    trainx, testx = pca.transform(trainx), pca.transform(testx)

    print("Running SVM classification")
    acc = SVC(C=2.4).fit(trainx, trainy).predict(testx)
    acc = (torch.tensor(acc) == testy).sum() / len(testy)
    print(f"Accuracy: {acc*100}")
