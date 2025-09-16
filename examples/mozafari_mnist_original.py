import os

import numpy
import torch
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, TensorDataset

import spyker
from spyker import DoGFilter as F


def dataset(root, device, batch):
    device = torch.device(device.kind)
    trainx, trainy, testx, testy = spyker.read_mnist(
        root + "/train-images-idx3-ubyte",
        root + "/train-labels-idx1-ubyte",
        root + "/t10k-images-idx3-ubyte",
        root + "/t10k-labels-idx1-ubyte",
    )
    trainx, trainy, testx, testy = spyker.to_torch(trainx, trainy, testx, testy)
    trainx = trainx.div(255).to(torch.float32).to(device)
    testx = testx.div(255).to(torch.float32).to(device)
    train = DataLoader(TensorDataset(trainx, trainy), batch_size=batch)
    test = DataLoader(TensorDataset(testx, testy), batch_size=batch)
    return train, test


def update(config, mult, limit):
    ratio = config.negative / config.positive
    config.positive = min(config.positive * mult, limit)
    config.negative = config.positive * ratio


def total(network, transform, dataset):
    total_data, total_target = [], []
    for data, target in dataset:
        data = network(transform(data))
        total_data.append(data.cpu())
        total_target.append(target)
    return torch.cat(total_data), torch.cat(total_target)


def perform(labels, target):
    perf = numpy.zeros(4)
    silent = (labels == -1).sum().item()
    perf[0] = (labels == target).sum().item()
    perf[1] = (labels != target).sum().item() - silent
    perf[2] = silent
    perf[3] = len(target)
    return perf


def test(network, transform, dataset):
    perf = numpy.zeros(4)
    for data, target in dataset:
        perf += perform(network(transform(data)), target)
    return perf


class Transform:
    def __init__(self, device):
        filters = [
            F(3 / 9, 6 / 9),
            F(6 / 9, 3 / 9),
            F(7 / 9, 14 / 9),
            F(14 / 9, 7 / 9),
            F(13 / 9, 26 / 9),
            F(26 / 9, 13 / 9),
        ]
        self.filters = spyker.DoG(3, filters, pad=3, device=device)

    def __call__(self, array):
        return spyker.code(spyker.threshold(self.filters(array), 0.02), 15)


class Network:
    def __init__(self, device):
        self.count1, self.count2, self.thresh1, self.thresh2 = 0, 0, 15, 5
        self.conv1 = spyker.Conv(6, 30, 5, pad=2, mean=0.8, std=0.05, device=device)
        self.conv2 = spyker.Conv(30, 250, 3, pad=1, mean=0.8, std=0.05, device=device)
        self.conv3 = spyker.Conv(250, 200, 5, pad=2, mean=0.8, std=0.05, device=device)

        self.conv1.stdpconfig = [spyker.STDPConfig(0.004, -0.003)]
        self.conv2.stdpconfig = [spyker.STDPConfig(0.004, -0.003)]
        reward = spyker.STDPConfig(0.004, -0.003, False, 0.2, 0.8)
        punish = spyker.STDPConfig(-0.004, 0.0005, False, 0.2, 0.8)
        self.conv3.stdpconfig = [reward, punish]

        self.wta1 = lambda x: spyker.convwta(x, 3, 5)
        self.wta2 = lambda x: spyker.convwta(x, 1, 8)
        self.wta3 = lambda x: spyker.convwta(x, 0, 1)

    def train_layer1(self, array):
        output = spyker.inhibit(spyker.threshold(self.conv1(array), self.thresh1))
        self.conv1.stdp(array, self.wta1(output), spyker.fire(output))

        self.count1 += array.size(0)
        if self.count1 > 500:
            self.count1 -= 500
            update(self.conv1.stdpconfig[0], 2, 0.15)

    def train_layer2(self, array):
        array = spyker.pool(spyker.fire(self.conv1(array), self.thresh1), 2)
        output = spyker.inhibit(spyker.threshold(self.conv2(array), self.thresh2))
        self.conv2.stdp(array, self.wta2(output), spyker.fire(output))

        self.count2 += array.size(0)
        if self.count2 > 500:
            self.count2 -= 500
            update(self.conv2.stdpconfig[0], 2, 0.15)

    def train_layer3(self, array, target):
        array = spyker.pool(spyker.fire(self.conv1(array), self.thresh1), 2)
        array = spyker.pool(spyker.fire(self.conv2(array), self.thresh2), 3)
        output = spyker.infinite(self.conv3(array))
        winners = self.wta3(output)

        labels = torch.zeros(len(winners), dtype=torch.long).fill_(-1)
        for i in range(len(winners)):
            if len(winners[i]) == 1:
                labels[i] = winners[i][0].z // (output.size(2) // 10)
                winners[i][0].c = labels[i] != target[i]

        output = spyker.fire(output)
        self.conv3.stdp(array, winners, output)
        return labels

    def __call__(self, array):
        array = spyker.pool(spyker.fire(self.conv1(array), self.thresh1), 2)
        array = spyker.pool(spyker.fire(self.conv2(array), self.thresh2), 3)
        array = spyker.infinite(self.conv3(array))
        winners = self.wta3(array)

        labels = torch.zeros(len(winners), dtype=torch.long).fill_(-1)
        for i in range(len(winners)):
            if len(winners[i]) == 1:
                labels[i] = winners[i][0].z // (array.size(2) // 10)
        return labels

    def save(self, path):
        kernel1 = spyker.to_numpy(self.conv1.kernel)
        kernel2 = spyker.to_numpy(self.conv2.kernel)
        kernel3 = spyker.to_numpy(self.conv3.kernel)
        numpy.savez(path, conv1_kernel=kernel1, conv2_kernel=kernel2, conv3_kernel=kernel3)

    def load(self, path):
        data = numpy.load(path)
        spyker.to_tensor(data["conv1_kernel"]).to(self.conv1.kernel)
        spyker.to_tensor(data["conv2_kernel"]).to(self.conv2.kernel)
        spyker.to_tensor(data["conv3_kernel"]).to(self.conv3.kernel)


if __name__ == "__main__":
    batch_size = 64
    data_root = "./MNIST/"
    model_path = "mozafari_mnist_original.npz"
    device = spyker.device("cuda" if spyker.cuda_available() else "cpu")

    network = Network(device)
    transform = Transform(device)
    trainset, testset = dataset(data_root, device, batch_size)

    if not os.path.isfile(model_path):
        for i in range(2):
            print(f"Training first layer iteration: {i+1}")
            for data, target in trainset:
                network.train_layer1(transform(data))

        for i in range(4):
            print(f"Training second layer iteration: {i+1}")
            for data, target in trainset:
                network.train_layer2(transform(data))

        train_max, test_max = 0.0, 0.0
        rpos = network.conv3.stdpconfig[0].positive
        rneg = network.conv3.stdpconfig[0].negative
        ppos = network.conv3.stdpconfig[1].positive
        pneg = network.conv3.stdpconfig[1].negative

        for i in range(680):
            print(f"Training third layer iteration: {i+1}")
            perf = numpy.zeros(4)
            perf_batch = numpy.zeros(4)

            for data, target in trainset:
                labels = network.train_layer3(transform(data), target)
                temp = perform(labels, target)
                perf_batch += temp
                perf += temp

                if perf_batch[3] % 1024 == 0:  # Note: BATCH must be in the form 2^N
                    network.conv3.stdpconfig[0].positive = rpos * (perf_batch[1] / perf_batch[3])
                    network.conv3.stdpconfig[0].negative = rneg * (perf_batch[1] / perf_batch[3])
                    network.conv3.stdpconfig[1].positive = ppos * (perf_batch[0] / perf_batch[3])
                    network.conv3.stdpconfig[1].negative = pneg * (perf_batch[0] / perf_batch[3])
                    perf_batch = numpy.zeros(4)

            train_max = max(train_max, perf[0] / perf[3] * 100)
            perf = test(network, transform, testset)
            test_now = perf[0] / perf[3] * 100

            if test_now > test_max:
                print(f"Saving model with accuracy {test_now} to: {model_path}")
                network.save(model_path)
                test_max = test_now

    print(f"Loading model from: {model_path}")
    network.load(model_path)
    perf = test(network, transform, testset)
    print(f"Accuracy: {perf[0]/perf[3]*100}")
