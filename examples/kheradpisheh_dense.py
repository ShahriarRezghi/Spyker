import os, spyker, torch, numpy
from sklearn.svm import LinearSVC
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def dataset(root, device, batch):
    device = torch.device(device.kind)
    trainx, trainy, testx, testy = spyker.read_mnist(
        root+'/train-images-idx3-ubyte', root+'/train-labels-idx1-ubyte',
        root+ '/t10k-images-idx3-ubyte', root+ '/t10k-labels-idx1-ubyte')
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
    for data, target in tqdm(dataset):
        data = network(transform(data))
        total_data.append(data.cpu())
        total_target.append(target)
    return torch.cat(total_data), torch.cat(total_target)


class Transform:
    def __init__(self, device):
        filters = [spyker.DoGFilter(1, 2), spyker.DoGFilter(2, 1)]
        self.filters = spyker.DoG(3, filters, pad=3, device=device)

    def __call__(self, array):
        return spyker.code(spyker.threshold(self.filters(array), .01), 15)


class Network:
    def __init__(self, device):
        self.count, self.thresh1, self.thresh2 = 0, 10, 1
        self.conv1 = spyker.Conv(2, 32, 5, pad=2, mean=.8, std=.05, device=device)
        self.conv2 = spyker.Conv(32, 150, 2, pad=1, mean=.8, std=.05, device=device)
        self.conv1.stdpconfig = [spyker.STDPConfig(.004, -.003)]
        self.conv2.stdpconfig = [spyker.STDPConfig(.004, -.003)]
        self.wta1 = lambda x: spyker.convwta(x, 2, 5)
        self.wta2 = lambda x: spyker.convwta(x, 1, 8)

    def train1(self, array):
        output = spyker.inhibit(spyker.threshold(self.conv1(array), self.thresh1))
        self.conv1.stdp(array, self.wta1(output), spyker.fire(output))

        self.count += array.size(0)
        if self.count > 500:
            self.count -= 500
            update(self.conv1.stdpconfig[0], 2, .15)

    def train2(self, array):
        array = spyker.pool(spyker.fire(self.conv1(array), self.thresh1), 2, pad=1)
        output = spyker.inhibit(spyker.threshold(self.conv2(spyker.inhibit(array)), self.thresh2))
        self.conv2.stdp(array, self.wta2(output), spyker.fire(output))

    def __call__(self, array):
        array = spyker.pool(spyker.fire(self.conv1(array), self.thresh1), 2, pad=1)
        array = spyker.pool(spyker.fire(self.conv2(array), self.thresh2), 2, pad=1)
        return array.amax(dim=1).flatten(1)

    def save(self, path):
        kernel1 = spyker.to_numpy(self.conv1.kernel)
        kernel2 = spyker.to_numpy(self.conv2.kernel)
        numpy.savez(path, conv1_kernel=kernel1, conv2_kernel=kernel2)

    def load(self, path):
        data = numpy.load(path)
        spyker.to_tensor(data['conv1_kernel']).to(self.conv1.kernel)
        spyker.to_tensor(data['conv2_kernel']).to(self.conv2.kernel)


if __name__ == '__main__':
    batch_size = 64
    data_root = './MNIST/'
    model_path = 'model.npz'
    device = spyker.device('cuda' if spyker.cuda_available() else 'cpu')

    network = Network(device)
    transform = Transform(device)
    trainset, testset = dataset(data_root, device, batch_size)

    if not os.path.isfile(model_path):
        for i in range(2):
            print(f'Training first layer iteration: {i+1}')
            for data, target in tqdm(trainset):
                network.train1(transform(data))

        for i in range(20):
            print(f'Training second layer iteration: {i+1}')
            for data, target in tqdm(trainset):
                network.train2(transform(data))

        print(f'Saving model to: {model_path}')
        network.save(model_path)

    print(f'Loading model from: {model_path}')
    network.load(model_path)

    print('Computing network outputs')
    trainx, trainy = total(network, transform, trainset)
    testx, testy = total(network, transform, testset)

    print('Running SVM classification')
    acc = LinearSVC(C=2.4).fit(trainx, trainy).predict(testx)
    acc = (torch.tensor(acc) == testy).sum() / len(testy)
    print(f'Accuracy: {acc*100}')
