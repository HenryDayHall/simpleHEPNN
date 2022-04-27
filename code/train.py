# cribed from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch
from torch import nn
from model import NeuralNetwork
import awkward as ak


class Logger:
    def __init__(self, model):
        self.model = model
        self._train_batch = []
        self._train_loss = []
        self._test_batch = []
        self._test_loss = []

    def train(self, batch, loss):
        self._train_batch.append(batch)
        self._train_loss.append(float(loss))

    def test(self, *args):
        loss = args[-1]
        if len(args) == 2:
            batch = args[0]
        else:
            if self._train_batch:
                batch = self._train_batch[-1]
            else:
                batch = 0
        self._test_batch.append(batch)
        self._test_loss.append(float(loss))

    @property
    def last_train_batch(self):
        return max(self._train_batch + [-1])

    def plot(self, ax=None, show=False):
        from matplotlib import pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self._train_batch, self._train_loss, label="training")
        ax.plot(self._test_batch, self._test_loss, label="testing")
        ax.set_xlabel("batch")
        ax.set_ylabel("loss")
        ax.semilogy()
        ax.legend()
        if show: plt.show()


def get_dataloader(labels, weights, inputs, batch_size=50, epoch_length=10000):
    if epoch_length is None: epoch_length = len(weights)
    sampler = torch.utils.data.WeightedRandomSampler(weights, epoch_length)
    inputs, labels = torch.Tensor(inputs).to('cpu', dtype=torch.float), torch.Tensor(ak.to_numpy(labels)).to('cpu', dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader


def epoch(dataloader, model, optimizer, loss_fn, logger):
    inverse_size = 1/len(dataloader.dataset)
    model.train()
    print()
    epoch_start = logger.last_train_batch + 1
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        logger.train(epoch_start + batch, loss)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current*inverse_size:2.1%}]", end='\r')
    print()



def train(test_dataloader, train_dataloader, model, logger, n_epochs=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    test(test_dataloader, model, loss_fn, logger)
    for n in range(n_epochs):
        print(f"Starting epoch {n}")
        epoch(train_dataloader, model, optimizer, loss_fn, logger)
        test(test_dataloader, model, loss_fn, logger)


def test(dataloader, model, loss_fn, logger):
    inputs, targets = dataloader.dataset.tensors
    predicted = model(inputs)
    loss = loss_fn(predicted, targets)
    logger.test(loss)


if __name__ == "__main__":
    import data_readers, preprocess
    parts, data = data_readers.read()
    attribute_names, test_parts, train_parts = preprocess.make_test_train(data)
    test_dataloader, train_dataloader = \
        get_dataloader(*test_parts), get_dataloader(*train_parts)
    model = NeuralNetwork(len(attribute_names))
    logger = Logger(model)
    train(test_dataloader, train_dataloader, model, logger)
    logger.plot(show=True)


