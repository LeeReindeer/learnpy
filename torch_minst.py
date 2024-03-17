import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # input image is 28 * 28
        # full-connect
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        # output is 10 probability
        self.fc4 = torch.nn.Linear(64, 10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

    def optimize(self):
        self.optimizer.step()


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("data", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data: DataLoader, net: Net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in test_data:
            outputs = net.forward(x.view(-1, 28 * 28).cuda())
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device, "cuda", torch.cuda.is_available())

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    net.to(device)

    # 1/10
    print("initial accuracy:", evaluate(test_data, net))
    for epoch in range(2):
        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(output, y)
            # print("epoch", epoch, "loss", loss)
            loss.backward()
            net.optimize()
        print("epoch", epoch, "accuracy", evaluate(test_data, net))

    for n, (x, _) in enumerate(test_data):
        if n >= 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28).to(device)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
