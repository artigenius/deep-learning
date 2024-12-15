import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

torch.manual_seed(42)

writer = SummaryWriter()

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=True
)

class MNISTPerceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 200)
        self.linear2 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.softmax(x)

        return x


def loss_fn(y_pred, y_true):
    return torch.nn.functional.cross_entropy(y_pred, y_true)

model = MNISTPerceptron()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

best_accuracy = 0

for epoch in range(epochs):
    model.train()

    error = torch.tensor([0.0])
    diff_global = 0
    for i, (x, y) in enumerate(train_dataloader):
        started = datetime.datetime.now()
        optimizer.zero_grad()
        y_pred = model(x)
        y_one_hot = torch.zeros_like(y_pred)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1.0)
        loss = loss_fn(y_pred, y_one_hot)
        loss.backward()
        optimizer.step()

        finished = datetime.datetime.now()
        diff = finished - started
        print(diff.microseconds)
        #writer.add_scalar('Diff', diff.microseconds)
        diff_global += diff.microseconds

        error = loss + error

    writer.add_scalar('Diff', diff_global / len(train_dataset))
    writer.add_scalar('Train loss', error / len(train_dataset), epoch)

    model.eval()
    correct = 0
    for i, (inputs, labels) in enumerate(test_dataloader):
        output = model(inputs)

        correct += (torch.argmax(inputs) == torch.argmax(output)).float().sum()

    accuracy = correct / len(test_dataset)

    writer.add_scalar('Test accuracy', accuracy, epoch)

    if accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_mnist.pt')
        best_accuracy = correct / len(test_dataset)

