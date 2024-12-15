from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(42)
writer = SummaryWriter()


class TitanicDataset(Dataset):
    def __init__(self):
        self.titatic_dataframe = pd.read_csv('titanic.csv')
        self.titatic_dataframe = self.titatic_dataframe.dropna()

    def __len__(self):
        return self.titatic_dataframe.shape[0]

    def __getitem__(self, idx):
        row = self.titatic_dataframe.iloc[idx]
        survived = torch.tensor(row['Survived'], dtype=torch.float)
        fare = torch.tensor(row['Fare'])
        age = torch.tensor(row['Age'])
        sex = torch.tensor([0.]) if row['Sex'] == 'male' else torch.tensor([1.])
        x = torch.tensor([fare, age, sex], dtype=torch.float)

        y_survived = torch.tensor([1.0, 0.0])
        y_not_survived = torch.tensor([0.0, 1.0])

        y = y_survived if survived else y_not_survived

        return x, y

titanic_dataset = TitanicDataset()
train_dataset, test_dataset = torch.utils.data.random_split(titanic_dataset, [len(titanic_dataset) - 40, 40])

train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

class SurvivalPredictionModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, out_size):
        super().__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, 20)
        self.out = torch.nn.Linear(20, out_size)
        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.nn.functional.sigmoid

    def forward(self, input):
        x = self.hidden(input)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.out(x)
        x = torch.nn.functional.softmax(x)

        return x

model = SurvivalPredictionModel(3, 30, 2)

def loss_function(y_pred, y_actual):
    return torch.nn.functional.cross_entropy(y_pred, y_actual)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

best_accuracy = 0

for epoch in range(40):

    model.train()

    error = torch.tensor([0.0])
    for x, y in train_dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()

        error = loss + error

    writer.add_scalar('Train loss', error/len(titanic_dataset), epoch)

    model.eval()

    correct = 0
    for x, y in test_dataloader:
        y_pred = model(x)

        correct += (torch.argmax(y_pred) == torch.argmax(y)).sum()

    if correct / len(test_dataset) > best_accuracy:
        torch.save(model.state_dict(), 'best.pt')
        best_accuracy = correct / len(test_dataset)

    writer.add_scalar('Test acuracy', correct / len(test_dataset), epoch)
