from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch


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
        y = survived

        return x, y

titanic_dataset = TitanicDataset()
dataloader = DataLoader(dataset=titanic_dataset, batch_size=1, shuffle=True)


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
        x = self.sigmoid(x)

        return x

model = SurvivalPredictionModel(3, 20, 1)

def loss_function(y_pred, y_actual):
    return torch.nn.functional.mse_loss(y_pred, y_actual)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(40):
    error = torch.tensor([0.0])
    for x, y in dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()

        error = loss + error

    print(error/len(titanic_dataset))

torch.save(model.state_dict(), 'checkpoint.pt')

model.eval()

model.load_state_dict(torch.load('checkpoint.pt'))
x = torch.tensor(titanic_dataset[40][0])
pred = model(x)
pred
