import torch
from torch.optim import SGD

torch.manual_seed(42)

teeth = torch.tensor([[0.9, 0.9]])
mouse = torch.tensor([[0.1, 0.1]])
brick = torch.tensor([[0.5, 0.8]])
bottle = torch.tensor([[0.2, 0.9]])
ball = torch.tensor([[0.2, 0.2]])
hedgehog = torch.tensor([[0.7, 0.1]])

dataset = [
    (teeth, torch.tensor([[0.9]])),
    (mouse, torch.tensor([[0.1]])),
    (brick, torch.tensor([[0.6]])),
    (bottle, torch.tensor([[0.5]])),
    (ball, torch.tensor([[0.2]])),
    (hedgehog, torch.tensor([[0.8]]))
]

w0_sharpness = torch.rand((1,1), requires_grad=True) # изменяем веса для вычисления ошибок 
w1_hardness = torch.rand((1,1), requires_grad=True)

bias = torch.tensor([[0.5]], requires_grad=True)

optimizer = SGD([w0_sharpness, w1_hardness, bias], lr=0.001) # какие переменные мы будем оптимизироватоь

def calc_threat_level(object: torch.Tensor) -> torch.Tensor:
    weights = torch.cat([w0_sharpness, w1_hardness], dim=1)  # Объединяем веса
    return object @ weights.T + bias  # Перемножаем объект с транспонированным весом и добавляем смещение


def loss_function(y_pred, y_actual):
    return torch.nn.functional.mse_loss(y_pred, y_actual)

for item in dataset:
    print('Before learning', w1_hardness, w0_sharpness, bias)
    optimizer.zero_grad()

    actual_threat_level = item[1]
    predicted_threat_level = calc_threat_level(item[0])

    loss = loss_function(predicted_threat_level, actual_threat_level)

    loss.backward()  # Вычисление градиентов

    optimizer.step()  # Обновление параметров

    print('After learning: ', w1_hardness, w0_sharpness, bias)
    print('Difference: ', loss.item())  # Вывод значения потерь



#for epoch in range(20):
    #for item in dataset:
     #   optimizer.zero_grad()
      #  actual_threat_level = item[1]
       # predicted_threat_level = calc_threat_level(item[0])
#
 #       loss = loss_function(predicted_threat_level, actual_threat_level)
  #      loss.backward()
#
 #       optimizer.step()
#
#
 #       print('Loss ', loss)
