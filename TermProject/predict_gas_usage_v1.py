import pandas as pd
import numpy as np
import torch

# set seed
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)
if device == "cuda":
    torch.cuda.manual_seed(1)

# load data
train_df = pd.read_csv("./data/train_data.csv")
test_df = pd.read_csv("./data/test_data.csv")
submission_df = pd.read_csv("./data/submission.csv")

# preprocessing
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
area_name = np.array(pd.concat([train_df["area_name"], test_df["area_name"]], axis=0))
le.fit(area_name)
train_df["area_name"] = le.transform(train_df["area_name"])
test_df["area_name"] = le.transform(test_df["area_name"])

# set data in torch
train_x = np.array(train_df.drop(["gas_usage", "year"], axis=1))
test_x = np.array(test_df.drop(["year"], axis=1))
train_y = np.array(train_df["gas_usage"])
val_y = np.array(submission_df["gas_usage"])

train_x = torch.Tensor(train_x).to(device)
test_x = torch.Tensor(test_x).to(device)
train_y = torch.Tensor(train_y).to(device)
val_y = torch.Tensor(val_y).to(device)

print("data shape :", train_x.shape, train_y.shape, val_y.shape)

# define and init layer (Use NN)
layer1 = torch.nn.Linear(2, 16).to(device)
layer2 = torch.nn.Linear(16, 1).to(device)
relu = torch.nn.ReLU()

torch.nn.init.xavier_normal_(layer1.weight)
torch.nn.init.xavier_normal_(layer2.weight)

# define model
model = torch.nn.Sequential(layer1, relu, layer2).to(device)

# set learning param
# epochs = 1000
# lr = 1e-3
# 일때 val_cost = 70.99
epochs = 2000
lr = 1e-4
loss = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

# learning
for epoch in range(epochs + 1):
    output = model(train_x)
    cost = loss(output, train_y.unsqueeze(1))
    # val_cost = loss(model(test_x), val_y.unsqueeze(1))
    optim.zero_grad()
    cost.backward()
    optim.step()
    if epoch % (epochs / 10) == 0:
        print(epoch, cost.item())

with torch.no_grad():
    predict = model(test_x)
    cost = loss(predict, val_y.unsqueeze(1))
    print(cost.item())

# kaggle과 local에서 계산이 다르다
