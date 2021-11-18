import numpy as np
import pandas as pd
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


# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# area_name = np.array(pd.concat([train_df["area_name"], test_df["area_name"]], axis=0))
# le.fit(area_name)
# train_df["area_name"] = le.transform(train_df["area_name"])
# test_df["area_name"] = le.transform(test_df["area_name"])

# 행정동별 가중치(인구수, 가스설비 노후화 등)를 area_name을 LabelEncoding한 컬럼 한개의 가중치로 나타낼 수 없다고 생각하였다.
# 각 행정동 별로 가중치 값을 갖기 위해서는 행정동을 각 컬럼으로 구성하는 것이 맞다고 생각하였다.
# 따라서, one hot encoding을 사용하여 각 행정동을 컬럼으로 구성하고자 한다.
train_df = pd.concat([pd.get_dummies(train_df["area_name"], prefix="area_name"), train_df], axis=1)
test_df = pd.concat([pd.get_dummies(test_df["area_name"], prefix="area_name"), test_df], axis=1)
train_df = train_df.drop(["area_name"], axis=1)
test_df = test_df.drop(["area_name"], axis=1)

# set data in torch
train_x = np.array(train_df.drop(["gas_usage", "year"], axis=1))
test_x = np.array(test_df.drop(["year"], axis=1))
train_y = np.array(train_df["gas_usage"])

train_x = torch.Tensor(train_x).to(device)
test_x = torch.Tensor(test_x).to(device)
train_y = torch.Tensor(train_y).to(device)

print("data shape :", train_x.shape, train_y.shape)

# define and init layer (Use DNN)
layer1 = torch.nn.Linear(438, 512)
layer2 = torch.nn.Linear(512, 256)
layer3 = torch.nn.Linear(256, 64)
layer4 = torch.nn.Linear(64, 1)
relu = torch.nn.ReLU()

torch.nn.init.xavier_normal_(layer1.weight)
torch.nn.init.xavier_normal_(layer2.weight)
torch.nn.init.xavier_normal_(layer3.weight)
torch.nn.init.xavier_normal_(layer4.weight)

# define model
model = torch.nn.Sequential(layer1, relu, layer2, relu, layer3, relu, layer4).to(device)

# set learning param
epochs = 1000
lr = 1e-3
loss = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

# learning
for epoch in range(epochs + 1):
    output = model(train_x)
    cost = loss(output, train_y.unsqueeze(1))

    optim.zero_grad()
    cost.backward()
    optim.step()
    if epoch % (epochs / 10) == 0:
        print(epoch, cost.item())

# submission
with torch.no_grad():
    predict = model(test_x)
    predict = predict.cpu().detach()
    submission_df['gas_usage'] = predict
print(submission_df)
submission_df.to_csv("submission.csv", index=False)

# kaggle과 local에서 계산이 다르다
