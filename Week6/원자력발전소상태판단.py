import pandas as pd
import torch

torch.manual_seed(1)

train_x_df = pd.read_csv("./원자력발전소상태판단_data/pca_train.csv")
test_df = pd.read_csv("./원자력발전소상태판단_data/pca_test.csv")
train_y_df = pd.read_csv("./원자력발전소상태판단_data/train_label.csv")

# Standard Scalar를 이용한 데이터 정규화

# Tensor 등록
x_train = torch.FloatTensor(train_x_df.to_numpy()).cuda()
y_train = torch.FloatTensor(train_y_df['label'].to_numpy()).unsqueeze(1).cuda()
x_test = torch.FloatTensor(test_df.to_numpy()).cuda()
print(x_train.shape)
print(y_train.shape)

# Model 생성
linear1 = torch.nn.Linear(256, 128).cuda()
linear2 = torch.nn.Linear(128, 64).cuda()
linear3 = torch.nn.Linear(64, 32).cuda()
linear4 = torch.nn.Linear(32, 16).cuda()
linear5 = torch.nn.Linear(16, 8).cuda()
linear6 = torch.nn.Linear(8, 1).cuda()
model = torch.nn.Sequential(linear1, linear2, linear3, linear4, linear5, linear6).cuda()

# Adam Optimizer 이용
optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
# loss = torch.nn.BCELoss()

# 학습
epochs = 5000
for epoch in range(epochs + 1):
    # 그래디언트 초기화
    optimizer.zero_grad()

    # Forward 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = ((hypothesis - y_train) ** 2).sum()

    # Backward 계산
    cost.backward()

    # 가중치 갱신
    optimizer.step()

    if epoch % (epochs / 10) == 0:
        print("epochs =", str(epochs), "| epoch =", str(epoch), "| cost =", cost)

# Model 평가

with torch.no_grad():
    predict = model(x_train).int()
    print(predict)
    accuracy = (predict == y_train).float().mean()
    print(accuracy.cpu().detach().numpy())
