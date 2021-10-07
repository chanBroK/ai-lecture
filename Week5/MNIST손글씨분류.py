import pandas as pd
import torch
import torch.nn.functional as F

torch.manual_seed(1)

train_df = pd.read_csv("./MNIST손글씨분류_data/train.csv")
test_df = pd.read_csv("./MNIST손글씨분류_data/test.csv")
submit_df = pd.read_csv("./MNIST손글씨분류_data/sample_submit.csv")

y_train = torch.LongTensor(train_df['label']).cuda()
x_train = torch.FloatTensor(train_df.drop(['label'], axis=1).to_numpy()).cuda()

class_num = 10

w = torch.zeros((784, class_num), requires_grad=True, device="cuda")
b = torch.zeros(1, requires_grad=True, device="cuda")

optimizer = torch.optim.SGD([w, b], lr=1e-5)

epochs = 1000

for epoch in range(epochs + 1):
    # one_hot encoding
    # hypothesis = F.softmax(x_train.matmul(w) + b, dim=1)
    # y_one_hot = torch.zeros(len(y_train), class_num)
    # y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    # cost = (y_one_hot * torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()
    # cross_entropy
    hypothesis = x_train.matmul(w) + b
    cost = F.cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()
        ))

# 학습된 W,b를 통한 클래스 예측
hypothesis = F.softmax(x_train.matmul(w) + b, dim=1)  # or .mm or @
# argmax -> 리스트 중 제일 큰 값의 인덱스로 변환
predict = torch.argmax(hypothesis, dim=1)
# 정확도 계산
correct_prediction = predict.float() == y_train
accuracy = correct_prediction.sum().item() / len(correct_prediction)
print('The model has an accuracy of {:2.2f}% for the training set.'.format(accuracy * 100))

x_test = torch.FloatTensor(test_df.to_numpy()).cuda()

hypothesis = F.softmax(x_test.matmul(w) + b, dim=1)  # or .mm or @
# argmax -> 리스트 중 제일 큰 값의 인덱스로 변환
predict = torch.argmax(hypothesis, dim=1)

submit_df['label'] = predict.cpu().detach().numpy()
print(submit_df.head())
