import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2

class StudentNet(nn.Module):
    def __init__(self, base=16, width_mult=1):
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        bandwidth = [ base * m for m in multiplier]

        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)
            print(bandwidth[i])

        self.cnn = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU(),
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

#将不重要的参数进行剪枝，然后将原来的model中的参数copy到新的model上，只需要定义new_dim的大小即可，一般大小为原model的百分之多少
def network_slimming(old_model, new_model):
    params = old_model.state_dict()
    new_params = new_model.state_dict()

    selected_idx = []
    for i in range(8):
        importance = params[f'cnn.{i}.1.weight']
        # 抓取的是gamma参数，是在BatchNorm()中，用来给数据做归一化，这个因子决定着网络的重要性
        old_dim = len(importance)
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        ranking = torch.argsort(importance, descending=True)
        selected_idx.append(ranking[:new_dim])
    print("selected_idx = {}".format(selected_idx))

    now_processed = 1
    for (name, p1), (name2, p2) in zip(params.items(), new_params.items()):
        if name.startswith('cnn') and p1.size() != torch.Size([]) and now_processed != len(selected_idx):
            if name.startswith(f'cnn.{now_processed}.3'):
                now_processed += 1

            if name.endswith('3.weight'):
                if len(selected_idx) == now_processed:
                    new_params[name] = p1[:, selected_idx[now_processed - 1]]
                else:
                    new_params[name] = p1[selected_idx[now_processed]][:, selected_idx[now_processed - 1]]
            else:
                new_params[name] = p1[selected_idx[now_processed]]
        else:
            new_params[name] = p1

    new_model.load_state_dict(new_params)
    return new_model

class ImgDataset(Dataset):
    def __init__(self,x,y=None,transform=None):
        self.x=x
        self.y=y
        if y is not None:
            self.y=torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return 200
    def __getitem__(self, item):
        X=self.x[item]
        if self.transform is not None:
            X=self.transform(X)
        if self.y is not None:
            Y= self.y[item]
            return X,Y
        else:
            return X

def readfile(path, label=None):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 224, 224, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(224, 224))
        if label is not None:
            y[i] = int(file.split("_")[0])
    if label is not None:
      return x, y
    else:
      return x

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
    transforms.RandomRotation(15), # 隨機旋轉圖片
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

workspace_dir = './data/food-11'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
# print(train_x.shape)
# print("Size of training data = {}".format(len(train_x)))
# print("Size of training data = {}".format(len(train_y)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
#print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), None)
#print("Size of Testing data = {}".format(len(test_x)))
print("reading over")

batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# print(len(train_loader))
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

net = StudentNet().cuda()
net.load_state_dict(torch.load('student_custom_small.bin'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=1e-3)


def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, labels = batch_data
        inputs = inputs.cuda()
        labels = labels.cuda()

        logits = net(inputs)
        loss = criterion(logits, labels)
        if update:
            loss.backward()
            optimizer.step()

        total_hit += torch.sum(torch.argmax(logits, dim=1) == labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)

    return total_loss / total_num, total_hit / total_num


now_width_mult = 1
for i in range(5):
    now_width_mult *= 0.95
    # 每次剪掉的网络为原来的0.05
    new_net = StudentNet(width_mult=now_width_mult).cuda()
    params = net.state_dict()
    net = network_slimming(net, new_net)
    #现在的net为进行剪枝之后的net
    now_best_acc = 0
    for epoch in range(5):
        net.train()
        train_loss, train_acc = run_epoch(train_loader, update=True)
        net.eval()
        valid_loss, valid_acc = run_epoch(val_loader, update=False)
        #不同的width的进行训练，将表现最好的model存下来
        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(net.state_dict(), f'custom_small_rate_{now_width_mult}.bin')
        print('rate {:6.4f} epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
            now_width_mult,
            epoch, train_loss, train_acc, valid_loss, valid_acc))