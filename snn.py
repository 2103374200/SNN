from sklearn.metrics import  matthews_corrcoef, confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import  LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score
import data_read_save
from models.akconv import AKConv
import numpy as np
import random
import os
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
import torch.nn.functional as F
from models.GNConv import GNconv
from models.msca import MSCASpatialAttention
from models.resnetse import ResNet18_SE


seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.use_deterministic_algorithms(True)
from torchvision import models



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)  # 设置默认GPU设备为GPU 1


trainfile_path = './data/trainset.txt'
testfile_path = './data/testset.txt'
# 读取数据
seqs=data_read_save.read_sequences(trainfile_path);
test_seqs = data_read_save.read_sequences(testfile_path)
# 将数据转换为DataFrame
df = pd.DataFrame(seqs, columns=['id', 'sequence'])
df=df['sequence'].tolist()
test_df = pd.DataFrame(test_seqs, columns=['id', 'sequence'])
test_df=test_df['sequence'].tolist()

# 前2206个样本的标签为1，后2206个样本的标签为0
labels = torch.cat((torch.ones(2206, dtype=torch.long), torch.zeros(2206, dtype=torch.long)))
# 生成测试集的标签
test_labels = torch.cat((torch.ones(552, dtype=torch.long), torch.zeros(552, dtype=torch.long)))

train_scpsednc = './feature_script/features/SCPseDNC.csv'  # 请将文件路径替换为你的CSV文件路径
test_scpsednc = './feature_script/features/SCPseDNC_test.csv'  # 请将文件路径替换为你的CSV文件路径
train_scpsednc = pd.read_csv(train_scpsednc, header=None,index_col=0)
train_scpsednc =train_scpsednc.iloc[:].values
test_scpsednc = pd.read_csv(test_scpsednc, header=None, index_col=0)
test_scpsednc =test_scpsednc.iloc[:].values

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # euclidean_distance = F.pairwise_distance(output1, output2)
        # loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
        #                                                   2))
        #
        # return loss_contrastive

        cos_distance = D(output1, output2)
        # print("ED",euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 3))

        return loss_contrastive

class NerDataset(torch.utils.data.Dataset):
    def __init__(self, data, sc,labels):
        self.data = data
        self.sc=sc
        self.labels = labels

    def __getitem__(self, idx):
        item1 = self.data[idx]
        item2=self.sc[idx]
        label = self.labels[idx]
        return item1,item2,label

    def __len__(self):
        return len(self.labels)


train_dataset = NerDataset(df,train_scpsednc,  labels)
test_dataset = NerDataset(test_df,test_scpsednc,  test_labels)

def rna_to_token(dna_seq, token_len=1):
    base_map = {'A': 3, 'C': 2, 'G': 1, 'T': 4}
    num_seq = [base_map[base] for base in dna_seq]

    tokens = []
    for i in range(0, len(num_seq), token_len):
        token = num_seq[i:i + token_len]
        token_str = ''.join([str(num) for num in token])
        tokens.append(int(token_str))

    return tokens

class MyModel(nn.Module):
    def __init__(self, vocab_size):
        super(MyModel, self).__init__()

        # Define some hyperparameters
        self.n_layers = n_layers = 1
        self.hidden_dim = hidden_dim = 256

        embedding_dim = 800

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=4, kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=(1, 1))
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=(1, 1))
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            bidirectional=True,
                            batch_first=True
                           )
        self.akconv1= AKConv(inc=1,outc=8,num_param=3)
        # self.akconv2= AKConv(inc=4107,outc=38,num_param=3)
        # self.gnconv1= GNconv(16)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8,batch_first=True)  # 添加自注意力层

        self.pool = nn.MaxPool1d(kernel_size=2)


        # self.dropout1 = nn.Dropout(0.5)
        self.dropout1 = nn.Dropout(0.4)

        # 最后的预测层
        self.relu = nn.ReLU()
        self.fc1 = torch.nn.Linear(512, 2)
        self.fc2 = torch.nn.Linear(442, 2)
        self.fc3 = torch.nn.Linear(64, 2)
        # self.fc4 = torch.nn.Linear(128, 2)
    def forward(self, tokens,scs):
        embeds = self.embedding(tokens)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out, hidden = self.attention(lstm_out,lstm_out,lstm_out)
        lstm_out = self.fc1(lstm_out)

        hidden_states = scs.unsqueeze(1).float()
        conv_out = self.conv1(hidden_states)
        conv_out = self.pool(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = self.pool(conv_out)

        lstm_out = lstm_out.reshape(lstm_out.size(0),-1)


        return lstm_out,conv_out
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim)
                 )
        return hidden
    def trainModel(self, tokens,scs):
        with torch.no_grad():
            lstm_out,conv_out = self.forward(tokens,scs)
        conv_out = conv_out.reshape(conv_out.size(0),-1)
        outputs =torch.cat((lstm_out,conv_out),dim=1)
        outputs = self.dropout1(outputs)
        outputs = self.fc2(outputs)
        outputs = outputs.softmax(dim=1)
        return outputs


def collate_func(batch):
    tokens1,tokens2,scs1,scs2,labels1,labels2,labels =[],[],[],[],[],[],[]
    batch_size = len(batch)

    for i in range(int(batch_size) - 1):
        token1, sc1,label1 = rna_to_token(batch[i][0]) , batch[i][1], batch[i][2]
        token1=torch.tensor(token1)
        sc1=torch.tensor(sc1)
        for j in range(i + 1, int(batch_size)):
            token2,sc2,label2 = rna_to_token(batch[j][0]) , batch[j][1], batch[j][2]
            token2 = torch.tensor(token2)
            sc2 = torch.tensor(sc2)
            labels1.append(label1.unsqueeze(0))
            labels2.append(label2.unsqueeze(0))
            label = (label1 ^ label2)  # 异或, 相同为 0 ,相异为 1
            # tokens1.append(token1)
            tokens1.append(token1.unsqueeze(0))
            tokens2.append(token2.unsqueeze(0))
            scs1.append(sc1.unsqueeze(0))
            scs2.append(sc2.unsqueeze(0))
            labels.append(label.unsqueeze(0))
    token1 = torch.cat(tokens1).to(device)
    token2 = torch.cat(tokens2).to(device)
    sc1=torch.cat(scs1).to(device)
    sc2=torch.cat(scs2).to(device)
    label = torch.cat(labels).to(device)
    label1 = torch.cat(labels1).to(device)
    label2 = torch.cat(labels2).to(device)
    return token1,token2,sc1,sc2,label,label1,label2
def collate_func1(batch):
    tokens,scs,labels = [], [],[]
    for item in batch:
        tokens.append(rna_to_token(item[0]))
        scs.append(item[1])
        labels.append(item[2])
    return torch.tensor(tokens),torch.tensor(scs),torch.tensor(labels)


trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_func)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_func1)

model = MyModel(5)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion_con = ContrastiveLoss()
optimizer = Adam(model.parameters(), lr=0.0001)


def train(model, trainloader, criterion):
    # 训练循环
    num_epochs = 500  # 根据需要修改这个值
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loss_1 = 0
        loss_2 = 0

        for token1,token2,sc1,sc2,label,label1,label2 in tqdm(trainloader):
            # tokens,scs, labels = tokens.to(device),scs.to(device),labels.to(device)
            output1 = model(token1.long(),sc1)[0]
            output2 = model(token2.long(),sc2)[0]
            output3 = model.trainModel(token1.long(),sc1)
            output4 = model.trainModel(token2.long(),sc2)
            loss1 = criterion_con(output1, output2, label)
            loss2 = criterion(output3, label1)
            loss3 = criterion(output4, label2)
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()
            total_loss += loss.item()
            loss_1+=loss1.item()
            loss_2+=loss2.item()
            loss.backward()
            optimizer.step()

            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(trainloader)
        # accuracy = 100 * correct / total
        print(f'第 {epoch + 1} 轮,训练集上的平均损失: {avg_loss}')
        print(f'第 {epoch + 1} 轮,训练集上的loss1: {loss_1/ len(trainloader)}')
        print(f'第 {epoch + 1} 轮,训练集上的loss2: {loss_2/ len(trainloader)}')
        # print(f'第 {epoch + 1} 轮,训练集上的准确率: {accuracy}%')

        save_dir = f'./model/resnet'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        epoch_model_path = f'{save_dir}/model_epoch_{epoch + 1}.pth'


# 测试循环
def test(epoch,model, testloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_prob=[]
    all_labels = []
    with torch.no_grad():
        for tokens,scs, labels in tqdm(testloader):
            tokens,scs, labels = tokens.to(device),scs.to(device),labels.to(device)
            outputs = model.trainModel(tokens.long(),scs)
            prob=outputs[:,1]
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (prob>0.41)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())
    avg_loss = total_loss / len(testloader)
    accuracy = 100 * correct / total

    # 将数组中的每个元素转换为字符串，并用逗号连接
    data_str = ','.join(map(str, all_prob))
    # 打开一个文件来写入数据
    with open('./probs/snn.txt', 'w') as f:
        f.write(data_str)


    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    SEN = tp / (tp + fn)
    SPE = tn / (tn + fp)
    MCC = matthews_corrcoef(all_labels, all_predictions)
    AUROC = roc_auc_score(all_labels, all_prob)
    ACC = accuracy_score(all_labels, all_predictions)
    print(f'测试集平均损失avg_loss: {avg_loss}')
    print(f'测试集准确率accuracy: {ACC}%')
    print(f'敏感度SEN: {SEN}')
    print(f'特异度SPE: {SPE}')
    print(f'MCC: {MCC}')
    print(f'AUROC: {AUROC}')

model.load_state_dict(torch.load('./model/resnet/snn.pth',map_location='cuda:1') )


test(0,model, testloader, criterion)
