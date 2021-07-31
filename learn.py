import argparse
import joblib
import copy
import numpy as np
from torchvision import datasets, transforms
import joblib
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn, autograd
import random
from sklearn import metrics
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime

class SNet(nn.Module):
	# define a simple neural net
    def __init__(self, n_features):
        super(SNet, self).__init__()
        self.fc1 =  nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        return torch.sigmoid(self.fc3(x))

class MyDataset(Dataset):
	# utility to load the dataset
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path).fillna(0).values
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        flow = torch.FloatTensor(self.data[index, :-2])
        label = torch.LongTensor(np.array(self.data[index, -1]))
        #category = self.data.iloc[0, -2]
        if self.transform is not None:
            flow = self.transform(flow)
        return flow, label

class DatasetSplit(Dataset):
	# utility to get only needed samples from dataset
	# used to split data between nodes
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
	# the training loop used at the nodes
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        #self.loss_func = nn.BCELoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = getattr(torch.optim, args.optimizer)(net.parameters(), lr=args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (flows, labels) in enumerate(tqdm(self.ldr_train, position=0, leave=True)):
                flows, labels = flows.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(flows)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(flows), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def FedAvg(w):
	# federated averaging algorithm. Simply average the weights
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def test_flow(net_g, datatest, args):
	# evaluation loop for test set
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def process_data(args):
	# construct cleaned data from raw CSVs
    data = pd.read_csv(args.raw_dataset)
    X = data.drop(['Label', 'Attack'], 1)
    y = data['Label']
    y_cat = data['Attack']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y_cat)
    X_train.reset_index(inplace=True)
    unique_ips = list(X_train['sourceIPAddress'].value_counts().index)
    dict_users = {}
    counter = 0
    for ip in unique_ips:
        dict_users[counter] = list(X_train.loc[X_train['sourceIPAddress'] == ip].index)
        counter += 1

    with open(args.network_users, 'wb') as f:
        pickle.dump(dict_users, f, pickle.HIGHEST_PROTOCOL)

    X_train['Label'] = y_train
    X_train.drop(['flowStartMilliseconds', 'sourceIPAddress', 'destinationIPAddress'], 1, inplace=True)
    X_test['Label'] = y_test
    X_test.drop(['flowStartMilliseconds', 'sourceIPAddress', 'destinationIPAddress'], 1, inplace=True)
    X_train.to_csv(f"training_set_{args.dataset}.csv", header=True, index=False)
    X_test.to_csv(f"testing_set_{args.dataset}.csv", header=True, index=False)
    if args.verbose:
        print(">> Datasets generated successfully!")

def fed_training(dataset_train, args, dict_users):
	# federated training loop
    net_glob = SNet(dataset_train[0][0].shape[0]).to(args.device)
    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()

    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print(">> Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        print('>> Epoch :', iter)
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        start = args.num_users_first if args.num_users_first else 0
        end = args.num_users_last if args.num_users_last else len(dict_users)
        idxs_users = range(start, end)
        for idx in idxs_users:
            print('>> >> Node :', idx)
            net_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = net_local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
    return net_glob

def normal_training(dataset_train, args):
	# normal training loop
    net_glob = SNet(dataset_train[0][0].shape[0]).to(args.device)
    print(net_glob)
    optimizer = getattr(torch.optim, args.optimizer)(net_glob.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True)

    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)
    return net_glob

def generate_indicies(train_size, users):
	# iid spliting utility
    idss = list(range(train_size))
    random.shuffle(idss)
    n = int(len(idss)/users)
    d_u = {j:idss[i:i + n] for j,i in enumerate(range(0, len(idss), n))}
    return d_u

def main(args):
	# this is where the magic happens
    if args.extract_data:
        process_data(args)

    #trans = transforms.Compose([transforms.ToTensor()])
    dataset_train = MyDataset(f"training_set_{args.dataset}.csv")
    dataset_test = MyDataset(f"testing_set_{args.dataset}.csv")

    if not args.n_users:
        with open(args.network_users, 'rb') as f:
            dict_users = pickle.load(f)
    else:
        dict_users = generate_indicies(len(dataset_train), args.n_users)

    time_now = datetime.now().strftime("%d-%m-%Y-%H-%M")

    if args.normal_training:
        normal_model = normal_training(dataset_train, args)
        joblib.dump(normal_model, f'normal_model_{time_now}.gz')
    elif args.fed_training:
        federated_model = fed_training(dataset_train, args, dict_users)
        joblib.dump(federated_model, f'federated_model_{time_now}.gz')
    else:
        pass

    if args.test:
        if args.normal_training:
            print(">> Normal testing")
            accuracy_normal, test_loss_normal = test_flow(normal_model, dataset_test, args)
        elif args.fed_training:
            print(">> Federated testing")
            accuracy_federated, test_loss_federated = test_flow(federated_model, dataset_test, args)
        else:
            pass
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=4, help="rounds of training")
    parser.add_argument('--num_users_first', type=int, help="number of users: K")
    parser.add_argument('--num_users_last', type=int, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=32, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', default="Adam", type=str)
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--raw_dataset', type=str, default='../../CAIA_15.csv', help="name of dataset")
    parser.add_argument('--dataset', type=str, default='unsw2015', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--extract_data', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--network_users', type=str, default='network_users.pkl', help='model name')
    parser.add_argument('--n_users', type=int, help="GPU ID, -1 for CPU")
    parser.add_argument('--normal_training', action='store_true', help='aggregation over all clients')
    parser.add_argument('--fed_training', action='store_true', help='aggregation over all clients')
    parser.add_argument('--test', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    #joblib.dump(args, 'args.gz')

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)
    main(args)