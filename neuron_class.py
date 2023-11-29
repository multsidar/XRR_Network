import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def find_numbers_in_quotes(text):
    """
    :param text: float numbers
    :return: arrays from cells of dataset
    """
    matches = re.findall("[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    numbers = [float(match) for match in matches]
    return numbers


def find_numbers_in_quotes1(text):
    pattern = "\d+"
    matches = re.findall(pattern, text)
    numbers = [float(match) for match in matches]
    return numbers


def process_array_peaks(input_array):
    peaks, _ = find_peaks(input_array)
    result_array = np.zeros_like(input_array)

    for i in range(len(peaks)):
        result_array[peaks[i]] = input_array[peaks[i]]
    return result_array


def regen_data(df):
    """
    :param df: dataset with str in cells
    :return: torch.tensor
    """
    q = df.iloc[:, [0]].values
    r = df.iloc[:, [2]].values
    z = df.iloc[:, [3]].values
    rho = df.iloc[:, [4]].values
    rough = df.iloc[:, [5]].values
    pca = PCA(n_components=1)

    temp = []
    temp1 = []
    temp2 = []
    for i in range(len(q)):
        str_q = str(q[i])
        str_r = str(r[i])
        arr_q = find_numbers_in_quotes(str_q)
        arr_r = find_numbers_in_quotes(str_r)

        arr_r = [float(match/max(arr_r)) for match in arr_r]
        #plt.plot(arr_q,arr_r)
        #plt.yscale('log')
        #plt.savefig('test'+str(i))
        #plt.close()


        if len(arr_q) == len(arr_r):

            arr_grad = np.gradient(arr_r, arr_q )
            arr_grad1 = [float(match/max(arr_grad)) for match in arr_grad]
            arr_peaks = process_array_peaks(arr_r)
            arr_peaks1 = [float(match/max(arr_peaks)) for match in arr_peaks]
            arr_grad_peaks = process_array_peaks(arr_grad1)
            arr_grad_peaks1 = [float(match/max(arr_grad_peaks)) for match in arr_grad_peaks]

            arr_fourier = np.fft.fft(arr_q)
            arr_fourier1 = [float(match/max(arr_fourier)) for match in arr_fourier]
            n = len(arr_q)  # Количество точек
            dt = arr_q[1] - arr_q[0]
            freq = np.fft.fftfreq(n, dt)
            freq1 = [float(match) for match in freq]

            a = []
            for j in range(len(arr_q)):
                a.append([arr_q[j], arr_r[j],arr_fourier1[j],freq1[j]])
            temp.append(a)


    for j in range(len(z)):


        str_z = str(z[j])
        str_rho = str(rho[j])
        str_rough = str(rough[j])

        arr_z = find_numbers_in_quotes(str_z)
        #arr_rho = find_numbers_in_quotes(str_rho)
        #arr_rough = find_numbers_in_quotes(str_rough)
        #a = []

        #a.append(float(len(arr_z)))

        temp2.append(float(len(arr_z))-2)

    b = torch.tensor(temp)
    print(b.shape)
    c = torch.tensor(temp2)
    return b,c


class LayerCounter(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(LayerCounter, self).__init__()

        self.lstm = nn.LSTM(4, 2, 5, batch_first=True, dropout=0.2)
        self.fc_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in
                                        zip([2,512, 1024, 512, 256, 128, 64, 32, 16], [512,1024, 512, 256, 128, 64, 32, 16, 2])])
        self.relu = nn.ReLU6()
        self.fc = nn.Linear(2, 2)
        self.fc_output = nn.Linear(2, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[-1,:]
        output = self.fc(output)
        for layer in self.fc_layers:
            output = self.relu(layer(output))
        output = self.fc_output(output)
        return output


class DynamicOutputLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DynamicOutputLayer, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, lengths):
        packed_input = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output

"""
class multifilm_estimator(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data):

        super(multifilm_estimator, self).__init__()

        self.layer_counter = layer_counter(input_size=2,hidden_size=256,num_layers=10)
        self.rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=0.2)
        self.relu = nn.ReLU6()
        output_size = layer_counter(data)
        self.multilinear = DynamicOutputLayer(input_size=hidden_size, hidden_size=hidden_size, output_size=output_size)
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.relu(output)
        output = self.multilinear(output)
        return output

"""


class Estimator_1d(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Estimator_1d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers,dropout=0.2,batch_first=True,nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size,hidden_size)
        self.bn = nn.BatchNorm1d(num_layers * hidden_size)
        self.fc_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in
                                        zip([hidden_size, 4, 16, 32, 16], [4, 16, 32, 16, output_size])])

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_output = nn.Linear(output_size,output_size)

    def forward(self, x,):
        output, _ = self.rnn(x)
        output = self.relu(output)
        output = self.bn(output)
        output=self.sigmoid(output)
        output = self.fc(output[:,int(output.shape[2])-1])
# linear layers
        for layer in self.fc_layers:
            output = self.relu(layer(output))

        return output


def load_data():
    df = pd.read_csv("train_data_for_counter.csv")
    df = df.sample(frac=1)
    n = 1200
    random_indices = np.random.choice(df.index, n, replace=False)
    df.drop(random_indices, inplace=True)
    train_data , train_labels = regen_data(df)
    #train_data = torch.rand(300,1000,4)
    #train_labels = torch.rand(300)
    return train_data,train_labels




def train_model(model,batch_size,train_data,train_labels,name):
    model.train()
    loss_arr = []
    for epoch in tqdm(range(20)):
        l = []
        batch_size = 1
        for i in tqdm(range(0, len(train_data), batch_size)):

            optimizer.zero_grad()

            batch_x_train = train_data[i,:,:]
            batch_y_train = train_labels[i]


            batch_x_train = torch.tensor(batch_x_train)
            batch_y_train = torch.tensor(batch_y_train).reshape(1)


            output = model.forward(batch_x_train)

            print('--------------------------------')
            print("output:" + str(output)+'\n'
                  + "batch_y_train:" + str(batch_y_train)+'')
            print('--------------------------------')

            loss = criterion(output, batch_y_train)
            loss.backward()
            optimizer.step()

            l.append(loss.item())
            print(loss.item())
            loss_arr.append(sum(l) / len(l))


    torch.save(model.state_dict(), name + '.pth')

    plt.plot(range(len(loss_arr)),loss_arr)
    plt.show()


model = LayerCounter(input_size=4,hidden_size=64,num_layers=5)
criterion = nn.L1Loss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
train_data,train_labels=load_data()
model.to('cuda')
train_data = train_data.to('cuda')
train_labels = train_labels.to('cuda')
batch_size = 1

train_model(model,batch_size,train_data,train_labels,name='layer_counter_10_64')

