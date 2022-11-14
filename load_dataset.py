import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BtpDataset(Dataset):
    """Btp time series dataset."""
    def __init__(self, csv_file, normalize=True):
        """
        Args:
            csv_file (string): path to csv file
            normalize (bool): whether to normalize the data in [-1,1]
        """
        df = pd.read_csv(csv_file, sep=";")
        df['Timestamp'] = pd.to_datetime(df["data_column"].map(str) + " " + df["orario_column"], dayfirst=True)
        df = df.drop(['data_column', 'orario_column'], axis=1).set_index("Timestamp")
        btp_price = df.BTP_Price
        data = torch.from_numpy(np.expand_dims(np.array([group[1] for group in btp_price.groupby(df.index.date)]), -1)).float()
        self.data = self.normalize(data) if normalize else data
        self.seq_len = data.size(1)
        
        #Estimates distribution parameters of deltas (Gaussian) from normalized data
        original_deltas = data[:, -1] - data[:, 0]
        self.original_deltas = original_deltas
        self.or_delta_max, self.or_delta_min = original_deltas.max(), original_deltas.min() 
        deltas = self.data[:, -1] - self.data[:, 0]
        self.deltas = deltas
        self.delta_mean, self.delta_std = deltas.mean(), deltas.std()
        self.delta_max, self.delta_min = deltas.max(), deltas.min()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min())/(x.max() - x.min()) - 1)
    
    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x*self.max - x*self.min + self.max + self.min)
    
    def sample_deltas(self, number):
        """Sample a vector of (number) deltas from the fitted Gaussian"""
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std
    
    def normalize_deltas(self, x):
        return ((self.delta_max - self.delta_min) * (x - self.or_delta_min)/(self.or_delta_max - self.or_delta_min) + self.delta_min)
    

class IntelDataset(Dataset):
    """Btp time series dataset."""
    def __init__(self, csv_file, normalize=True):
        """
        Args:
            csv_file (string): path to csv file
            normalize (bool): whether to normalize the data in [-1,1]
        """
        df = pd.read_csv(csv_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index("Timestamp")
        df = df[['Temperature', 'Humidity', 'Light', 'Voltage', 'mote_id']]
        df_ = df[['Temperature', 'Humidity', 'Light', 'Voltage']]
        df_ = self.normalize(df_) if normalize else data
        df_ = pd.DataFrame(df_, columns = ['Temperature', 'Humidity', 'Light', 'Voltage'])
        df_['Timestamp'] = df.index
        df_ = df_.set_index("Timestamp")
        data_list = []
        for group in df_[['Temperature', 'Humidity', 'Light', 'Voltage']].groupby([df.mote_id, df.index.date]):
            if len(group[1]) == 24:
                data_list.append(group[1].values)    
            
        self.data = torch.FloatTensor(np.array(data_list))
        #self.data = self.normalize(data) if normalize else data
        self.seq_len = self.data.size(1)
        
        #Estimates distribution parameters of deltas (Gaussian) from normalized data
        original_deltas = self.data[:, :, -1] - self.data[:,: , 0]
        self.original_deltas = original_deltas
        #self.or_delta_max, self.or_delta_min = original_deltas.max(), original_deltas.min() 
        deltas = self.data[:, :, -1] - self.data[:,:, 0]
        self.deltas = deltas
        self.delta_mean, self.delta_std = deltas.mean(), deltas.std()
        #self.delta_max, self.delta_min = deltas.max(), deltas.min()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.scaler = MinMaxScaler()
        self.scaler.fit(x)
        scaled_x = self.scaler.transform(x)
        return scaled_x

    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        return self.scaler.inverse_transform(x)
    
    def sample_deltas(self, number):
        """Sample a vector of (number) deltas from the fitted Gaussian"""
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std
