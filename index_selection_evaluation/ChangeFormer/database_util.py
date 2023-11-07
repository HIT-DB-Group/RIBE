import numpy as np
import pandas as pd
import csv
import torch
import torch.nn.functional as F

class Batch():
    def __init__(self, features, attention_bias,attention_bias_SuperNode, join_schema_bias,join_features,heights,index_features):
        super(Batch, self).__init__()

        self.features = features
        self.attention_bias = attention_bias
        self.attention_bias_SuperNode=attention_bias_SuperNode
        self.join_schema_bias = join_schema_bias
        self.heights=heights
        self.join_features=join_features
        self.index_features=index_features
        
    def to(self, device):

        self.features = self.features.to(device)
        self.attention_bias = self.attention_bias.to(device)
        self.attention_bias_SuperNode = self.attention_bias_SuperNode.to(device)        
        self.join_schema_bias = self.join_schema_bias.to(device)
        self.heights=self.heights.to(device)
        self.join_features=self.join_features.to(device)
        self.index_features=self.index_features.to(device)

        return self

    def __len__(self):
        return self.features.shape[0]


def collator(small_set):
    y = small_set[1]
    y_tensor = torch.tensor(y)
    one_hot_tensor = F.one_hot(y_tensor, num_classes=2)
    length=np.array([s['length'] for s in small_set[0]])
    maxlength=np.max(length)
    pad_length=maxlength-length
    
    features=[np.pad(s['features'], ((0, pad_length[i]), (0, 0)), mode='constant', constant_values=-1) for i,s in enumerate(small_set[0])]
    features=torch.tensor(np.stack(features), dtype=torch.float)+1
    features[:,:,[-1,-2]]=features[:,:,[-1,-2]]-1
    features[features<0]=0

    attention_bias=[np.pad(s['attention_bias'], ((0, pad_length[i]), (0, pad_length[i])), mode='constant', constant_values=-1) for i,s in enumerate(small_set[0])]
    attention_bias=torch.tensor(np.stack(attention_bias), dtype=torch.float)+1
    attention_bias_SuperNode=torch.zeros([maxlength+1,maxlength+1], dtype=torch.float)
    
    join_schema_bias=[np.pad(s['join_schema_bias'], ((1,0), (1,0)), mode='constant', constant_values=-1) for i,s in enumerate(small_set[0])]
    join_schema_bias=torch.tensor(np.stack(join_schema_bias), dtype=torch.float)+1
    join_features=torch.tensor([i for i in range(join_schema_bias.shape[1])], dtype=torch.float)
    
    index_features=[s['indexe_features'] for i,s in enumerate(small_set[0])]
    index_features=torch.tensor(np.stack(index_features), dtype=torch.float)

    heights=[np.pad(s['heights'], ((0, pad_length[i])), mode='constant', constant_values=-1) for i,s in enumerate(small_set[0])]
    heights=torch.tensor(np.stack(heights), dtype=torch.float)+1

    return Batch(features, attention_bias,attention_bias_SuperNode, join_schema_bias,join_features,heights,index_features), torch.LongTensor(one_hot_tensor)
