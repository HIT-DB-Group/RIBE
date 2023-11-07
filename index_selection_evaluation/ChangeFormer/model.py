import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class Prediction(nn.Module):
    def __init__(self, in_feature = 69, hid_units = 256, contract = 1, mid_layers = True, res_con = True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)
        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)
        self.mid_mlp3 = nn.Linear(hid_units, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 2)

    def forward(self, features):
        
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            mid = F.relu(self.mid_mlp3(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = self.out_mlp2(hid)

        return out

class FeatureEmbed(nn.Module):
    def __init__(self, embed_size=32, types=32,columns=300):
        super(FeatureEmbed, self).__init__()
        self.embed_size=embed_size
        self.typeEmbed = nn.Embedding(types, embed_size,padding_idx=0)
        self.columnEmbed = nn.Embedding(columns, embed_size,padding_idx=0)
        self.project = nn.Linear(embed_size*10 +1 +1 , embed_size*10 +1 +1 )

    def forward(self, feature, join_tables):
        typeId,tableId,columnsIds,joinId,joinColumnsIds,cost_card=torch.split(feature,(1,1,2,3,3,2), dim = -1)
        typeEmb = self.getType(typeId.long())        
        
        bt,sq,ta=joinId.shape
        joinId=joinId.reshape(bt,-1).long()
        joinEmb=join_tables[torch.arange(join_tables.size(0)).unsqueeze(1), joinId].squeeze(-2)
        joinEmb=joinEmb.reshape(bt,sq,-1)

        bt,sq,ta=tableId.shape
        tableEmb=join_tables[torch.arange(join_tables.size(0)).unsqueeze(1), tableId.view(bt,-1).long()].squeeze(-2)
        tableEmb=tableEmb.view(bt,sq,-1)
        
        columnsEmb = self.getcolumn(columnsIds)
        columnsEmb=columnsEmb.reshape(bt,sq,-1)
        joinColumnsEmb = self.getcolumn(joinColumnsIds)
        joinColumnsEmb=joinColumnsEmb.reshape(bt,sq,-1)

        final = torch.cat((typeEmb, tableEmb, columnsEmb, joinEmb, joinColumnsEmb, cost_card), dim = 2)
        final = F.leaky_relu(self.project(final))
        
        return final
    
    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())
        return emb.squeeze(2)

    def getcolumn(self, columnId):
        emb = self.columnEmbed(columnId.long())
        return emb
    
    def get_output_size(self):
        size = self.embed_size*10 +1 +1
        return size


class TableEmbed(nn.Module):
    def __init__(self, embed_size=32,output_dim=32,n_tables=32):
        super(TableEmbed, self).__init__()
        self.tableEmbed = nn.Embedding(n_tables, embed_size)
        self.linearTable = nn.Linear(embed_size, output_dim)
        
    
    def forward(self, feature):
        output=self.tableEmbed(feature)
        tables_embedding=F.leaky_relu(self.linearTable(output))
        return tables_embedding

class IndexesEmbed(nn.Module):
    def __init__(self, embed_size=32, columns=300, position_num=40,hid_dim=256):
        super(IndexesEmbed, self).__init__()

        self.embed_size=embed_size
        self.position_num=position_num
        self.indexEmb = nn.Embedding(columns, embed_size,padding_idx=0)

        self.project = nn.Linear(embed_size*position_num , embed_size*position_num )
        self.mid1=nn.Linear(embed_size*position_num,hid_dim)
        self.output=nn.Linear(hid_dim,embed_size)

    def forward(self, feature):
        final = self.getIndex(feature)
        final = final.reshape(final.shape[0],-1)
        final = F.leaky_relu(self.project(final))
        final = F.leaky_relu(self.mid1(final))
        final = F.leaky_relu(self.output(final))
        
        return final

    def getIndex(self, indexID):
        emb = self.indexEmb(indexID.long())
        return emb
    
    
    def get_output_size(self):
        size = self.embed_size
        return size

class ChangeFormer(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, join_schema_head_size=2,\
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, join_schema_layers=2, \
                 pred_hid = 256
                ):
        
        super(ChangeFormer,self).__init__()
        hidden_dim = emb_size*10 +1 +1
        self.join_schema_output_dim=emb_size
        
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.join_schema_head_size=join_schema_head_size
        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.tree_bias_encoder = nn.Embedding(64, head_size, padding_idx=0)
        self.join_bias_encoder = nn.Embedding(8, join_schema_head_size, padding_idx=0)

        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        
        join_schema_encoders = [EncoderLayer(self.join_schema_output_dim, ffn_dim, dropout, attention_dropout_rate, join_schema_head_size)
                    for _ in range(join_schema_layers)]
        
        self.layers = nn.ModuleList(encoders)
        self.join_schema_layers=nn.ModuleList(join_schema_encoders)
        
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        self.indexEmb = IndexesEmbed(emb_size)

        self.embbed_layer = FeatureEmbed(emb_size)
        self.join_schema_embbed_layer=TableEmbed(output_dim=self.join_schema_output_dim)
        self.pred = Prediction(hidden_dim+self.indexEmb.get_output_size(), pred_hid)

        
    def forward(self, batched_data):
        features=batched_data.features
        index_features=batched_data.index_features

        attention_bias = batched_data.attention_bias
        attention_bias_SuperNode=batched_data.attention_bias_SuperNode.clone()
        join_schema_bias=batched_data.join_schema_bias
        join_features=batched_data.join_features
        heights = batched_data.heights
        
        n_batch=features.shape[0]
        n_table=join_schema_bias.shape[1]
        assert n_table==join_features.shape[0]
        join_schema_bias=self.join_bias_encoder(join_schema_bias.long())
        join_schema_bias=join_schema_bias.permute(0, 3, 1, 2)
        
        join_features=join_features.unsqueeze(1).repeat(n_batch, 1).view(-1,n_table)
        join_features=self.join_schema_embbed_layer(join_features.long()).view(n_batch,-1, self.join_schema_output_dim)

        for enc_layer in self.join_schema_layers:
            join_features = enc_layer(join_features, join_schema_bias)

        attention_bias_SuperNode=attention_bias_SuperNode.unsqueeze(0).repeat(n_batch,1, 1) 
        attention_bias_SuperNode=attention_bias_SuperNode.unsqueeze(1).repeat(1, self.head_size, 1, 1) 
        attention_bias=self.tree_bias_encoder(attention_bias.long()).permute(0, 3, 1, 2)

        attention_bias_SuperNode[:, :, 1:, 1:] = attention_bias_SuperNode[:, :, 1:, 1:] + attention_bias

        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        attention_bias_SuperNode[:, :, 1:, 0] = attention_bias_SuperNode[:, :, 1:, 0] + t
        attention_bias_SuperNode[:, :, 0, :] = attention_bias_SuperNode[:, :, 0, :] + t
        
        node_feature = self.embbed_layer(features,join_features)
        node_feature = node_feature + self.height_encoder(heights.long())
        
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)

        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, attention_bias_SuperNode)
        output = self.final_ln(output)
        output=output[:,0,:] 
        output = output.view(n_batch, -1)
        index_output=self.indexEmb(index_features)
        final_output = torch.cat([output, index_output], dim=1)
        return self.pred(final_output)
        





class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()
        self.head_size = head_size
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v) 

        x = x.transpose(1, 2).contiguous() 
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

