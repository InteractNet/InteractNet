import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import TAGConv
from dgl import unbatch
from dgl.nn import TWIRLSConv
from torch.nn import TransformerEncoderLayer
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from torch.autograd import Variable

from Attention import MultiHeadAttention


class InteractNet(nn.Module):

    def __init__(self):
        super(InteractNet, self).__init__()

        self.transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout= 0.01)
        self.protein_graph_conv = nn.ModuleList()
        for i in range(5):
            self.protein_graph_conv.append(TAGConv(31, 31, 2))

        self.ligand_graph_conv = nn.ModuleList()


        self.ligand_graph_conv.append(TAGConv(74, 70, 2))
        self.ligand_graph_conv.append(TAGConv(70, 65, 2))
        self.ligand_graph_conv.append(TAGConv(65, 60, 2))
        self.ligand_graph_conv.append(TAGConv(60, 55, 2))
        self.ligand_graph_conv.append(TAGConv(55, 31, 2))
        #self.ligand_graph_conv.append(TAGConv(50, 31, 2))
        # self.ligand_graph_conv.append(TAGConv(45, 40, 2))
        # self.ligand_graph_conv.append(TAGConv(40, 31, 2))
        #self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        #self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        #self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        #self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))

        self.pooling_ligand = nn.Linear(31, 1)
        self.pooling_protein = nn.Linear(31, 1)

        self.dropout = 0.2

        self.bilstm = nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=self.dropout)

        self.fc_in = nn.Linear(8680, 4340) #1922

        self.fc_out = nn.Linear(4340, 1)
        self.attention = MultiHeadAttention(62, 62, 2)
    #    self.W_s1 = nn.Linear(60, 45) #62
    #    self.W_s2 = nn.Linear(45, 30)

    #def attention_net(self, lstm_output):
    #    attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
    #    attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
    #    attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

    #    return attn_weight_matrix

    def forward(self, g):
        feature_protein = g[0].ndata['h']
        feature_smile = g[1].ndata['h']

        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[0], feature_protein))

        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(g[1], feature_smile))


        pool_ligand = GlobalAttentionPooling(self.pooling_ligand)
        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        protein_rep = pool_protein(g[0], feature_protein).view(-1, 31)
        ligand_rep = pool_ligand(g[1], feature_smile).view(-1, 31)
        #sequence = []
        #for item in protein_rep:
        #    sequence.append(item.view(1, 31))
        #    sequence.append(ligand_rep)
        Seq_protein = self.transformer(feature_protein)
        Seq_ligand = self.transformer(feature_smile)
        Graph = torch.cat((ligand_rep, protein_rep), dim=0).view(1, -1, 31)
        mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).cuda()
        mask[0, Graph.size()[1]:140, :] = 0
        mask[0, :, Graph.size()[1]:140] = 0
        mask[0, :, Graph.size()[1] - 1] = 1
        mask[0, Graph.size()[1] - 1, :] = 1
        mask[0,  Graph.size()[1] - 1,  Graph.size()[1] - 1] = 0
        sequence = F.pad(input=Graph, pad=(0, 0, 0, 140 - Graph.size()[1]), mode='constant', value=0)
        sequence = sequence.permute(1, 0, 2)
        Seq = torch.cat((Seq_protein, Seq_ligand), dim=0).view(1, -1, 31)
        h_0 = Variable(torch.zeros(2, 1, 31).cuda())
        c_0 = Variable(torch.zeros(2, 1, 31).cuda())

        output, _ = self.bilstm(sequence, (h_0, c_0))

        output = output.permute(1, 0,  2)

        out = self.attention(output, mask=mask)
        #attn_weight_matrix = self.attention_net(output)
        #out = torch.bmm(attn_weight_matrix, output)
        out = F.relu(self.fc_in(out.view(-1, out.size()[1]*out.size()[2])))

        out = torch.sigmoid(self.fc_out(out))
        return out


