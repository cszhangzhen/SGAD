import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.agument_num
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)

        # self.conv1 = GINConv(Sequential(Linear(self.num_features, self.nhid), BatchNorm1d(self.nhid)), train_eps=True)
        # self.conv2 = GINConv(Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid)), train_eps=True)
        # self.conv3 = GINConv(Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid)), train_eps=True)
        
        # Base model with GATConv
        # self.conv1 = GATConv(self.num_features, self.nhid, heads=8, concat=False)
        # self.conv2 = GATConv(self.nhid, self.nhid, heads=8, concat=False)
        # self.conv3 = GATConv(self.nhid, self.nhid, heads=8, concat=False)

        self.lin1 = torch.nn.Linear(self.nhid, self.num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index))
        x1 = gmp(x, batch)

        x = F.relu(self.conv2(x, edge_index))
        x2 = gmp(x, batch)

        x = F.relu(self.conv3(x, edge_index))
        x3 = gmp(x, batch)

        x = x3
        x = F.log_softmax(self.lin1(x), dim=-1)

        return x

    def inference(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index))
        x1 = gmp(x, batch)

        x = F.relu(self.conv2(x, edge_index))
        x2 = gmp(x, batch)

        x = F.relu(self.conv3(x, edge_index))
        x3 = gmp(x, batch)

        x = x3

        x = F.softmax(self.lin1(x), dim=-1)

        return x