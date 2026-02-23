"""
Graph Neural Network for Task Dependency Modeling.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TaskGNN(nn.Module):
    """
    GNN for encoding task dependency graphs.
    Predicts optimal execution priority or estimated duration based on graph structure.
    """
    
    def __init__(self, node_features: int = 12, hidden_dim: int = 64, output_dim: int = 1):
        super(TaskGNN, self).__init__()
        
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass.
        x: Node features [num_nodes, node_features]
        edge_index: Graph connectivity [2, num_edges]
        batch: Batch vector [num_nodes]
        """
        
        # 1. Message Passing via GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # 2. Global Pooling (Graph-level representation)
        # Using mean pooling, could also try max or attention pooling
        x = global_mean_pool(x, batch)
        
        # 3. Readout / Prediction
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output

    def get_node_embeddings(self, x, edge_index):
        """Get embeddings for individual tasks (nodes) without pooling"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

# Global instance placeholder
task_gnn = None
