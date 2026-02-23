"""
Resource Forecasting Module using LSTM-Transformer Hybrid.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for time series forecasting.
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src):
        # src: [batch_size, seq_len, input_dim]
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerLSTMHybrid(nn.Module):
    """
    Hybrid model combining Transformer (for long-term dependencies) 
    and LSTM (for short-term sequential patterns) to forecast resource usage.
    """
    def __init__(self, input_features: int = 4, hidden_dim: int = 64, output_horizon: int = 12):
        super(TransformerLSTMHybrid, self).__init__()
        
        # Transformer Feature Extractor
        self.transformer = TimeSeriesTransformer(input_features, d_model=hidden_dim)
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        # Prediction Heads
        self.fc_cpu = nn.Linear(hidden_dim, output_horizon)
        self.fc_memory = nn.Linear(hidden_dim, output_horizon)
        
    def forward(self, x):
        """
        x: [batch, seq_len, features] (e.g. past 60 seconds of metrics)
        Returns: Forecast for CPU and Memory for next `output_horizon` steps
        """
        # 1. Transformer features
        # [batch, seq_len, hidden_dim]
        tf_out = self.transformer(x)
        
        # 2. LSTM processing
        # Use only the last hidden state for prediction
        lstm_out, (h_n, c_n) = self.lstm(tf_out)
        
        # Take the output of the last time step
        # [batch, hidden_dim]
        last_step = lstm_out[:, -1, :]
        
        # 3. Forecast
        pred_cpu = self.fc_cpu(last_step)
        pred_memory = self.fc_memory(last_step)
        
        return pred_cpu, pred_memory

# Global instance placeholder
resource_forecaster = None
