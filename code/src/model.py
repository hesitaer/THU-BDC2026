import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CrossStockAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, stock_features):
        attended, _ = self.cross_attention(stock_features, stock_features, stock_features)
        output = self.norm1(stock_features + self.dropout(attended))
        ffn_out = self.ffn(output)
        output = self.norm2(output + self.dropout(ffn_out))
        return output

class FeatureAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.time_attention = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        attended, _ = self.time_attention(x, x, x)
        attended = self.norm(x + self.dropout(attended))
        attended = self.linear(attended)
        attended = torch.tanh(attended)
        weights = torch.mean(attended, dim=-1, keepdim=True)
        weights = F.softmax(weights, dim=1)
        output = torch.sum(x * weights, dim=1)
        return self.dropout(output)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks):
        super(StockTransformer, self).__init__()
        self.model_type = 'EnhancedRankingTransformer'
        self.config = config
        self.num_stocks = num_stocks
        
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
        
        self.cross_stock_layers = nn.ModuleList([
            CrossStockAttention(config['d_model'], config['nhead'], config['dropout'])
            for _ in range(2)
        ])
        
        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.GELU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)
        
        temporal_features = self.temporal_encoder(src_proj)
        
        aggregated_features = self.feature_attention(temporal_features)
        
        stock_features = aggregated_features.view(batch_size, num_stocks, -1)
        
        for layer in self.cross_stock_layers:
            stock_features = layer(stock_features)
        
        interactive_features = stock_features.view(batch_size * num_stocks, -1)
        
        ranking_features = self.ranking_layers(interactive_features)
        
        scores = self.score_head(ranking_features)
        
        output = scores.view(batch_size, num_stocks)
        
        return output