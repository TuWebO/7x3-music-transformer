import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Precompute up to max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added, same shape
        """
        batch_size, seq_len, _ = x.size()
        if seq_len <= self.max_len:
            return x + self.pe[:, :seq_len]
        # Dynamically compute longer sequences
        device = x.device
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() *
                             (-math.log(10000.0) / self.d_model))
        pe_dynamic = torch.zeros(seq_len, self.d_model, device=device)
        pe_dynamic[:, 0::2] = torch.sin(position * div_term)
        pe_dynamic[:, 1::2] = torch.cos(position * div_term)
        pe_dynamic = pe_dynamic.unsqueeze(0)
        return x + pe_dynamic

class NextNoteTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 5000,
        grid_dims: int = None,
        batch_first: bool = True
    ):
        super().__init__()
        # Project input features to model dimension
        self.input_proj = nn.Linear(feature_dim, d_model)
        # Positional encoding module
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Project back to grid output dimension
        out_dim = grid_dims if grid_dims is not None else feature_dim
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, feature_dim)
        Returns:
            logits: Tensor of shape (batch_size, seq_len, out_dim)
        """
        # Input projection
        x = self.input_proj(x)
        # Add positional encodings
        x = self.pos_encoder(x)
        # Transformer encoding
        x = self.transformer_encoder(x)
        # Output projection
        logits = self.output_proj(x)
        return logits
