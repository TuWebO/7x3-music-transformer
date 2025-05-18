import torch
import pytest
from models import PositionalEncoding, NextNoteTransformer


def test_positional_encoding_shape():
    batch_size, seq_len, d_model = 2, 50, 16
    x = torch.zeros(batch_size, seq_len, d_model)
    pe = PositionalEncoding(d_model=d_model, max_len=100)
    out = pe(x)
    assert out.shape == (batch_size, seq_len, d_model)


@pytest.mark.parametrize(
    "feature_dim,d_model,nhead,num_layers,seq_len,batch_size,grid_dims", [
        (20, 32, 4, 2, 10, 3, 20),  # grid_dims equals feature_dim
        (15, 16, 2, 1, 5, 4, 10),   # explicit grid_dims smaller than feature_dim
    ]
)
def test_transformer_forward_shape(feature_dim, d_model, nhead, num_layers, seq_len, batch_size, grid_dims):
    model = NextNoteTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        grid_dims=grid_dims,
        batch_first=True
    )
    x = torch.randn(batch_size, seq_len, feature_dim)
    logits = model(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, seq_len, grid_dims)


def test_default_grid_dims_equals_feature_dim():
    feature_dim = 12
    d_model = 8
    nhead = 2
    num_layers = 1
    seq_len = 7
    batch_size = 5

    model = NextNoteTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        grid_dims=None,
        batch_first=True
    )
    x = torch.randn(batch_size, seq_len, feature_dim)
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, feature_dim)
