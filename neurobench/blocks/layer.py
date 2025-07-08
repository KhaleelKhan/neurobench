import torch.nn as nn
from evnn_pytorch import EGRU

STATELESS_LAYERS = (
    nn.Linear,
    nn.Conv2d,
    nn.Conv1d,
    nn.Conv3d,
)

RECURRENT_CELLS = (nn.RNNCellBase,)

RECURRENT_LAYERS = (nn.RNNBase,)

EGRU_LAYERS = (EGRU,)

SUPPORTED_LAYERS = STATELESS_LAYERS + RECURRENT_LAYERS + RECURRENT_CELLS + EGRU_LAYERS
