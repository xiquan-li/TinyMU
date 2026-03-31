import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        """Initialize a Batchnorm layer. """
        m.bias.data.fill_(0.)
        m.weight.data.fill_(1.)


def get_projector(proj_name, d_in, d_out, **kwargs): 
    if proj_name.lower() == "reslinearprojector": 
        return ResLinearProjector(d_in, d_out, **kwargs)
    elif proj_name.lower() == "linearprojector": 
        return LinearProjector(d_in, d_out, **kwargs)
    elif proj_name.lower() == "transformerprojector": 
        return TransformerProjector(d_in, d_out)
    else: 
        raise Exception('The projector {} is not supported'.format(proj_name))
    

class ResLinearProjector(nn.Module):
    def __init__(
            self, 
            d_in: int, 
            d_out: int, 
            p: float=0.5,
            **kwargs  
    ) -> None:
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.linear1)
        init_layer(self.linear2)
        init_bn(self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class LinearProjector(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_out: int, 
                 **kwargs
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_h = kwargs.get("d_h", 1024)
        self.d_out = d_out
        self.linear1 = nn.Linear(self.d_in, self.d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.d_h, self.d_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    

class TransformerProjector(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_out: int, 
                 **kwargs
    ) -> None:
        super().__init__()
        self.prefix_length = kwargs.get("prefix_length", 32)
        self.num_layers = kwargs.get("num_layers", 8)
        self.transformer = Transformer(d_out, 8, self.num_layers)
        self.linear = nn.Linear(d_in, d_out)
        self.prefix_const = nn.Parameter(torch.randn(
            self.prefix_length, 
            d_out
        ), requires_grad=True)

    def forward(self, x, mask=None):
        x = self.linear(x)  # (btz, seq, d)
        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0], 
            *self.prefix_const.shape
        )  # (btz, p_len, d)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix, mask)[:, x.size(1):]  # (btz, prefix_len, d_out)
        return out 
        

class Transformer(nn.Module):
    def __init__(self, 
                 dim_self: int, 
                 num_heads: int, 
                 num_layers: int, 
                 dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., 
                 act=F.relu, 
                 norm_layer: nn.Module = nn.LayerNorm, 
                 enc_dec: bool = False
    ) -> None:
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=F.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention

    
class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=F.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x