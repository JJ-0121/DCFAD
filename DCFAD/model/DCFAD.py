import torch
import torch.nn as nn
from einops import rearrange
from .crossattn import MultiHeadCrossAttention
from .embed import DataEmbedding, ChannelEmbedding
from .RevIN import RevIN
from torch_frft.dfrft_module import *

class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_list=(3, 5, 7), groups=1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels,out_channels,kernel_size=k,padding="same",padding_mode="circular",bias=False,groups=groups)
            for k in kernel_list
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.convs:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):  # x: [B, C, L]
        outs = [conv(x) for conv in self.convs]
        # average over branches -> [B, C, L]
        return torch.stack(outs, dim=-1).mean(-1)

class T_FEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_q, x_kv, attn_mask=None):
        att_list = []
        for layer in self.attn_layers:
            x_q, att = layer(x_q, x_kv, attn_mask)
            att_list.append(att)
        if self.norm is not None:
            x_q = self.norm(x_q)
        return x_q, att_list


class T_FEnc(nn.Module):
    def __init__(self, d_model, e_layers, n_heads=8, dropout=0.0):
        super().__init__()
        self.enc = T_FEncoder(
            [MultiHeadCrossAttention(d_model, n_heads, dropout) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x_q, x_kv, attn_mask=None):
        x, att = self.enc(x_q, x_kv, attn_mask)
        return self.pro(x), att


class DCFAD(nn.Module):

    def __init__(self,c_in,c_out,revin,d_model=512,e_layers=3,win_size=100,n_heads=8,dropout=0.1,device=None,f_tr=0.5,conv_kernel_list=(3,5,7)):
        super().__init__()
        self.device = device
        self.pre_conv = Inception_Block(
            in_channels=c_in,
            out_channels=c_in,
            kernel_list=conv_kernel_list,
            groups=c_in,
        )
        self.pre_conv_bn = nn.BatchNorm1d(c_in)
        self.pre_conv_act = nn.GELU()
        self.tem_tf = T_FEnc(d_model, e_layers, n_heads=n_heads, dropout=dropout)
        self.var_ft = T_FEnc(d_model, e_layers, n_heads=n_heads, dropout=dropout)
        self.emb_t = DataEmbedding(c_in, d_model)
        self.emb_t_f = DataEmbedding(2 * c_in, d_model)
        self.emb_c = ChannelEmbedding(win_size, d_model)
        self.emb_c_f = ChannelEmbedding(2 * win_size, d_model)
        self.revin = revin
        self.revin_layer = RevIN(num_features=c_in)
        # Learnable fractional orders in (0,1)
        self.t_alpha = nn.Parameter(torch.tensor(0.0))
        self.c_alpha = nn.Parameter(torch.tensor(0.0))
        self.project_out_t = nn.Linear(d_model, c_in)
        self.project_out_c = nn.Linear(d_model, win_size)
        self.f_tr = f_tr

    def forward(self, x):  # x: [B, L, C]
        if self.device is not None:
            x = x.to(self.device)
        if self.revin:
            x = self.revin_layer(x,"norm")
        x_conv = x.permute(0, 2, 1)              # [B, C, L]
        x_conv = self.pre_conv_act(self.pre_conv_bn(self.pre_conv(x_conv)))
        x_conv = x_conv.permute(0, 2, 1)         # [B, L, C]
        # ---------- Temporal path (T → F) ----------
        alpha_t = torch.sigmoid(self.t_alpha)
        x_frft_t = dfrft(x_conv, alpha_t, dim=-1)  # along channel
        z_t = torch.cat((x_frft_t.real, x_frft_t.imag), dim=-1)  # [B, L, 2C]
        x_emb = self.emb_t(x_conv)
        z_emb = self.emb_t_f(z_t)
        tf_out, _ = self.tem_tf(x_emb, z_emb)
        tf_out = self.project_out_t(tf_out)
        # ---------- Variable path (F → T) ----------
        alpha_c = torch.sigmoid(self.c_alpha)
        x_frft_c = dfrft(x_conv, alpha_c, dim=-2)
        z_c = torch.cat((x_frft_c.real, x_frft_c.imag), dim=-2)  # [B, 2L, C]
        x_c = x_conv.transpose(1, 2)   # [B, C, L]
        z_c = z_c.transpose(1, 2)      # [B, C, 2L]
        x_c_emb = self.emb_c(x_c)
        z_c_emb = self.emb_c_f(z_c)
        ft_out, _ = self.var_ft(z_c_emb, x_c_emb)
        ft_out = self.project_out_c(ft_out).transpose(1, 2)  # [B, L, C]
        x = tf_out + self.f_tr * ft_out
        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x
