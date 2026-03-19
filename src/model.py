import torch
import torch.nn as nn
import torch.nn.functional as F

class SE1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, mid, 1),
            nn.SiLU(),
            nn.Conv1d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))

class SpatialAttention1D(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=k, padding=k // 2)

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx = torch.max(x, 1, keepdim=True)[0]
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))
        return x * w

class MultiScaleResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        b1 = nn.Conv1d(in_c, out_c // 3, 3, padding=1, dilation=1)
        b2 = nn.Conv1d(in_c, out_c // 3, 5, padding=2, dilation=1)
        b3 = nn.Conv1d(in_c, out_c - 2 * (out_c // 3), 3, padding=2, dilation=2)
        self.branches = nn.ModuleList([b1, b2, b3])
        self.gn = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()
        self.se = SE1D(out_c)
        self.spa = SpatialAttention1D(7)
        self.proj = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        outs = [self.act(b(x)) for b in self.branches]
        y = torch.cat(outs, 1)
        y = self.act(self.gn(y) + self.proj(x))
        y = self.se(y)
        y = self.spa(y)
        return y

class CNN_ATT_BiRNN_Model(nn.Module):
    def __init__(self, in_ch=50, rnn_type="gru", rnn_hidden=256, center_window=5, use_ln_pre_rnn=True):
        super().__init__()
        self.center_window = center_window
        self.stem = nn.Conv1d(in_ch, 64, 1)
        self.block1 = MultiScaleResBlock(64, 128)
        self.block2 = MultiScaleResBlock(128, 192)
        self.block3 = MultiScaleResBlock(192, 256)
        self.use_ln_pre_rnn = use_ln_pre_rnn

        if self.use_ln_pre_rnn:
            self.pre_rnn_norm = nn.LayerNorm(256)

        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        feat_dim = 2 * rnn_hidden
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.silu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.transpose(1, 2)

        if self.use_ln_pre_rnn:
            x = self.pre_rnn_norm(x)

        x, _ = self.rnn(x)
        B, L, D = x.shape
        c = L // 2
        w = self.center_window
        idx = torch.arange(c - w, c + w + 1, device=x.device).clamp(0, L - 1)
        neigh = x[:, idx, :]

        with torch.no_grad():
            dist = torch.arange(-w, w + 1, device=x.device, dtype=x.dtype)
            weight = torch.exp(-0.5 * (dist / 2.0) ** 2)
            weight = (weight / weight.sum()).view(1, -1, 1)

        pooled = (neigh * weight).sum(1)
        logit = self.head(pooled).squeeze(-1)
        return logitimport torch
import torch.nn as nn
import torch.nn.functional as F

class SE1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, mid, 1),
            nn.SiLU(),
            nn.Conv1d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))

class SpatialAttention1D(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=k, padding=k // 2)

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx = torch.max(x, 1, keepdim=True)[0]
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))
        return x * w

class MultiScaleResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        b1 = nn.Conv1d(in_c, out_c // 3, 3, padding=1, dilation=1)
        b2 = nn.Conv1d(in_c, out_c // 3, 5, padding=2, dilation=1)
        b3 = nn.Conv1d(in_c, out_c - 2 * (out_c // 3), 3, padding=2, dilation=2)
        self.branches = nn.ModuleList([b1, b2, b3])
        self.gn = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()
        self.se = SE1D(out_c)
        self.spa = SpatialAttention1D(7)
        self.proj = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        outs = [self.act(b(x)) for b in self.branches]
        y = torch.cat(outs, 1)
        y = self.act(self.gn(y) + self.proj(x))
        y = self.se(y)
        y = self.spa(y)
        return y

class CNN_ATT_BiRNN_Model(nn.Module):
    def __init__(self, in_ch=50, rnn_type="gru", rnn_hidden=256, center_window=5, use_ln_pre_rnn=True):
        super().__init__()
        self.center_window = center_window
        self.stem = nn.Conv1d(in_ch, 64, 1)
        self.block1 = MultiScaleResBlock(64, 128)
        self.block2 = MultiScaleResBlock(128, 192)
        self.block3 = MultiScaleResBlock(192, 256)
        self.use_ln_pre_rnn = use_ln_pre_rnn

        if self.use_ln_pre_rnn:
            self.pre_rnn_norm = nn.LayerNorm(256)

        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        feat_dim = 2 * rnn_hidden
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.silu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.transpose(1, 2)

        if self.use_ln_pre_rnn:
            x = self.pre_rnn_norm(x)

        x, _ = self.rnn(x)
        B, L, D = x.shape
        c = L // 2
        w = self.center_window
        idx = torch.arange(c - w, c + w + 1, device=x.device).clamp(0, L - 1)
        neigh = x[:, idx, :]

        with torch.no_grad():
            dist = torch.arange(-w, w + 1, device=x.device, dtype=x.dtype)
            weight = torch.exp(-0.5 * (dist / 2.0) ** 2)
            weight = (weight / weight.sum()).view(1, -1, 1)

        pooled = (neigh * weight).sum(1)
        logit = self.head(pooled).squeeze(-1)
        return logit
