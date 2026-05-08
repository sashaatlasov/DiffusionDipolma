import mlx.core as mx
import mlx.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.ReLU()

        # self.norm1 = nn.Identity()
        # self.norm2 = nn.Identity()

    def __call__(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def __call__(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def __call__(self, x, skip):
        x = self.up(x)

        if skip.shape[1] != x.shape[1] or skip.shape[2] != x.shape[2]:
            _, h_s, w_s, _ = skip.shape
            _, h_x, w_x, _ = x.shape

            crop_h = (h_s - h_x) // 2
            crop_w = (w_s - w_x) // 2

            skip = skip[:, crop_h:crop_h+h_x, crop_w:crop_w+w_x, :]

        x = mx.concatenate([x, skip], axis=-1)
        x = self.conv(x)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(1, dim)
        self.lin2 = nn.Linear(dim, dim)

    def __call__(self, t):
        t = t.reshape(-1, 1)
        t = mx.sin(self.lin1(t))
        t = self.lin2(t)
        return t


class ConditionEmbedding(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, dim)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(dim, dim)

    def __call__(self, x):
        x = self.act(self.lin1(x))
        x = self.lin2(x)
        return x
    

def pad_to_32(x):
    _, h, w, _ = x.shape
    pad_h = 32 - h
    pad_w = 32 - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return mx.pad(
        x,
        [(0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)]
    )

def crop_back(x):
    return x[:, 1:31, 1:31, :]


class UNet(nn.Module):
    def __init__(self, in_ch=1, base=128):
        super().__init__()

        self.init = ConvBlock(in_ch, base)

        self.down1 = DownBlock(base, base)
        self.down2 = DownBlock(base, base * 2)
        self.down3 = DownBlock(base * 2, base * 2)

        self.bottleneck = ConvBlock(base * 2, base * 2)

        self.up1 = UpBlock(base * 2, base * 2, base * 2)
        self.up2 = UpBlock(base * 2, base * 2, base)
        self.up3 = UpBlock(base, base, base)

        self.out = nn.Conv2d(base, 1, kernel_size=1)

        self.temb = TimestepEmbedding(base * 2)
        self.cemb = ConditionEmbedding(5, base * 2)

    def __call__(self, x, m, p, t):

        x = pad_to_32(x)

        x0 = self.init(x)

        x1, skip1 = self.down1(x0)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)

        x4 = self.bottleneck(x3)

        cond = mx.concatenate([m, p], axis=-1)

        temb = self.temb(t)
        cemb = self.cemb(cond)

        emb = temb + cemb
        emb = emb.reshape(-1, 1, 1, x4.shape[-1])

        x4 = x4 + emb

        x = self.up1(x4, skip3)  # 4→2
        x = self.up2(x, skip2)   # 2→1
        x = self.up3(x, skip1)

        x = self.out(x)
        x = crop_back(x)
        return x