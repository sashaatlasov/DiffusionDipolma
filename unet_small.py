import mlx.core as mx
import mlx.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = False):
        super().__init__()
        
        # В MLX Conv2d принимает (in_channels, out_channels, kernel_size)
        self.main_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.main_norm = nn.GroupNorm(8, out_channels)
        self.main_relu = nn.ReLU()
        
        self.res_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.res_norm1 = nn.GroupNorm(8, out_channels)
        self.res_relu1 = nn.ReLU()
        
        self.res_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.res_norm2 = nn.GroupNorm(8, out_channels)
        self.res_relu2 = nn.ReLU()
        
        self.is_res = residual

    def __call__(self, x: mx.array) -> mx.array:
        # Первая свертка
        x = self.main_conv(x)
        x = self.main_norm(x)
        x = self.main_relu(x)
        
        if self.is_res:
            # Residual block
            residual = x
            x = self.res_conv1(x)
            x = self.res_norm1(x)
            x = self.res_relu1(x)
            x = self.res_conv2(x)
            x = self.res_norm2(x)
            x = self.res_relu2(x)
            x = x + residual
            return x / 1.414
        else:
            # Обычный блок - только одна свертка
            return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_block(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 2, stride: int = 2):
        super().__init__()
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel, stride=stride
        )
        self.conv_block = ConvBlock(out_channels, out_channels)

    def __call__(self, x, skip):
        x = self.transpose_conv(x)

        if x.shape[1:3] != skip.shape[1:3]:
            _, h_x, w_x, _ = x.shape
            _, h_skip, w_skip, _ = skip.shape
            crop_h = (h_skip - h_x) // 2
            crop_w = (w_skip - w_x) // 2
            skip = skip[:, crop_h:crop_h + h_x, crop_w:crop_w + w_x, :]

        x = mx.concatenate([x, skip], axis=-1)  # ✅
        x = self.conv_block(x)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.reshape(-1, 1)
        x = mx.sin(self.lin1(x))
        x = self.lin2(x)
        return x
    

class ConditionEmbedding(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.act(self.lin1(x))
        x = self.lin2(x)
        return x
    
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiHeadAttention(dims=channels, num_heads=heads)
    
    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        
        # Reshape to (batch, seq_len, features)
        x_norm = self.norm(x)
        x_norm = x_norm.reshape(b, h * w, c)
        
        # Self-attention
        attn_out = self.attn(x_norm, x_norm, x_norm)
        
        # Reshape back
        attn_out = attn_out.reshape(b, h, w, c)
        
        return x + attn_out


class UnetModel(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, hidden_size: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        # Encoder
        self.init_conv = ConvBlock(in_channels, hidden_size, residual=True)

        self.down1 = DownBlock(hidden_size, hidden_size)
        self.attn1 = SelfAttentionBlock(hidden_size)
        self.down2 = DownBlock(hidden_size, 2 * hidden_size)
        self.down3 = DownBlock(2 * hidden_size, 2 * hidden_size)

        # Bottleneck
        self.bottleneck = SelfAttentionBlock(2 * hidden_size)

        # Embeddings
        self.timestep_embedding = TimestepEmbedding(2 * hidden_size)
        self.cond_embedding = ConditionEmbedding(5, 2 * hidden_size)

        # Decoder
        self.up1 = UpBlock(2 * hidden_size, 2 * hidden_size, kernel=2, stride=2)
        self.up2 = UpBlock(2 * hidden_size, hidden_size, kernel=2, stride=2)
        self.up3 = UpBlock(2 * hidden_size, hidden_size, kernel=2, stride=2)
        
        # Output convolution
        self.out_conv = nn.Conv2d(2 * hidden_size, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def __call__(
        self, 
        x: mx.array,      # image: (batch, channels, height, width)
        m: mx.array,      # momentum: (batch, 2)
        p: mx.array,      # point: (batch, 3)
        t: mx.array       # timestep: (batch, 1) normalized [0,1]
    ) -> mx.array:
        
        # Encoder path
        x = self.init_conv(x)  # (b, hidden_size, 30, 30)
        
        down1 = self.down1(x)      # (b, hidden_size, 15, 15)
        down1 = self.attn1(down1)  # attention after down1
        
        down2 = self.down2(down1)  # (b, 2*hidden_size, 7, 7)
        down3 = self.down3(down2)  # (b, 2*hidden_size, 3, 3)
        
        # Bottleneck
        bottleneck = self.bottleneck(down3)  # (b, 2*hidden_size, 3, 3)
        
        # Embeddings
        cond = mx.concatenate([m, p], axis=1)  # (b, 5)
        temb = self.timestep_embedding(t)      # (b, 2*hidden_size)
        cemb = self.cond_embedding(cond)       # (b, 2*hidden_size)
        
        # Add embeddings to bottleneck
        temb = temb.reshape(-1, 1, 1, 2 * self.hidden_size)
        cemb = cemb.reshape(-1, 1, 1, 2 * self.hidden_size,)
        bottleneck = bottleneck + temb + cemb
        
        # Decoder path
        # up1 = self.up1(bottleneck, down3)  # (b, 2*hidden_size, 6, 6)
        # up2 = self.up2(up1, down2)         # (b, hidden_size, 12, 12)
        # up3 = self.up3(up2, down1)         # (b, hidden_size, 24, 24)
        up1 = self.up1(bottleneck, down2)
        up2 = self.up2(up1, down1)
        up3 = self.up3(up2, x)  # или вообще убрать этот skip


        # Final convolution with skip connection
        # Обрезаем x до размера up3
        _, _, h_up3, w_up3 = up3.shape
        _, _, h_x, w_x = x.shape
        
        if h_x != h_up3 or w_x != w_up3:
            crop_h = (h_x - h_up3) // 2
            crop_w = (w_x - w_up3) // 2
            x_cropped = x[:, :, crop_h:crop_h + h_up3, crop_w:crop_w + w_up3]
        else:
            x_cropped = x
        
        out = self.out_conv(mx.concatenate([up3, x_cropped], axis=1))
        out = self.sigmoid(out)
        
        return out
