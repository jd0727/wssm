from models.funcational import *


class FeedForward(nn.Module):
    def __init__(self, channels, inner_channels, dropout=0.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(channels, inner_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_channels, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.backbone(x)


class FeedForwardResidual(FeedForward):
    def __init__(self, channels, inner_channels, dropout=0.0):
        super().__init__(channels, inner_channels, dropout=dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        out = self.norm(x)
        out = super(FeedForwardResidual, self).forward(out)
        return out + x


class LinearAttentionMutiHead(nn.Module):
    def __init__(self, channels, num_head=8, attn_channels=64, dropout=0.0):
        super().__init__()
        inner_channels = attn_channels * num_head
        reproject = not (num_head == 1 and attn_channels == channels)
        self.num_head = num_head
        self.attn_channels = attn_channels
        self.qkv = nn.Linear(channels, inner_channels * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(inner_channels, channels),
            nn.Dropout(dropout)
        ) if reproject else nn.Identity()

    def forward(self, x):
        Nb, L, C = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [itm.reshape(Nb, L, self.num_head, -1).permute(0, 2, 1, 3) for itm in qkv]
        dots = torch.matmul(q, k.transpose(-1, -2)) * (self.attn_channels ** -0.5)
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(Nb, L, self.attn_channels * self.num_head)
        return self.out(out)


class LinearAttentionResidual(LinearAttentionMutiHead):
    def __init__(self, channels, num_head=8, attn_channels=64, dropout=0.0):
        super().__init__(channels, num_head=num_head, attn_channels=attn_channels, dropout=dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        out = self.norm(x)
        out = super(LinearAttentionResidual, self).forward(out)
        return out + x


class Transformer(nn.Module):
    def __init__(self, channels, mlp_channels=256, attn_channels=64, depth=6, num_head=8, dropout=0.0):
        super().__init__()
        backbone = []
        for _ in range(depth):
            backbone += [
                LinearAttentionResidual(channels=channels, num_head=num_head, attn_channels=attn_channels, dropout=dropout),
                FeedForwardResidual(channels=channels, inner_channels=mlp_channels, dropout=dropout),
            ]
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ViT(nn.Module):
    def __init__(self, channels, img_size=(256, 256), patch_size=(32, 32), num_cls=20, depth=6, num_head=8,
                 mlp_channels=256, in_channels=3, attn_channels=64, dropout=0.0):
        super().__init__()
        W, H = img_size
        Wp, Hp = patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        assert W % Wp == 0 and H % Hp == 0, 'Image dimensions must be divisible by the patch size.'
        grid_size = (W // Wp, H // Hp)
        self.grid_size = grid_size

        num_patches = grid_size[0] * grid_size[1]
        patch_channels = in_channels * Hp * Wp
        self.projector = nn.Linear(patch_channels, channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, channels))
        self.cls_token = nn.Parameter(torch.randn(1, 1, channels))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(channels=channels, depth=depth, num_head=num_head,
                                       mlp_channels=mlp_channels, attn_channels=attn_channels, dropout=dropout)
        self.linear = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, num_cls)
        )

    def forward(self, img):
        Nb, C, H, W = img.size()
        Wp, Hp = self.patch_size
        Wg, Hg = self.grid_size
        img = img.view(Nb, C, Hg, Hp, Wg, Wp).permute(0, 2, 4, 1, 3, 5).reshape(Nb, Hg * Wg, C * Hp * Wp)
        x = self.projector(img)
        cls_tokens = self.cls_token.repeat(Nb, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(Hg * Wg + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    v = ViT(
        channels=1024, img_size=(256, 256), patch_size=(16, 16), num_cls=20, depth=2, num_head=8,
        mlp_channels=256, in_channels=3, attn_channels=64, dropout=0.1,
    )

    img = torch.randn(2, 3, 256, 256)
    preds = v(img)  # (1, 1000)
