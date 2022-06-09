import torch

from einops import repeat, rearrange
from einops.layers.torch import Rearrange


class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, tran_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        '''
        img_w, img_h = img_size, img_size
        patch_w, patch_h = patch_size, patch_size
        nw = img_w // patch_w # num of horizontal patches 
        nh = img_h // patch_h # num of vetical patchs

        num_patches = nw * nh
        patch_dim = in_c * patch_w * patch_h
        '''

        '''
        self.proj = torch.nn.Sequential(
                Rearrange('b c (nw pw) (nh ph) -> b (nw nh) (pw ph c)', pw=patch_w, ph=patch_h),
                torch.nn.Linear(patch_dim, tran_dim) #TODO: patch_dim should be divided into 3? or conv?
                )
        '''
        # timm version uses a conv layer instead of a FC layer
        self.proj = torch.nn.Conv2d(in_c, tran_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)

        return x


class MLP(torch.nn.Module):
    '''
    This module is used in each Encoder Block
    It contains 2 FC layers

    '''
    def __init__(self, tran_dim, hid_dim, dropout=0.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(tran_dim, hid_dim) #fc1
        self.active_fn = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hid_dim, tran_dim) #fc2

    def forward(self, x):
        x = self.active_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class MultiAttention(torch.nn.Module):
    def __init__(self, tran_dim, n_heads=8, head_dim=96, dropout=0.0):
        super().__init__()
        inner_dim = n_heads * head_dim

        self.n_heads = n_heads
        self.scale = head_dim ** -0.5

        self.softmax = torch.nn.Softmax(dim = -1)
        self.dropout = torch.nn.Dropout(dropout)

        self.qkv = torch.nn.Linear(tran_dim, inner_dim*3)

        self.proj = torch.nn.Sequential(
                torch.nn.Linear(inner_dim, tran_dim), # proj
                torch.nn.Dropout(dropout)
                )
        self.proj = torch.nn.Linear(inner_dim, tran_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # to get Q,K,V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)
        
        # use Q,K to get weght
        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale # -1:d, -2:n
        attn = self.softmax(dots) # b,h,n,n
        attn = self.dropout(attn)

        # use weight and V to get Z
        out = torch.matmul(attn, v) # b,h,n,d
        out = rearrange(out, 'b h n d -> b n (h d)') #concatenate to a long row

        out = self.proj(out)
        out = self.dropout(out)

        return out

# deprecated
class PreNorm(torch.nn.Module):
    def __init__(self, dim, af):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.af = af

    def forward(self, x):
        return self.af(self.norm(x))


class EncoderBlock(torch.nn.Module):
    def __init__(self, tran_dim, n_heads, head_dim, mlp_dim, dropout=0.0):
        super().__init__()

        self.norm1 = torch.nn.LayerNorm(tran_dim)
        self.attn = MultiAttention(tran_dim, n_heads=n_heads, head_dim=head_dim, dropout=dropout)
        self.norm2 = torch.nn.LayerNorm(tran_dim)
        self.mlp = MLP(tran_dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.norm1(self.attn(x))
        x = x + self.norm2(self.mlp(x))

        return x


class ViT(torch.nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, n_heads, mlp_dim, c_in=3, head_dim=96, dropout=0.0, emb_dropout=0.0):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, c_in, dim)

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = torch.nn.Parameter(torch.randn(1, img_size//patch_size+1, dim))

        self.blocks = torch.nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(EncoderBlock(dim, n_heads, head_dim, mlp_dim))

        self.to_latent = torch.nn.Identity()

        self.norm = torch.nn.LayerNorm(dim) # last norm
        self.head = torch.nn.Linear(dim, num_classes) # head

    def forward(self, x):
        b, n, patch_dim = x.shape

        x = self.patch_embed(x)
        # add class token
        cls_tokens = repeat(cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add position embedding
        x += self.pos_embed #TODO

        x = self.dropout(x)

        x = self.blocks(x)
        x = x[:, 0] #take only the class token

        x = self.norm(x)
        x = self.head(x)

        return x

####### Test only #######
if __name__ == '__main__':
    model = ViT(224, 16, 2, 768, 12, 8, 768*4)
    for param in model.state_dict():
        print(param, '\t', model.state_dict()[param].size())
