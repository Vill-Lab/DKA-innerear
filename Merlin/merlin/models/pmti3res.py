import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul
from torch.nn import Dropout
from torch import Tensor
from merlin.models.i3res import I3ResNet, Bottleneck3d
import torch.utils.checkpoint as checkpoint

from visualizer import get_local

class SvaAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    @get_local('attn_map')
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # print(q.shape, k.shape, v.shape, self.num_heads)
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn_map = attn
        # print(attn_map.shape)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class pmtI3ResNet(I3ResNet):
    def __init__(self, *args, **kwargs):
        prompt_length = kwargs.pop('prompt_length')
        super().__init__(*args, **kwargs)
        self.pmts = [False, False, False, True]
        self.seperate_num = 1
        if self.pmts[0]:
            self.layer1 = self.inflate_pmtreslayer(args[0].layer1, prompt_length=prompt_length)
        if self.pmts[1]:
            self.layer2 = self.inflate_pmtreslayer(args[0].layer2, prompt_length=prompt_length)
        if self.pmts[2]:
            self.layer3 = self.inflate_pmtreslayer(args[0].layer3, prompt_length=prompt_length)
        if self.pmts[3]:
            self.layer4 = self.inflate_pmtreslayer(args[0].layer4, prompt_length=prompt_length)

    def forward(self, x: Tensor, feats: Tensor = None) -> Tensor:
        feat_map = None
        skips = []
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        # x = checkpoint.checkpoint(self.layer1, x)
        # x = self.layer1(x, feats=feats)
        if self.pmts[0]:
            for i, layer in enumerate(self.layer1):
                if i % self.seperate_num == 0:
                    x = checkpoint.checkpoint(lambda x: layer(x, feats = feats), x)
                else:
                    x = checkpoint.checkpoint(lambda x: layer(x), x)
            # for layer in self.layer1:
            #     x = checkpoint.checkpoint(lambda x: layer(x, feats = feats), x)
        else:
            x = checkpoint.checkpoint(self.layer1, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        # x = checkpoint.checkpoint(self.layer2, x)
        # x = self.layer2(x, feats=feats)
        
        if self.pmts[1]:
            for i, layer in enumerate(self.layer2):
                if i % self.seperate_num == 0:
                    x = checkpoint.checkpoint(lambda x: layer(x, feats = feats), x)
                else:
                    x = checkpoint.checkpoint(lambda x: layer(x), x)
            # for layer in self.layer2:
            #     x = checkpoint.checkpoint(lambda x: layer(x, feats = feats), x)
        else:
            x = checkpoint.checkpoint(self.layer2, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        # x = checkpoint.checkpoint(self.layer3, x)
        # x = self.layer3(x, feats=feats)
        if self.pmts[2]:
            for i, layer in enumerate(self.layer3):
                if i % self.seperate_num == 0:
                    x = checkpoint.checkpoint(lambda x: layer(x, feats = feats), x)
                else:
                    x = checkpoint.checkpoint(lambda x: layer(x), x)
            # for layer in self.layer3:
            #     x = checkpoint.checkpoint(lambda x: layer(x, feats = feats), x)
        else:
            x = checkpoint.checkpoint(self.layer3, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        # x = checkpoint.checkpoint(self.layer4, x)
        # x = self.layer4(x, feats=feats)
        if self.pmts[3]:
            for i, layer in enumerate(self.layer4):
                if i % self.seperate_num == 0:
                    x = checkpoint.checkpoint(lambda x: layer(x, feats = feats), x)
                    # feat_map = x
                    # print(feat_map.shape)
                else:
                    x = checkpoint.checkpoint(lambda x: layer(x), x)
            # for layer in self.layer4:
            #     x = checkpoint.checkpoint(lambda x: layer(x, feats = feats), x)
        else:
            x = checkpoint.checkpoint(self.layer4, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
                    
        if self.conv_class:
            x_features = self.avgpool(x)
            
            if self.ImageEmbedding:
                return x_features.squeeze(2).squeeze(2).squeeze(2).unsqueeze(0)
            
            x_ehr = self.classifier(x_features)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.mean(2)
            x_contrastive = self.contrastive_head(x_features)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.mean(2)
            if self.return_skips:
                return x_contrastive, x_ehr, skips
            else:
                return x_contrastive, x_ehr
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x

    def inflate_pmtreslayer(self, reslayer2d, prompt_length):
        reslayers3d = []
        for i, layer2d in enumerate(reslayer2d):
            if i % self.seperate_num == 0:
                layer3d = pmtBottleneck3d(layer2d, prompt_length)
            else:
                layer3d = Bottleneck3d(layer2d)
            reslayers3d.append(layer3d)
        # for layer2d in reslayer2d:
        #     layer3d = pmtBottleneck3d(layer2d, prompt_length)
        #     reslayers3d.append(layer3d)
        return torch.nn.Sequential(*reslayers3d)

class pmtBottleneck3d(Bottleneck3d):
    def __init__(self, bottleneck2d, prompt_length):
        super().__init__(bottleneck2d)
        in_channels = bottleneck2d.conv1.in_channels
        self.prompt_length = prompt_length
        self.prompt_tune = True
        # exit()
        if self.prompt_tune:
            patch_size = 16
            val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size, patch_size), 1) + in_channels))  # noqa
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_length, in_channels))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            self.svanorm = nn.LayerNorm(in_channels)
            self.svaattention = SvaAttention(
                embedding_dim = in_channels,
                num_heads = in_channels // 8,
                downsample_rate=1
            )
            self.feats_num = [15, 15, 2]
            self.radiomics_feat_to_tokens = nn.ModuleList([
                nn.Linear(feat_num, in_channels) for feat_num in self.feats_num
            ])

    # @get_local('feat_map2')
    def forward(self, x: Tensor, feats: Tensor = None) -> Tensor:
        shortcut_x = x
        feats = [feats[ ... , : 15], feats[ ... , 15 : 30], feats[ ... , 30 : ]]
        feat_tokens = [linear(ft).unsqueeze(1) for ft, linear in zip(feats, self.radiomics_feat_to_tokens)]
        feat_tokens = torch.cat(feat_tokens, dim=1)
        prompt_tokens = self.prompt_embeddings.expand(x.shape[0], -1, -1)
        prompt_tokens = torch.cat([
            prompt_tokens[:, : self.prompt_length // 3, :] + feat_tokens[:, :1, :],
            prompt_tokens[:, self.prompt_length // 3 : self.prompt_length // 3 * 2, :] + feat_tokens[:, 1:2, :],
            prompt_tokens[:, self.prompt_length // 3 * 2 : self.prompt_length, :] + feat_tokens[:, 2:3, :],
        ], dim=1)
        prompt_tokens = torch.cat([prompt_tokens, feat_tokens], dim=1)
        # print(prompt_tokens.shape)
        prompt_tokens = self.svanorm(prompt_tokens)
        B, C, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1).view(B, H * W * D, C)
        x = self.svaattention(x, prompt_tokens, prompt_tokens)
        x = x.view(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        x = x + shortcut_x
        x = super().forward(x)
        return x
