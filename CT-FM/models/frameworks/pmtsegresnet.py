from monai.networks.nets.segresnet_ds import SegResBlock, SegResEncoder
import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul
from torch.nn import Dropout
from torch import Tensor

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

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
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
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class PromptSegResBLock(SegResBlock):
    """
    A modified version of SegResBlock that includes a prompt input.
    
    This class extends the original SegResBlock from MONAI and adds an additional
    input for prompts. The forward method is overridden to include the prompt input
    in the computation.
    
    Args:
        *args: Variable length argument list passed to SegResBlock
        **kwargs: Arbitrary keyword arguments passed to SegResBlock
    """
    def __init__(self, 
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: tuple | int = 3,
        act: tuple | str = "relu",
        prompt_length: int = 48,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            norm=norm,
            kernel_size=kernel_size,
            act=act,
        )
        self.prompt_length = prompt_length
        self.prompt_tune = True
        if self.prompt_tune:
            patch_size = 128 * 32 // in_channels
            val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size, patch_size), 1) + in_channels))  # noqa
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_length, in_channels))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            self.svanorm = nn.LayerNorm(in_channels)
            self.svaattention = SvaAttention(
                embedding_dim = in_channels,
                num_heads = in_channels // 8,
            )
            self.feats_num = [15, 15, 2]
            self.radiomics_feat_to_tokens = nn.ModuleList([
                nn.Linear(feat_num, in_channels) for feat_num in self.feats_num
            ])

    def forward(self, x, feats : Tensor = None):
        """
        Forward pass of the PromptSegResBLock.
        
        Args:
            x (torch.Tensor): Input tensor.
            prompt (torch.Tensor): Prompt tensor. If None, the block behaves like a standard SegResBlock.
            
        Returns:
            torch.Tensor: Output tensor after applying the block operations.
        """
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
        prompt_tokens = self.svanorm(prompt_tokens)
        B, C, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1).view(B, H * W * D, C)
        x = self.svaattention(x, prompt_tokens, prompt_tokens)
        x = x.view(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        x = x + shortcut_x
        return super().forward(x)

class PromptSegResEncoder_base(SegResEncoder):
    """
    A modified version of SegResEncoder where SegResBlock is replaced by PromptSegResBLock.
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        act: tuple | str = "relu",
        norm: tuple | str = "batch",
        blocks_down: tuple = (1, 2, 2, 4),
        head_module: nn.Module | None = None,
        anisotropic_scales: tuple | None = None,
        prompt_length: int = 48,  # New argument for PromptSegResBLock
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            act=act,
            norm=norm,
            blocks_down=blocks_down,
            head_module=head_module,
            anisotropic_scales=anisotropic_scales,
        )

        # Use PromptSegResBLock instead of SegResBlock
        filters = init_filters  # base number of features
        for i in range(len(blocks_down)):
            level = self.layers[i]  # Access the corresponding level from parent class

            # Replace SegResBlock with PromptSegResBLock
            blocks = [
                PromptSegResBLock(
                    spatial_dims=spatial_dims,
                    in_channels=filters,
                    kernel_size=(3, 3, 3),  # Assuming kernel_size (can customize)
                    norm=norm,
                    act=act,
                    prompt_length=prompt_length,  # Pass prompt_length to the PromptSegResBLock
                )
                for _ in range(blocks_down[i])
            ]
            level["blocks"] = nn.Sequential(*blocks)

            filters *= 2  # Update the filter size for the next level

    def _forward(self, x: torch.Tensor, feats: torch.Tensor = None) -> list[torch.Tensor]:
        outputs = []
        x = self.conv_init(x)

        for level in self.layers:
            for block in level["blocks"]:
                x = block(x, feats)
            outputs.append(x)
            x = level["downsample"](x)

        if self.head_module is not None:
            outputs = self.head_module(outputs)

        return outputs

    def forward(self, x: torch.Tensor, feats: torch.Tensor = None) -> list[torch.Tensor]:
        return self._forward(x, feats)

from huggingface_hub import PyTorchModelHubMixin

class PromptSegResEncoder(PromptSegResEncoder_base, PyTorchModelHubMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

if __name__ == "__main__":
    # Example usage
    block = PromptSegResBLock(
        spatial_dims=3,
        in_channels=32,
        norm="batch",
        kernel_size=(3, 3, 3),
        act="relu",
        prompt_length=48
    )
