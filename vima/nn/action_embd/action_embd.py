from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import build_mlp


class ActionEmbedding(nn.Module):
    def __init__(self, output_dim: int, *, embed_dict: dict[str, nn.Module]):
        super().__init__()
        self._embed_dict = nn.ModuleDict(embed_dict)
        embed_dict_output_dim = sum(
            embed_dict[k].output_dim for k in sorted(embed_dict.keys())
        )
        self._post_layer = (
            nn.Identity()
            if output_dim == embed_dict_output_dim
            else nn.Linear(embed_dict_output_dim, output_dim)
        )
        self._output_dim = output_dim

        self._input_fields_checked = False

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x_dict: dict[str, torch.Tensor]):
        '''
        input:
            x_dict: dict is self._de_discretize_actions(action)

        output:
            k : 'pos0_position','pos1_position'
            self._embed_dict[k]() is a Linear Layer, then cat them together(for different pos0/1)
            then pass to self._post_layer

            self._post_layer = (
            nn.Identity()
            if output_dim == embed_dict_output_dim
            else nn.Linear(embed_dict_output_dim, output_dim)
        )

        ActionEmbedding(
  (_embed_dict): ModuleDict(
    (pose0_position): ContinuousActionEmbedding(
      (_layer): Sequential(
        (0): Linear(in_features=2, out_features=256, bias=True)
        (1): Identity()
        (2): ReLU(inplace=True)
        (3): Linear(in_features=256, out_features=256, bias=True)
      )
    )
    (pose0_rotation): ContinuousActionEmbedding(
      (_layer): Sequential(
        (0): Linear(in_features=4, out_features=256, bias=True)
        (1): Identity()
        (2): ReLU(inplace=True)
        (3): Linear(in_features=256, out_features=256, bias=True)
      )
    )
    (pose1_position): ContinuousActionEmbedding(
      (_layer): Sequential(
        (0): Linear(in_features=2, out_features=256, bias=True)
        (1): Identity()
        (2): ReLU(inplace=True)
        (3): Linear(in_features=256, out_features=256, bias=True)
      )
    )
    (pose1_rotation): ContinuousActionEmbedding(
      (_layer): Sequential(
        (0): Linear(in_features=4, out_features=256, bias=True)
        (1): Identity()
        (2): ReLU(inplace=True)
        (3): Linear(in_features=256, out_features=256, bias=True)
      )
    )
  )
  (_post_layer): Linear(in_features=1024(256*4), out_features=768, bias=True)
)


        '''
        if not self._input_fields_checked:
            assert set(x_dict.keys()) == set(self._embed_dict.keys())
            self._input_fields_checked = True
        return self._post_layer(
            torch.cat(
                [self._embed_dict[k](x_dict[k]) for k in sorted(x_dict.keys())], dim=-1
            )
        )


class ContinuousActionEmbedding(nn.Module):
    def __init__(
        self, output_dim: int, *, input_dim: int, hidden_dim: int, hidden_depth: int
    ):
        super().__init__()

        self._layer = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor):
        return self._layer(x)
