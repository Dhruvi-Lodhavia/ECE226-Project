# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugginface's Transformer-XL.
"""

from typing import Dict, Optional, Tuple

import torch
from transformers import CTRLLMHeadModel

from archai.nlp.models.hf_ctrl.config_hf_ctrl import HfCTRLConfig
from archai.nlp.models.model_base import ArchaiModel


class HfCTRL(ArchaiModel):
    """Huggingface's Transformer-XL standard architecture.

    """
    
    def __init__(self, **kwargs) -> None:
        """Initializes the class by creating compatible configuration and model objects.
        
        """
        
        super().__init__()

        self.config = HfCTRLConfig(**kwargs)
        self.model = CTRLLMHeadModel(self.config)

        if self.config.tie_weight:
            self.model.tie_weights()

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mems: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                output_loss: Optional[bool] = True,
                output_prediction_scores: Optional[bool] = False
                ) -> Tuple[torch.Tensor, ...]:
        # Labels are the same as input_ids because they will be shifted inside the model
        outputs = self.model(input_ids=input_ids,
                                 labels=input_ids,
                                 attention_mask=torch.ones_like(input_ids),
                                 past_key_values=past_key_values)
        if output_loss:

            return (outputs.loss, None, None, outputs.past_key_values)

        if output_prediction_scores:
            # outputs = self.model(input_ids=input_ids,
            #                      mems=mems)
                                
            return (None, outputs.logits, None, outputs.past_key_values)

    def get_params(self) -> Dict[str, int]:
        params = {}

        params['embedding'] = self.get_params_from_layer(['Embedding'])
        params['layer_norm'] = self.get_params_from_layer(['LayerNorm'])
        params['attention'] = self.get_params_from_layer(['MultiHeadAttention'])
        params['ff'] = self.get_params_from_layer(['Sequential'])

        params['non_embedding'] = params['layer_norm'] + params['attention'] + params['ff']
        params['total'] = params['non_embedding'] + params['embedding']

        return params

    # def reset_length(self, tgt_len: int, ext_len: int, mem_len: int) -> None:
    #     if tgt_len < 1:
    #         raise ValueError(f'tgt_len: {tgt_len} should be >= 1.')
    #     if ext_len < 0:
    #         raise ValueError(f'ext_len: {ext_len} should be >= 0.')
    #     if mem_len < 0:
    #         raise ValueError(f'mem_len: {mem_len} should be >= 0.')

    #     self.model.config.tgt_len = tgt_len
    #     self.model.config.mem_len = mem_len
    #     self.model.config.ext_len = ext_len
