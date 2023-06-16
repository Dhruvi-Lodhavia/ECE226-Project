# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugginface's Transformer-XL.
"""

from typing import Dict, Optional, Tuple

import torch
from transformers import OPTForCausalLM
import torch.nn.functional as F
from archai.nlp.models.hf_opt.config_hf_opt import HfOPTConfig
from archai.nlp.models.model_base import ArchaiModel


class HfOPT(ArchaiModel):
    """Huggingface's Transformer-XL standard architecture.

    """
    
    def __init__(self, **kwargs) -> None:
        """Initializes the class by creating compatible configuration and model objects.
        
        """
        
        super().__init__()

        self.config = HfOPTConfig(**kwargs)
        self.model = OPTForCausalLM(self.config)

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
        assert mems is None, 'HfGPT2 does not support memory (mems).'
        outputs = self.model(input_ids=input_ids,
                                 labels=input_ids,
                                 attention_mask=torch.ones_like(input_ids),
                                 past_key_values=past_key_values)
                                #  mems=mems)
        if output_loss:
            return (outputs.loss, None, None,outputs.past_key_values)

        if output_prediction_scores:
            outputs = self.model(input_ids=input_ids,
                                 mems=mems)
                                
            return (None, F.log_softmax(outputs.logits, dim=-1), None, outputs.past_key_values)

    def get_params(self) -> Dict[str, int]:
        params = {}

        params['embedding'] = self.get_params_from_layer(['Embedding']) + self.get_params_from_layer(['OPTLearnedPositionalEmbedding'])
        params['attention'] = self.get_params_from_layer(['OPTAttention'])
        params['ff'] = self.get_params_from_layer(['fc1']) +self.get_params_from_layer(['fc2'])
        params['layer_norm'] = self.get_params_from_layer(['final_layer_norm'])

        params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
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
