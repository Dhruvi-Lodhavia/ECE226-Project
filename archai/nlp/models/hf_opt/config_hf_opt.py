# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Transformer-XL configurations.
"""

from typing import List, Optional

from archai.nlp.models.config_base import Config, SearchConfigParameter, SearchConfig
from transformers import CONFIG_MAPPING


class HfOPTConfig(Config):
    """Huggingface's Transformer-XL default configuration.

    """

    attribute_map = {
        'n_token': 'vocab_size',
        'tgt_len': 'max_position_embeddings',
        'd_model': 'hidden_size',
        'd_inner': 'ffn_dim',
        'dropatt': 'attention_dropout',
        'n_head' : 'num_attention_heads',
        'n_layer': 'num_hidden_layers'
        # 'weight_init_std': 'initializer_range',
        # 'word_embed_proj_dim': 'hidden_size',
        # 'n_inner': 'ffn_dim'

    }
    attribute_map.update(CONFIG_MAPPING['opt']().attribute_map)

    def __init__(self,
                 n_token: Optional[int] = 10000, # changed from 50257 for model's production
                 d_model: Optional[int] = 768,
                 num_hidden_layers: Optional[int] = 12,
                 d_inner: Optional[int] = 3072,
                 _remove_final_layer_norm: Optional[bool] = False,
                 do_layer_norm: Optional[bool] = True,
                 _remove_final_layer_norm_add: Optional[bool] = False,
                #  word_embed_proj_dim: Optional[bool] = None,
                n_layer: Optional[int] = 12,
                 n_head: Optional[int] = 12,
                 activation_function: Optional[str] = "relu",
                 layerdrop: Optional[float] = 0.0,
                 init_std: Optional[float] = 0.02,
                 tgt_len: Optional[int] = 2048,
                 dropout: Optional[float] = 0.1,
                 dropatt: Optional[float] = 0.0,
                 tie_weight: Optional[bool] = True,
                 pad_token_id: Optional[int] = 0,
                 bos_token_id: Optional[int] = 0,
                 eos_token_id: Optional[int] = 0,
                 **kwargs) -> None:
        """Initializes the class by overriding default arguments.

        Args:
            n_token: Size of the vocabulary (number of tokens).
            tgt_len: Maximum length of sequences (positional embeddings).
            d_model: Dimensionality of the model.
            d_inner: Dimensionality of inner feed-forward layers.
            d_head: Dimensionality of attention heads (`0` for using `d_model` // `n_head`)
            d_embed: Dimensionality of embedding layer (`0` for using same as `d_model`)
            n_layer: Number of layers.
            n_head: Number of attention heads.
            dropout: Dropout probability.
            dropatt: Attention dropout probability.
            div_val: Adaptive embedding/softmax divident.
            pre_lnorm: Whether layer normalization should be performed to input instead of output.
            cutoffs: Cutoffs values for adaptive embedding/softmax.
            mem_len: Maximum length of the memory.
            same_length: Whether every incoming sample should use the same attention length.
            attn_type: Type of attention mechanism (`0` for default attention).
            clamp_len: Uses the same positional embeddings after clamp_len (`0` for no clamp).
            sample_softmax: Number of samples in the sampled softmax (`-1` for disabling).
            adaptive: Whether to use adaptive softmax.
            weight_init_type: Type of weight initialization (`normal` for default).
            weight_init_range: Range to initialize the weights.
            weight_init_std: Standard deviation to initialize the weights.
            proj_init_std: Standard deviation to initialize the projections.
            tie_weight: Whether embedding and softmax weights should be tied.

        """
        # if d_embed == -1:
            # d_embed = d_model

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.n_token = n_token
        self.tgt_len = tgt_len
        # self.word_embed_proj_dim = word_embed_proj_dim
        self.d_inner = d_inner
        self.d_model = d_model
        self.n_layer = n_layer
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.dropatt = dropatt
        self.layerdrop = layerdrop
        self.init_std = init_std
        self.n_head = n_head
        self.activation_function = activation_function
        self._remove_final_layer_norm = _remove_final_layer_norm
        self.tie_weight = tie_weight
        self.do_layer_norm = do_layer_norm
        self._remove_final_layer_norm_add = _remove_final_layer_norm_add

        additional_config = CONFIG_MAPPING['opt']().to_dict()
        for key, value in additional_config.items():
            if key not in self.__dict__.keys():
                setattr(self, key, value)

        
    
        


class HfOPTSearchConfig(SearchConfig):
    """Huggingface's Transformer-XL search configuration.

    """

    def __init__(self) -> None:
        """Initializes the class by setting default parameters that are used during search.
        
        """
        
        # Default HfTransfoXL search options: n_layer, d_model, d_inner and n_head
        n_layer = SearchConfigParameter(per_layer=False, value=[3, 4, 5, 6, 7, 8, 9, 10])
        d_model = SearchConfigParameter(per_layer=False, value=list(range(128, 1024, 64)))
        d_inner = SearchConfigParameter(per_layer=False, value=list(range(128, 4096, 64)))
        n_head = SearchConfigParameter(per_layer=False, value=[2, 4, 8])

        super().__init__(n_layer=n_layer,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head)
