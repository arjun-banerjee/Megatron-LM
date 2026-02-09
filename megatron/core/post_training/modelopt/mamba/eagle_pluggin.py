# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""EAGLE Speculative Decoding Plugin for MambaModel.

This module registers MambaModel with ModelOpt's EagleDMRegistry to enable
EAGLE speculative decoding for hybrid Mamba models.
"""

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.transformer.transformer_config import TransformerConfig


def _is_modelopt_available():
    """Check if modelopt is available."""
    try:
        import modelopt.torch.speculative as mtsp
        from modelopt.torch.speculative.eagle import EagleModel, EagleDMRegistry
        return True
    except ImportError:
        return False


def register_mamba_eagle_plugin():
    """Register MambaModel with the EagleDMRegistry for EAGLE speculative decoding.

    This function should be called before mtsp.convert() to enable EAGLE support
    for MambaModel (hybrid Mamba architectures).
    """
    if not _is_modelopt_available():
        raise ImportError(
            "modelopt is not available. Please install nvidia-modelopt to use EAGLE "
            "with MambaModel."
        )

    from modelopt.torch.speculative.eagle import EagleModel, EagleDMRegistry

    # Check if MambaModel is already registered
    try:
        _ = EagleDMRegistry[MambaModel]
        # Already registered, skip
        return
    except KeyError:
        pass

    @EagleDMRegistry.register({MambaModel: "megatron.core.models.mamba.MambaModel"})
    class _DynamicEagleMambaModel(EagleModel):
        """Dynamic EAGLE-enhanced MambaModel for speculative decoding.

        This class wraps MambaModel to add EAGLE speculative decoding capabilities,
        including draft token prediction and auxiliary hidden state extraction for
        EAGLE-3 variants.
        """

        def _setup(self) -> List[str]:
            """Set up temporary attributes for EAGLE configuration."""
            return [
                "eagle_freeze_base_model",
                "eagle_architecture_config",
                "eagle_offline",
                "eagle_hidden_state_distillation",
                "eagle_self_logit_distillation",
                "eagle_report_acc",
                "eagle_reuse_base_decoder",
                "eagle_loss_decay_factor",
                "eagle_decoder_type",
            ]

        def modify(
            self,
            eagle_offline: bool = False,
            eagle_hidden_state_distillation: bool = False,
            eagle_self_logit_distillation: bool = False,
            eagle_freeze_base_model: bool = True,
            eagle_report_acc: bool = False,
            eagle_reuse_base_decoder: bool = False,
            eagle_loss_decay_factor: float = 1.0,
            eagle_architecture_config: Optional[Dict[str, Any]] = None,
            eagle_decoder_type: Optional[str] = None,
        ) -> None:
            """Modify the MambaModel to add EAGLE module.

            Args:
                eagle_offline: Whether to use offline EAGLE mode.
                eagle_hidden_state_distillation: Enable hidden state distillation.
                eagle_self_logit_distillation: Enable self logit distillation.
                eagle_freeze_base_model: Whether to freeze base model parameters.
                eagle_report_acc: Whether to report accuracy metrics.
                eagle_reuse_base_decoder: Whether to reuse base decoder.
                eagle_loss_decay_factor: Loss decay factor.
                eagle_architecture_config: Configuration for the EAGLE architecture.
                eagle_decoder_type: Type of decoder to use.
            """
            from modelopt.torch.speculative.plugins.megatron_eagle import (
                EagleModule,
                dict_to_config,
            )

            if self.config.pipeline_model_parallel_size > 1:
                warnings.warn(
                    "Pipeline parallelism detected! _DynamicEagleMambaModel only supports "
                    "pipeline parallelism during TensorRT-LLM checkpoint export."
                )

            # Enable heterogeneous checkpoint if available
            if hasattr(self.config, "hetereogenous_dist_checkpoint"):
                self.config.hetereogenous_dist_checkpoint = True

            # Call parent modify to store base attributes
            super().modify(
                eagle_offline=eagle_offline,
                eagle_hidden_state_distillation=eagle_hidden_state_distillation,
                eagle_self_logit_distillation=eagle_self_logit_distillation,
                eagle_freeze_base_model=eagle_freeze_base_model,
                eagle_report_acc=eagle_report_acc,
                eagle_reuse_base_decoder=eagle_reuse_base_decoder,
                eagle_loss_decay_factor=eagle_loss_decay_factor,
                eagle_architecture_config=eagle_architecture_config,
                eagle_decoder_type=eagle_decoder_type,
            )

            # Store additional attributes
            self.eagle_architecture_config = eagle_architecture_config or {}

            # Sequence parallel is not used in offline eagle
            if self.eagle_offline:
                self.config.sequence_parallel = False

            # Create EAGLE config from architecture config
            self.eagle_config = dict_to_config(
                eagle_architecture_config,
                self.config.use_cpu_initialization,
                self.config.fp16,
                self.config.bf16,
                self.config.sequence_parallel,
            )
            self.eagle_config.hidden_size = self.config.hidden_size
            self.eagle_config.vocab_size = self.vocab_size
            self.eagle_config.max_sequence_length = self.max_sequence_length
            self.eagle_config.draft_vocab_size = (
                self.vocab_size
                if self.eagle_config.draft_vocab_size is None
                else self.eagle_config.draft_vocab_size
            )

            if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                assert eagle_self_logit_distillation, (
                    "Only logit distillation is supported when draft_vocab_size != vocab_size!"
                )

            # Use default aux_hidden_state layers if use_aux_hidden_state is True
            # but no layer id is given
            if (
                self.eagle_config.use_aux_hidden_state
                and len(self.eagle_config.eagle_aux_hidden_state_layer_ids) == 0
            ):
                self._set_default_aux_hidden_state_layers()

            if len(self.eagle_config.eagle_aux_hidden_state_layer_ids) > 0:
                assert not self.eagle_hidden_state_distillation, (
                    "EAGLE-3 does not support hidden state distillation!"
                )

            # EAGLE-3 auxiliary hidden_states
            self._aux_hidden_states = []

            if self.eagle_config.position_embedding_type not in ["rope", "yarn"]:
                raise ValueError("For EAGLE, only RoPE or YaRN embedding are supported")

            # Register forward hooks to extract aux hidden_states
            if len(self.eagle_config.eagle_aux_hidden_state_layer_ids) > 0:
                if hasattr(self, 'decoder') and hasattr(self.decoder, 'layers'):
                    for layer in self.decoder.layers:
                        layer.register_forward_hook(self._transformer_layer_forward_hook)

            # Freeze all parameters
            if self.eagle_freeze_base_model:
                for name, param in self.named_parameters():
                    param.requires_grad = False

            # Only the last PP stage has the additional projection and decoder layer
            if self.post_process:
                # Get rotary embeddings - use existing if available (hybrid Mamba models)
                if hasattr(self, 'rotary_pos_emb') and self.rotary_pos_emb is not None:
                    rotary_pos_emb = self.rotary_pos_emb
                else:
                    # Create new rotary embeddings if model doesn't have them
                    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
                        RotaryEmbedding,
                    )
                    rotary_pos_emb = RotaryEmbedding(
                        kv_channels=self.eagle_config.kv_channels,
                        rotary_percent=self.eagle_config.rotary_percent,
                        rotary_interleaved=False,
                        seq_len_interpolation_factor=None,
                        rotary_base=self.eagle_config.rotary_base,
                        rope_scaling=getattr(self.eagle_config, 'rope_scaling', None),
                        rope_scaling_factor=getattr(self.eagle_config, 'rope_scaling_factor', 1.0),
                        use_cpu_initialization=self.eagle_config.use_cpu_initialization,
                    )

                if self.eagle_reuse_base_decoder:
                    eagle_module_config = copy.deepcopy(self.config)
                    # Overwrite values from the eagle config
                    eagle_module_config.num_layers = self.eagle_config.num_layers
                    eagle_module_config.use_last_layernorm = self.eagle_config.use_last_layernorm
                    eagle_module_config.use_input_layernorm_in_first_layer = (
                        self.eagle_config.use_input_layernorm_in_first_layer
                    )
                    eagle_module_config.eagle_aux_hidden_state_layer_ids = (
                        self.eagle_config.eagle_aux_hidden_state_layer_ids
                    )
                    eagle_module_config.use_mtp_layernorm = self.eagle_config.use_mtp_layernorm
                    eagle_module_config.draft_vocab_size = self.eagle_config.draft_vocab_size
                    eagle_module_config.has_lm_head = self.eagle_config.has_lm_head

                    self.eagle_module = EagleModule(
                        eagle_module_config,
                        rotary_pos_emb,
                        bias=False,
                    )
                else:
                    self.eagle_module = EagleModule(
                        self.eagle_config,
                        rotary_pos_emb,
                        bias=False,
                    )

        def _set_default_aux_hidden_state_layers(self) -> None:
            """Set default auxiliary hidden state layer indices."""
            # Use evenly spaced layers from the decoder
            if hasattr(self, 'decoder') and hasattr(self.decoder, 'layers'):
                num_layers = len(self.decoder.layers)
                # Default: use layers at 1/4, 1/2, 3/4 of the model
                default_layers = [
                    num_layers // 4,
                    num_layers // 2,
                    3 * num_layers // 4,
                ]
                self.eagle_config.eagle_aux_hidden_state_layer_ids = default_layers

        def _transformer_layer_forward_hook(
            self,
            module: nn.Module,
            input: Tuple[Tensor, ...],
            output: Union[Tensor, Tuple[Tensor, ...]],
        ) -> None:
            """Hook to capture auxiliary hidden states from transformer/mamba layers."""
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self._aux_hidden_states.append(hidden_states.detach())

        def _clear_auxiliary_states(self) -> None:
            """Clear cached auxiliary hidden states."""
            self._aux_hidden_states = []


def unregister_mamba_eagle_plugin():
    """Unregister MambaModel from EagleDMRegistry if needed."""
    if not _is_modelopt_available():
        return

    from modelopt.torch.speculative.eagle import EagleDMRegistry

    try:
        del EagleDMRegistry[MambaModel]
    except KeyError:
        pass