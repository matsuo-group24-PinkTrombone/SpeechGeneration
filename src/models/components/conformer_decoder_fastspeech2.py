from typing import Dict, Optional, Sequence, Tuple

import torch

from ...utils.nets_utils import make_non_pad_mask
from .conformer.encoder import Encoder as ConformerEncoder
from .conformer.postnet import Postnet


class ConformerDecoder(torch.nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        adim: int = 384,
        aheads: int = 4,
        layers: int = 6,
        units: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        postnet_dropout_rate: float = 0.5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attn_dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
        reduction_factor: int = 1,
        use_macaron_style: bool = True,
        pos_enc_layer_type: str = "rel_pos",
        self_attn_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel_size: int = 31,
    ):
        super().__init__()
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor

        self.decoder = ConformerEncoder(
            idim=0,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=units,
            num_blocks=layers,
            input_layer=None,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attn_dropout_rate,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style,
            pos_enc_layer_type=pos_enc_layer_type,
            selfattention_layer_type=self_attn_layer_type,
            activation_type=activation_type,
            use_cnn_module=use_cnn_module,
            cnn_module_kernel=cnn_module_kernel_size,
        )
        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )
        )

    def forward(
        self,
        feats: torch.Tensor,
        feats_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            feats (Tensor): Batch of padded target features (B, odim, feats_T).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).

        Returns:
        """
        # VITS posterior_encoderのshapeの違いを吸収
        feats = feats.transpose(1, 2)

        if feats_lengths is not None:
            h_masks = self._source_mask(feats_lengths)
        else:
            h_masks = None

        zs, _ = self.decoder(feats, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, T_feats, odim)

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        return after_outs.transpose(1, 2)  # vitsとの違いを吸収

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)
