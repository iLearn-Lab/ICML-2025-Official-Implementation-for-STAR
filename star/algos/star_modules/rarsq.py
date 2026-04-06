import einops
import numpy as np
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D

from star.algos.star_modules.rarsq_quantizer import (
    RotationAugmentedResidualSkillQuantizer,
)


class EncoderMLP(nn.Module):
    """A small MLP used by the RaRSQ encoder and decoder."""

    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=128,
        layer_num=1,
        last_activation=None,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, layer_num - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.last_layer = last_activation
        self.apply(weights_init_encoder)

    def forward(self, x):
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer is not None:
            state = self.last_layer(state)
        return state


class RaRSQ(nn.Module):
    def __init__(
        self,
        input_dim_h=1,
        input_dim_w=7,
        n_latent_dims=512,
        codebook_size=None,
        quantization_depth=None,
        vqvae_n_embed=None,
        vqvae_groups=None,
        hidden_dim=128,
        num_layers=1,
        device="cuda",
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
        use_rotation_augmentation=None,
        use_rotation_trick=None,
        use_transformer_decoder=False,
        use_causal_decoder=None,
        use_casual_decoder=None,
    ):
        super().__init__()

        if codebook_size is None:
            codebook_size = 16 if vqvae_n_embed is None else vqvae_n_embed
        if quantization_depth is None:
            quantization_depth = 2 if vqvae_groups is None else vqvae_groups
        if use_rotation_augmentation is None:
            use_rotation_augmentation = True if use_rotation_trick is None else use_rotation_trick
        if use_causal_decoder is None:
            use_causal_decoder = True if use_casual_decoder is None else use_casual_decoder

        self.n_latent_dims = n_latent_dims
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.rep_dim = self.n_latent_dims
        self.codebook_size = codebook_size
        self.quantization_depth = quantization_depth
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        # Legacy aliases for old configs and checkpoint metadata.
        self.vqvae_n_embed = codebook_size
        self.vqvae_groups = quantization_depth
        self.use_rotation_augmentation = use_rotation_augmentation
        self.use_rotation_trick = use_rotation_augmentation
        self.use_transformer_decoder = use_transformer_decoder
        self.use_causal_decoder = use_causal_decoder
        self.use_casual_decoder = use_causal_decoder

        self.vq_layer = RotationAugmentedResidualSkillQuantizer(
            dim=self.n_latent_dims,
            num_quantizers=self.quantization_depth,
            codebook_size=self.codebook_size,
            use_rotation_augmentation=self.use_rotation_augmentation,
        ).to(self.device)
        self.embedding_dim = self.n_latent_dims
        self.vq_layer.device = device

        self.encoder = EncoderMLP(
            input_dim=input_dim_w * self.input_dim_h,
            hidden_dim=self.hidden_dim,
            layer_num=self.num_layers,
            output_dim=n_latent_dims,
        ).to(self.device)

        if self.use_transformer_decoder:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_latent_dims,
                nhead=4,
                dim_feedforward=4 * n_latent_dims,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4).to(self.device)
            self.fixed_positional_emb = PositionalEncoding1D(n_latent_dims)
            self.action_head = nn.Linear(n_latent_dims, input_dim_w)
        else:
            self.decoder = EncoderMLP(
                input_dim=n_latent_dims,
                hidden_dim=self.hidden_dim,
                layer_num=self.num_layers,
                output_dim=input_dim_w * self.input_dim_h,
            ).to(self.device)

    def decode(self, codes, obs_emb=None):
        x = self.fixed_positional_emb(
            torch.zeros(
                (codes.shape[0], self.input_dim_h, self.n_latent_dims),
                dtype=codes.dtype,
                device=codes.device,
            )
        )
        if obs_emb is not None:
            codes = torch.cat([obs_emb, codes], dim=1)
        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.decoder(x, codes.unsqueeze(1), tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(x, codes)
        x = self.action_head(x)
        return einops.rearrange(x, "N T A -> N (T A)")

    def draw_logits_forward(self, encoding_logits):
        return self.vq_layer.draw_logits_forward(encoding_logits)

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent):
        if isinstance(self.decoder, nn.TransformerDecoder):
            output = self.decode(latent) * self.act_scale
        else:
            output = self.decoder(latent) * self.act_scale
        return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if self.input_dim_h == 1:
            state = state.squeeze(-2)
        else:
            state = einops.rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)

    def get_code(self, state, required_recon=False):
        state = state / self.act_scale
        state = self.preprocess(state)
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_rep_shape = state_rep.shape[:-1]
            state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
            state_vq = state_rep_flat.view(*state_rep_shape, -1)
            vq_code = vq_code.view(*state_rep_shape, -1)
            vq_loss_state = torch.sum(vq_loss_state)
            if required_recon:
                recon_state = self.decoder(state_vq) * self.act_scale
                recon_state_ae = self.decoder(state_rep) * self.act_scale
                if self.input_dim_h == 1:
                    return state_vq, vq_code, recon_state, recon_state_ae
                return (
                    state_vq,
                    vq_code,
                    torch.swapaxes(recon_state, -2, -1),
                    torch.swapaxes(recon_state_ae, -2, -1),
                )
            return state_vq, vq_code

    def forward(self, state):
        state = state / self.act_scale
        state = self.preprocess(state)
        state_rep = self.encoder(state)
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(*state_rep_shape, -1)
        vq_code = vq_code.view(*state_rep_shape, -1)
        vq_loss_state = torch.sum(vq_loss_state)

        if self.use_transformer_decoder:
            dec_out = self.decode(state_vq)
        else:
            dec_out = self.decoder(state_vq)
        encoder_loss = (state - dec_out).abs().mean()
        rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
        codebook_usage = len(torch.unique(vq_code)) / self.codebook_size

        return (
            dec_out,
            rep_loss,
            encoder_loss.clone().detach(),
            vq_loss_state.clone().detach(),
            codebook_usage,
        )


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def get_tensor(z, device):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return torch.FloatTensor(z.copy()).to(device).unsqueeze(0)
    return torch.FloatTensor(z.copy()).to(device)
