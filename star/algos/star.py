import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

import star.utils.tensor_utils as TensorUtils
from star.algos.base import ChunkPolicy
from star.algos.star_modules.mlp import MLP


class STAR(ChunkPolicy):
    def __init__(
        self,
        skill_quantizer=None,
        causal_skill_transformer=None,
        autoencoder=None,
        policy_prior=None,
        stage=0,
        loss_fn=None,
        action_refinement_loss_weight=None,
        first_code_loss_weight=None,
        second_code_loss_weight=None,
        offset_loss_multiplier=None,
        secondary_code_multiplier=None,
        frame_stack=10,
        skill_block_size=5,
        use_cross_entropy_loss=False,
        use_l1_loss=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if skill_quantizer is None:
            skill_quantizer = autoencoder
        if causal_skill_transformer is None:
            causal_skill_transformer = policy_prior
        if skill_quantizer is None or causal_skill_transformer is None:
            raise ValueError("STAR requires both `skill_quantizer` and `causal_skill_transformer`.")

        if action_refinement_loss_weight is None:
            action_refinement_loss_weight = (
                1.0e3 if offset_loss_multiplier is None else offset_loss_multiplier
            )
        if first_code_loss_weight is None:
            first_code_loss_weight = 2.0
        if second_code_loss_weight is None:
            second_code_loss_weight = (
                1.0 if secondary_code_multiplier is None else secondary_code_multiplier
            )
        if use_l1_loss is not None:
            use_cross_entropy_loss = use_l1_loss

        # Preserve legacy module names so older checkpoints keep loading.
        self.autoencoder = skill_quantizer
        self.policy_prior = causal_skill_transformer
        self.stage = stage

        self.frame_stack = frame_stack
        self.skill_block_size = skill_block_size
        self._action_refinement_loss_weight = action_refinement_loss_weight
        self._first_code_loss_weight = first_code_loss_weight
        self._second_code_loss_weight = second_code_loss_weight
        self._criterion = nn.CrossEntropyLoss() if use_cross_entropy_loss else loss_fn
        self.use_cross_entropy_loss = use_cross_entropy_loss

        # Quantization depth, codebook size, and embedding dimension.
        self._G = self.skill_quantizer.quantization_depth
        self._C = self.skill_quantizer.codebook_size
        self._D = self.skill_quantizer.embedding_dim

        self._action_refinement_head = MLP(
            in_channels=self.causal_skill_transformer.n_embd,
            hidden_channels=[
                1024,
                1024,
                self._G * self._C * (self.shape_meta.action_dim * self.skill_block_size),
            ],
        )

        self.n_embd = self.causal_skill_transformer.n_embd
        self.start_token = self._C

    @property
    def skill_quantizer(self):
        return self.autoencoder

    @property
    def causal_skill_transformer(self):
        return self.policy_prior

    @property
    def codebook_size(self):
        return self._C

    @property
    def quantization_depth(self):
        return self._G

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_rarsq_loss(data)
        if self.stage == 1:
            return self.compute_cst_loss(data)
        raise ValueError(f"Unsupported STAR stage: {self.stage}")

    def compute_rarsq_loss(self, data):
        action_input = data["actions"][:, :self.skill_block_size, :]
        reconstructed_actions, total_loss, reconstruction_loss, commitment_loss, codebook_usage = self.skill_quantizer(action_input)
        del reconstructed_actions
        info = {
            "reconstruction_loss": reconstruction_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "codebook_usage": codebook_usage,
        }
        return total_loss, info

    def compute_cst_loss(self, data):
        data = self.preprocess_input(data)

        context = self.get_context(data)
        action_seq = data["actions"]

        _, total_w, _ = action_seq.shape
        act_w = self.skill_quantizer.input_dim_h
        obs_w = total_w + 1 - act_w
        action_seq = action_seq[:, obs_w - 1 :, :]
        batch_size = action_seq.shape[0]

        with torch.no_grad():
            _, target_skill_codes = self.skill_quantizer.get_code(action_seq)

        start_tokens = torch.full(
            (context.shape[0], 1),
            fill_value=self.start_token,
            device=action_seq.device,
            dtype=torch.long,
        )
        input_skill_codes = torch.cat([start_tokens, target_skill_codes[:, :-1]], dim=1)

        predicted_action, decoded_action, sampled_skill_codes, skill_logits = self._predict(
            context, input_skill_codes
        )

        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(0)

        action_refinement_loss = torch.nn.L1Loss()(action_seq, predicted_action)
        quantized_action_loss = torch.nn.L1Loss()(action_seq, decoded_action)

        coarse_skill_loss = self._criterion(skill_logits[:, 0, :], target_skill_codes[:, 0])
        if self._G == 2:
            fine_skill_loss = self._criterion(skill_logits[:, 1, :], target_skill_codes[:, 1])
        elif self._G == 3:
            fine_skill_loss = self._criterion(skill_logits[:, 1, :], target_skill_codes[:, 1])
            extra_skill_loss = self._criterion(skill_logits[:, 2, :], target_skill_codes[:, 2])
        else:
            fine_skill_loss = None

        if self._G == 2:
            skill_prediction_loss = (
                coarse_skill_loss * self._first_code_loss_weight
                + fine_skill_loss * self._second_code_loss_weight
            )
        elif self._G == 1:
            skill_prediction_loss = coarse_skill_loss * self._first_code_loss_weight
        elif self._G == 3:
            skill_prediction_loss = (
                coarse_skill_loss * self._first_code_loss_weight
                + fine_skill_loss * self._second_code_loss_weight
                + extra_skill_loss * self._second_code_loss_weight
            )
        else:
            raise ValueError(f"Unsupported quantization depth: {self._G}")

        exact_skill_match_rate = (
            torch.sum(
                (torch.sum((target_skill_codes == sampled_skill_codes).int(), dim=1) == self._G).int()
            )
            / batch_size
        )
        coarse_skill_match_rate = torch.sum(
            (target_skill_codes[:, 0] == sampled_skill_codes[:, 0]).int()
        ) / batch_size
        if self._G == 2:
            fine_skill_match_rate = torch.sum(
                (target_skill_codes[:, 1] == sampled_skill_codes[:, 1]).int()
            ) / batch_size
        if self._G == 3:
            fine_skill_match_rate = torch.sum(
                (target_skill_codes[:, 1] == sampled_skill_codes[:, 1]).int()
            ) / batch_size
            third_skill_match_rate = torch.sum(
                (target_skill_codes[:, 2] == sampled_skill_codes[:, 2]).int()
            ) / batch_size

        loss = skill_prediction_loss + self._action_refinement_loss_weight * action_refinement_loss
        if self._G == 1:
            info = {
                "coarse_skill_loss": coarse_skill_loss.detach().cpu().item(),
                "skill_prediction_loss": skill_prediction_loss.detach().cpu().item(),
                "action_refinement_loss": action_refinement_loss.detach().cpu().item(),
                "total_loss": loss.detach().cpu().item(),
                "exact_skill_match_rate": exact_skill_match_rate.item(),
                "coarse_skill_match_rate": coarse_skill_match_rate.item(),
                "quantized_action_loss": quantized_action_loss.detach().cpu().item(),
            }
        elif self._G == 2:
            info = {
                "coarse_skill_loss": coarse_skill_loss.detach().cpu().item(),
                "fine_skill_loss": fine_skill_loss.detach().cpu().item(),
                "skill_prediction_loss": skill_prediction_loss.detach().cpu().item(),
                "action_refinement_loss": action_refinement_loss.detach().cpu().item(),
                "total_loss": loss.detach().cpu().item(),
                "exact_skill_match_rate": exact_skill_match_rate.item(),
                "coarse_skill_match_rate": coarse_skill_match_rate.item(),
                "fine_skill_match_rate": fine_skill_match_rate.item(),
                "quantized_action_loss": quantized_action_loss.detach().cpu().item(),
            }
        else:
            info = {
                "coarse_skill_loss": coarse_skill_loss.detach().cpu().item(),
                "fine_skill_loss": fine_skill_loss.detach().cpu().item(),
                "third_skill_loss": extra_skill_loss.detach().cpu().item(),
                "skill_prediction_loss": skill_prediction_loss.detach().cpu().item(),
                "action_refinement_loss": action_refinement_loss.detach().cpu().item(),
                "total_loss": loss.detach().cpu().item(),
                "exact_skill_match_rate": exact_skill_match_rate.item(),
                "coarse_skill_match_rate": coarse_skill_match_rate.item(),
                "fine_skill_match_rate": fine_skill_match_rate.item(),
                "third_skill_match_rate": third_skill_match_rate.item(),
                "quantized_action_loss": quantized_action_loss.detach().cpu().item(),
            }
        return loss, info

    def _decode_skill_codes(self, sampled_skill_codes, cst_context):
        batch_size = sampled_skill_codes.shape[0]
        centers = self.skill_quantizer.draw_code_forward(sampled_skill_codes).view(
            batch_size, -1, self._D
        )
        decoder_input = einops.rearrange(centers.clone().detach(), "B G D -> B (G D)")
        decoded_action = self.skill_quantizer.get_action_from_latent(decoder_input).clone().detach()

        action_refinement = self._action_refinement_head(cst_context)
        action_refinement = einops.rearrange(
            action_refinement, "B (G C WA) -> B G C WA", G=self._G, C=self._C
        )
        indices = (
            torch.arange(batch_size).unsqueeze(1).to(sampled_skill_codes.device),
            torch.arange(self._G).unsqueeze(0).to(sampled_skill_codes.device),
            sampled_skill_codes,
        )
        sampled_refinement = action_refinement[indices].sum(dim=1)
        sampled_refinement = einops.rearrange(
            sampled_refinement,
            "B (W A) -> B W A",
            W=self.skill_quantizer.input_dim_h,
        )
        predicted_action = decoded_action + sampled_refinement
        return predicted_action, decoded_action

    def _predict(self, context, input_skill_codes):
        skill_logits, cst_context = self.causal_skill_transformer(input_skill_codes, context)

        with torch.no_grad():
            probs = torch.softmax(skill_logits, dim=-1)
            sampled_skill_codes = torch.multinomial(probs.view(-1, skill_logits.shape[-1]), 1)
            sampled_skill_codes = sampled_skill_codes.view(-1, skill_logits.shape[1])

        predicted_action, decoded_action = self._decode_skill_codes(
            sampled_skill_codes, cst_context
        )
        return predicted_action, decoded_action, sampled_skill_codes, skill_logits

    def get_optimizers(self):
        if self.stage == 0:
            decay, no_decay = TensorUtils.separate_no_decay(self.skill_quantizer)
        elif self.stage == 1:
            decay, no_decay = TensorUtils.separate_no_decay(self, name_blacklist=("autoencoder",))
        else:
            raise ValueError(f"Unsupported STAR stage: {self.stage}")

        return [
            self.optimizer_factory(params=decay),
            self.optimizer_factory(params=no_decay, weight_decay=0.0),
        ]

    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        context = self.get_context(data)

        sampled_skill_codes = self.causal_skill_transformer.get_indices_top_k(context, self.codebook_size)
        start_tokens = torch.full(
            (context.shape[0], 1),
            fill_value=self.start_token,
            device=context.device,
            dtype=torch.long,
        )
        _, cst_context = self.causal_skill_transformer(start_tokens, context)
        predicted_action, _ = self._decode_skill_codes(sampled_skill_codes, cst_context)

        predicted_action = einops.rearrange(
            predicted_action, "(N T) W A -> N T W A", T=self.frame_stack
        )[:, -1, :, :]
        predicted_action = predicted_action.permute(1, 0, 2)
        return predicted_action.detach().cpu().numpy()

    def get_context(self, data):
        obs_emb = self.obs_encode(data)
        task_emb = self.get_task_emb(data).unsqueeze(1)
        return torch.cat([task_emb, obs_emb], dim=1)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()
