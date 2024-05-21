# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import math
from typing import List, Optional, Tuple

from diffusers import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.model import Input, InputSpec
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler, Scaler, StdScaler
from gluonts.torch.util import repeat_along_dim, unsqueeze_expand
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...util import get_lags_for_frequency, lagged_sequence_values
from ..epsilon_theta import EpsilonTheta


# # adapation of pytorch documentation
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 5000, mode: str = "forward"):
#         super().__init__()
#         assert mode in ("forward", "backward")
#         self.mode = mode
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pos_enc = torch.zeros(1, max_len, d_model)
#         pos_enc[0, :, 0::2] = torch.sin(position * div_term)
#         pos_enc[0, :, 1::2] = torch.cos(position * div_term)
#         if self.mode == "backward":
#             pos_enc = -torch.flip(pos_enc, (1,))
#         self.register_buffer("pos_enc", pos_enc)

#     def _extend_pos_enc(self, zeros: torch.Tensor) -> None:
#         if self.mode == "forward":
#             self.pos_enc = torch.cat((
#                 self.pos_enc, zeros
#             ), dim=1)
#         elif self.mode == "backward":
#             self.pos_enc = torch.cat((
#                 zeros, self.pos_enc
#             ), dim=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.size(1) > self.pos_enc.size(1): # distant in time positional encodings are zero
#             zeros = torch.zeros((
#                 self.pos_enc.size(0),
#                 x.size(1) - self.pos_enc.size(1),
#                 self.pos_enc.size(2)
#             ), device=self.pos_enc.device)
#             self._extend_pos_enc(zeros)
#         if self.mode == "forward":
#             return x + self.pos_enc[:, :x.size(1)]
#         elif self.mode == "backward":
#             return x + self.pos_enc[:, -x.size(1):]


class Decoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int,
        decoder_layer: "TransformerDecoderLayer", num_layers: int
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
        )
        # self.pos_enc = PositionalEncoding(
        #     d_model=hidden_size,
        #     max_len=5000,
        #     mode="forward",
        # )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt = self.mlp(tgt)
        # tgt = self.pos_enc(tgt)
        return self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )


class TransTimeGradModel(nn.Module):
    """
    Module implementing the TransTimeGrad model.

    Parameters
    ----------
    freq
        String indicating the sampling frequency of the data to be processed.
    context_length
        Length of the Model unrolling prior to the forecast date.
    prediction_length
        Number of time points to predict.
    num_feat_dynamic_real
        Number of dynamic real features that will be provided to ``forward``.
    num_feat_static_real
        Number of static real features that will be provided to ``forward``.
    num_feat_static_cat
        Number of static categorical features that will be provided to
        ``forward``.
    cardinality
        List of cardinalities, one for each static categorical feature.
    embedding_dimension
        Dimension of the embedding space, one for each static categorical
        feature.
    num_layers
        Number of layers in the Model.
    hidden_size
        Size of the hidden layers in the Model.
    dropout_rate
        Dropout rate to be applied at training time.
    lags_seq
        Indices of the lagged observations that the Model takes as input. For
        example, ``[1]`` indicates that the Model only takes the observation at
        time ``t-1`` to produce the output for time ``t``; instead,
        ``[1, 25]`` indicates that the Model takes observations at times ``t-1``
        and ``t-25`` as input.
    scaling
        Whether to apply mean scaling to the observations (target).
    default_scale
        Default scale that is applied if the context length window is
        completely unobserved. If not set, the scale in this case will be
        the mean scale in the batch.
    num_parallel_samples
        Number of samples to produce when unrolling the Model in the prediction
        time range.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        scheduler: SchedulerMixin,
        input_size: int = 1,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        num_feat_static_cat: int = 1,
        cardinality: List[int] = [1],
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        lags_seq: Optional[List[int]] = None,
        scaling: Optional[str] = "mean",
        default_scale: float = 0.0,
        num_parallel_samples: int = 100,
        num_inference_steps: int = 100,
    ) -> None:
        super().__init__()

        assert num_feat_dynamic_real > 0
        assert num_feat_static_real > 0
        assert num_feat_static_cat > 0
        assert len(cardinality) == num_feat_static_cat
        assert (
            embedding_dimension is None
            or len(embedding_dimension) == num_feat_static_cat
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.input_size = input_size

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.lags_seq = [lag - 1 for lag in self.lags_seq]
        self.num_parallel_samples = num_parallel_samples
        self.past_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling == "mean":
            self.scaler: Scaler = MeanScaler(
                dim=1, keepdim=True, default_scale=default_scale
            )
        elif scaling == "std":
            self.scaler: Scaler = StdScaler(dim=1, keepdim=True)
        else:
            self.scaler: Scaler = NOPScaler(dim=1, keepdim=True)
        model_input_size = (
            self.input_size * len(self.lags_seq) + self._number_of_features
        )

        ########
        nhead = 8
        dim_feedforward_to_d_model_ratio = 4
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward_to_d_model_ratio*hidden_size,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.encoder = nn.Sequential(
            nn.Linear(model_input_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # PositionalEncoding(
            #     d_model=hidden_size,
            #     max_len=5000,
            #     mode="backward",
            # ),
            nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            ),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward_to_d_model_ratio*hidden_size,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.decoder = Decoder(
            input_size=self.input_size+self._number_of_features,
            hidden_size=hidden_size,
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )
        ########

        self.unet = EpsilonTheta(target_dim=input_size, cond_dim=hidden_size)
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "feat_static_cat": Input(
                    shape=(batch_size, self.num_feat_static_cat),
                    dtype=torch.long,
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.num_feat_static_real),
                    dtype=torch.float,
                ),
                "past_time_feat": Input(
                    shape=(
                        batch_size,
                        self._past_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "past_target": Input(
                    shape=(batch_size, self._past_length)
                    if self.input_size == 1
                    else (batch_size, self._past_length, self.input_size),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self._past_length)
                    if self.input_size == 1
                    else (batch_size, self._past_length, self.input_size),
                    dtype=torch.float,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            },
            zeros_fn=torch.zeros,
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + self.input_size * 2  # the log(scale) and log1p(abs(loc))
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def prepare_model_input(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
        context = past_target[:, -self.context_length:]
        observed_context = past_observed_values[:, -self.context_length:]

        input, loc, scale = self.scaler(context, observed_context)
        future_length = future_time_feat.shape[-2]
        if future_length > 1:
            assert future_target is not None
            input = torch.cat(
                (input, (future_target[:, : future_length - 1, ...] - loc) / scale),
                dim=1,
            )
        prior_input = (past_target[:, : -self.context_length, ...] - loc) / scale

        lags = lagged_sequence_values(self.lags_seq, prior_input, input, dim=1)
        time_feat = torch.cat(
            (
                past_time_feat[:, -self.context_length + 1:, ...],
                future_time_feat,
            ),
            dim=1,
        )

        embedded_cat = self.embedder(feat_static_cat)
        log_abs_loc = (
            loc.abs().log1p() if self.input_size == 1 else loc.squeeze(1).abs().log1p()
        )
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()

        static_feat = torch.cat(
            (embedded_cat, feat_static_real, log_abs_loc, log_scale),
            dim=-1,
        )
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=1, size=time_feat.shape[-2]
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        return torch.cat((lags, features), dim=-1), loc, scale, static_feat

    def unroll_lagged_model(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Applies the underlying Model to the provided target data and covariates.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            Tensor of dynamic real features in the future,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        future_target
            (Optional) Tensor of future target values,
            shape: ``(batch_size, prediction_length)``.

        Returns
        -------
        Tuple
            A tuple containing, in this order:
            - Parameters of the output distribution
            - Scaling factor applied to the target
            - Raw output of the Model
            - Static input to the Model
            - (Optional) Output state from the Model
        """
        model_input, loc, scale, static_feat = self.prepare_model_input(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        encoder_output = self.encoder(model_input)

        return loc, scale, encoder_output, static_feat

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Invokes the model on input data, and produce outputs future samples.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        num_parallel_samples
            (Optional) How many future samples to produce.
            By default, self.num_parallel_samples is used.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        loc, scale, encoder_output, static_feat = self.unroll_lagged_model(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        repeated_encoder_output = encoder_output.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ) # decoder looks at all encoder states or only at the last one?

        next_sample = self.sample(
            repeated_encoder_output[:, -1:], loc=repeated_loc, scale=repeated_scale
        ) # BOS token of decoded sequence
        next_samples = torch.clone(next_sample)

        next_features = torch.zeros((
            repeated_static_feat.size(0), 0, self._number_of_features
        ), device=encoder_output.device)

        feature_samples = torch.zeros((
            next_samples.size(0), 0, self.input_size + self._number_of_features
        ), device=encoder_output.device)

        for k in range(1, self.prediction_length):
            next_feature = torch.cat((
                repeated_static_feat, repeated_time_feat[:, k: k + 1]
            ), dim=-1)
            next_features = torch.cat((next_features, next_feature), dim=1)

            feature_sample = torch.cat((
                (next_samples[:, k - 1:] - repeated_loc) / repeated_scale, next_features[:, k - 1:]
            ), dim=-1)
            feature_samples = torch.cat((
                feature_samples, feature_sample
            ), dim=1)

            target_mask = nn.Transformer(). \
                generate_square_subsequent_mask(feature_samples.size(1)). \
                to(encoder_output.device)
            decoder_output = self.decoder(
                tgt=feature_samples,
                memory=repeated_encoder_output,
                tgt_mask=target_mask,
            )

            next_sample = self.sample(decoder_output[:, -1:], loc=repeated_loc, scale=repeated_scale)
            next_samples = torch.cat((next_samples, next_sample), dim=1)

        next_samples = next_samples.reshape(
            (-1, num_parallel_samples, self.prediction_length, self.input_size)
        )

        return next_samples.squeeze(-1)

    def get_loss_values(self, model_output, loc, scale, target, observed_values):
        B, T = model_output.shape[:2]
        # Sample a random timestep for each sample in the batch
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (B * T,),
            device=model_output.device,
        ).long()
        noise = torch.randn(target.shape, device=target.device)

        scaled_target = (target - loc) / scale

        noisy_output = self.scheduler.add_noise(
            scaled_target.view(B * T, 1, -1), noise.view(B * T, 1, -1), timesteps
        )

        model_output = self.unet(
            noisy_output, timesteps, model_output.reshape(B * T, 1, -1)
        )
        if self.scheduler.config.prediction_type == "epsilon":
            target_noise = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target_noise = self.scheduler.get_velocity(
                scaled_target.view(B * T, 1, -1),
                noise.view(B * T, 1, -1),
                timesteps,
            )

        return (
            F.smooth_l1_loss(
                model_output.view(B, T, -1),
                target_noise.view(B, T, -1),
                reduction="none",
            ).mean(-1)
            * observed_values
        )

    def sample(self, context, loc, scale):
        # context [B, T, H]
        # loc [B, 1, D]
        # scale [B, 1, D]
        B, T = context.shape[:2]
        sample_shape = (B * T, 1, self.input_size)
        sample = randn_tensor(sample_shape, device=context.device)

        self.scheduler.set_timesteps(self.num_inference_steps)
        for t in self.scheduler.timesteps:
            model_output = self.unet(sample, t, context.view(B * T, 1, -1))
            sample = self.scheduler.step(model_output, t, sample).prev_sample

        return (sample.view(B, T, -1) * scale) + loc

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        future_only: bool = True,
        aggregate_by=torch.mean,
    ) -> torch.Tensor:
        extra_dims = len(future_target.shape) - len(past_target.shape)
        extra_shape = future_target.shape[:extra_dims]
        # batch_shape = future_target.shape[: extra_dims + 1]

        repeats = prod(extra_shape)
        feat_static_cat = repeat_along_dim(feat_static_cat, 0, repeats)
        feat_static_real = repeat_along_dim(feat_static_real, 0, repeats)
        past_time_feat = repeat_along_dim(past_time_feat, 0, repeats)
        past_target = repeat_along_dim(past_target, 0, repeats)
        past_observed_values = repeat_along_dim(past_observed_values, 0, repeats)
        future_time_feat = repeat_along_dim(future_time_feat, 0, repeats)

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1:],
        )
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1:],
        )

        loc, scale, encoder_output, static_feat = self.unroll_lagged_model(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target_reshaped,
        )
        repeated_static_feat = static_feat.unsqueeze(1).repeat_interleave(self.prediction_length, dim=1)

        next_features = torch.cat((
            repeated_static_feat, future_time_feat
        ), dim=-1)
        feature_samples = torch.cat((
            (future_target_reshaped - loc) / scale, next_features
        ), dim=-1)
        
        target_mask = nn.Transformer(). \
            generate_square_subsequent_mask(feature_samples.size(1)). \
            to(encoder_output.device)
        decoder_output = self.decoder(
            tgt=feature_samples,
            memory=encoder_output,
            tgt_mask=target_mask,
        )

        if future_only:
            sliced_decoder_output = decoder_output[:, -self.prediction_length:]
            observed_values = (
                future_observed_reshaped.all(-1)
                if future_observed_reshaped.ndim == 3
                else future_observed_reshaped
            )
            loss_values = self.get_loss_values(
                model_output=sliced_decoder_output,
                loc=loc,
                scale=scale,
                target=future_target_reshaped,
                observed_values=observed_values,
            )
        else:
            context_target = past_target[:, -self.context_length + 1:]
            target = torch.cat((context_target, future_target_reshaped), dim=1)
            context_observed = past_observed_values[:, -self.context_length + 1:]
            observed_values = torch.cat(
                (context_observed, future_observed_reshaped), dim=1
            )
            observed_values = (
                observed_values.all(-1)
                if observed_values.ndim == 3
                else observed_values
            )

            loss_values = self.get_loss_values(
                model_output=decoder_output,
                loc=loc,
                scale=scale,
                target=target,
                observed_values=observed_values,
            )

        return aggregate_by(loss_values, dim=(1,))
