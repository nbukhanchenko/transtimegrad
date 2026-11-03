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

from typing import List, Optional

import torch
import torch.nn as nn
from diffusers import SchedulerMixin
from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.model import InputSpec
from gluonts.torch.util import repeat_along_dim

from ...modules import StudentTOutput
from ...util import get_lags_for_frequency
from ..diffusion import (
    auxiliary_context_losses,
    diffusion_loss,
    diffusion_sample,
    validate_features,
    validate_model_config,
    validate_scheduler,
)
from ..epsilon_theta import EpsilonTheta
from ..transformer import TransformerModel


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
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        lags_seq: Optional[List[int]] = None,
        scaling: Optional[str] = "mean",
        default_scale: Optional[float] = None,
        num_parallel_samples: int = 100,
        num_inference_steps: int = 100,
    ) -> None:
        super().__init__()
        cardinality = [1] if cardinality is None else list(cardinality)

        validate_features(
            num_feat_dynamic_real,
            num_feat_static_real,
            num_feat_static_cat,
            cardinality,
            embedding_dimension,
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
        self.lags_seq = (
            get_lags_for_frequency(freq_str=freq) if lags_seq is None else list(lags_seq)
        )
        validate_model_config(
            context_length,
            prediction_length,
            input_size,
            self.lags_seq,
            scaling,
            num_parallel_samples,
        )
        self.num_parallel_samples = num_parallel_samples
        self.past_length = self.context_length + max(self.lags_seq)
        self.encoder_decoder = TransformerModel(
            freq=freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_model=hidden_size,
            nhead=8,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * hidden_size,
            activation="gelu",
            dropout=dropout_rate,
            input_size=input_size,
            distr_output=StudentTOutput(dim=input_size) if input_size > 1 else StudentTOutput(),
            num_feat_dynamic_real=self.num_feat_dynamic_real,
            num_feat_static_real=self.num_feat_static_real,
            num_feat_static_cat=self.num_feat_static_cat,
            cardinality=cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=scaling,
            default_scale=default_scale,
            num_parallel_samples=self.num_parallel_samples,
        )

        self.unet = EpsilonTheta(
            target_dim=input_size,
            cond_dim=hidden_size,
            max_steps=scheduler.config.num_train_timesteps,
        )
        self.scheduler = validate_scheduler(scheduler, num_inference_steps)
        self.num_inference_steps = num_inference_steps

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return self.encoder_decoder.describe_inputs(batch_size)

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

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
        past_target = torch.where(past_observed_values.bool(), past_target, 0.0)
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples
        if num_parallel_samples < 1:
            raise ValueError("num_parallel_samples must be positive")

        encoder_inputs, loc, scale, static_feat = self.encoder_decoder.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            # future_time_feat[:, :1],
        )
        enc_pos = self.encoder_decoder.pos_embedding(encoder_inputs.size())
        enc_out = self.encoder_decoder.transformer.encoder(
            self.encoder_decoder.enc_embedding(encoder_inputs) + enc_pos
        )

        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, future_time_feat.shape[1], -1)
        features = torch.cat((expanded_static_feat, future_time_feat), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_enc_out = enc_out.repeat_interleave(repeats=num_parallel_samples, dim=0)

        future_samples = []

        # greedy decoding
        for k in range(self.prediction_length):
            lagged_sequence = self.encoder_decoder.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

            decoder_input = torch.cat(
                (reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1
            )

            dec_pos = self.encoder_decoder.pos_embedding(
                decoder_input.size(), past_key_values_length=self.context_length
            )
            output = self.encoder_decoder.transformer.decoder(
                self.encoder_decoder.dec_embedding(decoder_input) + dec_pos,
                repeated_enc_out,
                tgt_mask=self.encoder_decoder.tgt_mask[: k + 1, : k + 1],
            )

            #     params, loc=repeated_loc, scale=repeated_scale

            next_sample = self.sample(output[:, -1:], loc=repeated_loc, scale=repeated_scale)

            repeated_past_target = torch.cat(
                (repeated_past_target, (next_sample - repeated_loc) / repeated_scale),
                dim=1,
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        concat_future_samples = concat_future_samples.reshape(
            (-1, num_parallel_samples, self.prediction_length, self.input_size)
        ).squeeze(-1)

        return concat_future_samples

    def get_loss_values(self, model_output, loc, scale, target, observed_values):
        return diffusion_loss(
            self.unet, self.scheduler, model_output, loc, scale, target, observed_values
        )

    def sample(self, context, loc, scale):
        return diffusion_sample(
            self.unet,
            self.scheduler,
            context,
            loc,
            scale,
            self.input_size,
            self.num_inference_steps,
        )

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
        # A one-step context has no auxiliary targets.
        future_only = future_only or self.context_length == 1
        future_target = torch.where(future_observed_values.bool(), future_target, 0.0)
        extra_dims = len(future_target.shape) - len(past_target.shape)
        extra_shape = future_target.shape[:extra_dims]

        repeats = prod(extra_shape)
        feat_static_cat = repeat_along_dim(feat_static_cat, 0, repeats)
        feat_static_real = repeat_along_dim(feat_static_real, 0, repeats)
        past_time_feat = repeat_along_dim(past_time_feat, 0, repeats)
        past_target = repeat_along_dim(past_target, 0, repeats)
        past_observed_values = repeat_along_dim(past_observed_values, 0, repeats)
        future_time_feat = repeat_along_dim(future_time_feat, 0, repeats)

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )

        transformer_inputs, loc, scale, _ = self.encoder_decoder.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target_reshaped,
        )

        enc_input = self.encoder_decoder.enc_embedding(transformer_inputs[:, : self.context_length])
        enc_pos = self.encoder_decoder.pos_embedding(enc_input.size())
        # The primary forecast path is identical for both loss modes and inference.
        enc_out = self.encoder_decoder.transformer.encoder(enc_input + enc_pos)
        future_inputs = transformer_inputs[:, self.context_length :]
        future_emb = self.encoder_decoder.dec_embedding(future_inputs)
        future_pos = self.encoder_decoder.pos_embedding(future_emb.size(), self.context_length)
        dec_output = self.encoder_decoder.transformer.decoder(
            future_emb + future_pos, enc_out, tgt_mask=self.encoder_decoder.tgt_mask
        )
        future_output = dec_output[:, -self.prediction_length :]
        loss_values = self.get_loss_values(
            model_output=future_output,
            loc=loc,
            scale=scale,
            target=future_target_reshaped,
            observed_values=future_observed_reshaped,
        )
        if not future_only and self.context_length > 1:
            context_losses = auxiliary_context_losses(
                self,
                feat_static_cat,
                feat_static_real,
                past_time_feat,
                past_target,
                past_observed_values,
            )
            loss_values = torch.cat((*context_losses, loss_values), dim=1)
        return aggregate_by(loss_values, dim=1)

    def _one_step_conditioning(
        self,
        feat_static_cat,
        feat_static_real,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
    ):
        past_target = torch.where(past_observed_values.bool(), past_target, 0.0)
        backbone = self.encoder_decoder
        inputs, loc, scale, static = backbone.create_network_inputs(
            feat_static_cat, feat_static_real, past_time_feat, past_target, past_observed_values
        )
        memory = backbone.transformer.encoder(
            backbone.enc_embedding(inputs) + backbone.pos_embedding(inputs.size())
        )
        lags = backbone.get_lagged_subsequences(past_target.sub(loc).div(scale), 1, shift=1)
        features = torch.cat((static.unsqueeze(1), future_time_feat), dim=-1)
        decoder_input = torch.cat((lags.reshape(lags.shape[0], 1, -1), features), dim=-1)
        positions = backbone.pos_embedding(decoder_input.size(), self.context_length)
        output = backbone.transformer.decoder(
            backbone.dec_embedding(decoder_input) + positions, memory
        )
        return loc, scale, output
