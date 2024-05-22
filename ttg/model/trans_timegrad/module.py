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

from diffusers import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.model import Input, InputSpec
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler, Scaler, StdScaler
from gluonts.torch.util import repeat_along_dim
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..epsilon_theta import EpsilonTheta
from ..transformer import TransformerModel
from ...util import get_lags_for_frequency


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
        # self.lags_seq = [lag - 1 for lag in self.lags_seq]
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
        # model_input_size = (
        #     self.input_size * len(self.lags_seq) + self._number_of_features
        # )

        ########
        nhead = 8
        dim_feedforward_scale = 4
        activation = "gelu"
        # distr_output = StudentTOutput()
        
        self.encoder_decoder = TransformerModel(
            freq=freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward_scale*hidden_size,
            activation=activation,
            dropout=dropout_rate,
            input_size=input_size,
            num_feat_dynamic_real=self.num_feat_dynamic_real,
            num_feat_static_real=self.num_feat_static_real,
            num_feat_static_cat=self.num_feat_static_cat,
            cardinality=cardinality,
            embedding_dimension=self.embedding_dimension,
            # distr_output=distr_output,
            lags_seq=self.lags_seq,
            scaling=scaling,
            default_scale=default_scale,
            num_parallel_samples=self.num_parallel_samples,
        ).type(torch.float64)
        ########

        self.unet = EpsilonTheta(target_dim=input_size, cond_dim=hidden_size).type(torch.float64)
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

        encoder_inputs, loc, scale, static_feat = self.encoder_decoder.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            # future_time_feat[:, :1],
        )
        enc_pos = self.encoder_decoder.pos_embedding(encoder_inputs.size())
        enc_out = self.encoder_decoder.transformer.encoder(self.encoder_decoder.enc_embedding(encoder_inputs) + enc_pos)

        repeated_scale = scale.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_feat.shape[1], -1
        )
        features = torch.cat((expanded_static_feat, future_time_feat), dim=-1)
        repeated_features = features.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        repeated_enc_out = enc_out.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        future_samples = []

        # greedy decoding
        for k in range(self.prediction_length):
            # self.encoder_decoder._check_shapes(repeated_past_target, next_sample, next_features)
            # sequence = torch.cat((repeated_past_target, next_sample), dim=1)

            lagged_sequence = self.encoder_decoder.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )

            decoder_input = torch.cat(
                (reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1
            )

            dec_pos = self.encoder_decoder.pos_embedding(
                decoder_input.size(), past_key_values_length=self.context_length
            )
            output = self.encoder_decoder.transformer.decoder(
                self.encoder_decoder.dec_embedding(decoder_input) + dec_pos, repeated_enc_out
            )

            # params = self.encoder_decoder.param_proj(output[:, -1:])
            # distr = self.encoder_decoder.output_distribution(
            #     params, loc=repeated_loc, scale=repeated_scale
            # )
            # next_sample = distr.sample()

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
#         concat_future_samples = concat_future_samples.reshape(
#             (-1, num_parallel_samples, self.prediction_length) + self.encoder_decoder.target_shape,
#         )
        
        return concat_future_samples
    
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
            noisy_output.type(torch.float64),
            timesteps.type(torch.float64),
            model_output.reshape(B * T, 1, -1).type(torch.float64),
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
            model_output = self.unet(
                sample.type(torch.float64),
                t.type(torch.float64),
                context.view(B * T, 1, -1).type(torch.float64),
            )
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

        transformer_inputs, loc, scale, _ = self.encoder_decoder.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target_reshaped,
            # future_target,
        )

        enc_input = self.encoder_decoder.enc_embedding(
            transformer_inputs[:, : self.context_length, ...]
        )
        enc_pos = self.encoder_decoder.pos_embedding(enc_input.size())
        enc_out = self.encoder_decoder.transformer.encoder(enc_input + enc_pos)

        dec_input = self.encoder_decoder.dec_embedding(
            transformer_inputs[:, self.context_length:, ...]
        )
        dec_pos = self.encoder_decoder.pos_embedding(
            dec_input.size(), past_key_values_length=self.context_length
        )
        dec_output = self.encoder_decoder.transformer.decoder(
            dec_input + dec_pos, enc_out, tgt_mask=self.encoder_decoder.tgt_mask
        )

        # params = self.encoder_decoder.output_params(transformer_inputs)
        # distr = self.encoder_decoder.output_distribution(params, loc=loc, scale=scale)

        # observed_values = (
        #     future_observed_values.all(-1)
        #     if future_observed_values.ndim == 3
        #     else future_observed_values
        # )

        # loss_values = loss(distr, future_target) * observed_values

        if future_only:
            sliced_dec_output = dec_output[:, -self.prediction_length:]
            observed_values = (
                future_observed_reshaped.all(-1)
                if future_observed_reshaped.ndim == 3
                else future_observed_reshaped
            )
            loss_values = self.get_loss_values(
                model_output=sliced_dec_output,
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
                model_output=dec_output,
                loc=loc,
                scale=scale,
                target=target,
                observed_values=observed_values,
            )

        return aggregate_by(loss_values, dim=(1,))