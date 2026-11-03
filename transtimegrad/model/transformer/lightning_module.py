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

from typing import Optional, Union

from gluonts.core.component import validated

from ...serialization import checkpoint_decode, checkpoint_encode
from ..loss import DistributionLoss, NegativeLogLikelihood
from ..rnn_timegrad.lightning_module import TimeGradLightningModule
from .module import TransformerModel


class TransformerLightningModule(TimeGradLightningModule):
    """Shared forecasting training loop with a distribution likelihood loss."""

    model_type = TransformerModel

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        loss: Optional[Union[DistributionLoss, dict]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__(
            model_kwargs=model_kwargs, lr=lr, weight_decay=weight_decay, patience=patience
        )
        self.loss = checkpoint_decode(loss) if loss is not None else NegativeLogLikelihood()
        self.save_hyperparameters({"loss": checkpoint_encode(self.loss)})
