from typing import Optional

import torch
from torch import nn
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import SequenceClassifierOutput

from .config import MODEL_NAME, NUM_CLASSES


class VoiceAntiSpoofModel(nn.Module):
    """Wav2Vec2 model fine-tuned for binary voice anti-spoofing."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        freeze_feature_extractor: bool = False,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("VoiceAntiSpoofModel requires exactly two classes")

        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.backbone.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

        if freeze_feature_extractor:
            self.freeze_feature_extractor()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        freeze_feature_extractor: bool = False,
    ) -> "VoiceAntiSpoofModel":
        """Load a pretrained Wav2Vec2 backbone with a new classification head."""
        return cls(
            model_name=model_name,
            num_classes=num_classes,
            freeze_feature_extractor=freeze_feature_extractor,
        )

    def freeze_feature_extractor(self) -> None:
        """Freeze the convolutional Wav2Vec2 feature extractor."""
        self.backbone.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> SequenceClassifierOutput:
        """Return binary classification logits for batched audio waveforms."""
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            feature_mask = self.backbone._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            mask = feature_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled_output = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        logits = self.classifier(self.dropout(pooled_output))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_pretrained_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    freeze_feature_extractor: bool = False,
) -> VoiceAntiSpoofModel:
    """Create a VoiceAntiSpoofModel from a pretrained Wav2Vec2 backbone."""
    return VoiceAntiSpoofModel.from_pretrained(
        model_name=model_name,
        num_classes=num_classes,
        freeze_feature_extractor=freeze_feature_extractor,
    )