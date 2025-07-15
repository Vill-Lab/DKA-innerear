import os

import torch
from torch import nn

from merlin.models.build import MerlinArchitecture
from merlin.utils import download_file


class Merlin(nn.Module):
    def __init__(self, ImageEmbedding: bool = False):
        super(Merlin, self).__init__()
        self.ImageEmbedding = ImageEmbedding
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.local_dir = os.path.join(self.current_path, "checkpoints")
        self.checkpoint_name = (
            "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
        )
        self.repo_id = "stanfordmimi/Merlin"
        self.model = self._load_model()

    """
    Load the Merlin model with the initialized weights
    """

    def _load_model(self):
        self._download_checkpoint()
        model = MerlinArchitecture(ImageEmbedding=self.ImageEmbedding)
        model.load_state_dict(
            torch.load(os.path.join(self.local_dir, self.checkpoint_name))
        )
        return model

    """ 
    Download the Merlin weights from the Hugging Face Hub
    """

    def _download_checkpoint(self):
        download_file(
            repo_id=self.repo_id,
            filename=self.checkpoint_name,
            local_dir=self.local_dir,
        )

    def forward(self, *input):
        return self.model(*input)
    
from merlin.models.build import pmtImageEncoder

class pmtMerlin(Merlin):
    def __init__(self, ImageEmbedding: bool = False, prompt_length: int = 48):
        super(pmtMerlin, self).__init__(ImageEmbedding=ImageEmbedding)
        self.model.encode_image = pmtImageEncoder(ImageEmbedding, prompt_length=prompt_length)
        state_dict = torch.load(
            os.path.join(self.local_dir, self.checkpoint_name), map_location="cpu"
        )
        model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        unique_model_keys = model_keys - checkpoint_keys
        unique_checkpoint_keys = checkpoint_keys - model_keys
        
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, *input, feats):
        return self.model(*input, feats=feats)
