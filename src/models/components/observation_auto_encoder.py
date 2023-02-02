import torch

from ..abc.observation_auto_encoder import ObservationEncoder as AbsObservationEncoder

class ObservationEncoder(AbsObservationEncoder):
    def __init__(
        self,
    ):
        super().__init__()