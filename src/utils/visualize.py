import numpy as np
import matplotlib.pyplot as plt
from torch.utils import SummaryWriter

def make_spectrogram_figure(
    target: np.ndarray,
    generated: np.ndarray,
    predicted_generated: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1)
    fig.tight_layout()
    labels = {"xlabel": "timestamp", "ylabel": "Hz"}

    data = {
        "Target": target,
        "Generated": generated,
        "Predicted Generated": predicted_generated,
    }
    # show target mel spectrogram
    for i, (title, spect) in enumerate(data.items()):
        mappable = axes[i].imshow(spect, aspect="auto")
        axes[i].set(**labels, title=title)
        fig.colorbar(mappable, ax=axes[i])
    
    return fig

