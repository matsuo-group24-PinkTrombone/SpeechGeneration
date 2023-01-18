import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.components.conformer.swish import Swish

def test_swish():
    """
    グラフにプロットして確認する
    """
    # plot
    x = torch.tensor([x*0.1 for x in range(-100,100)],dtype=torch.float32)
    swish = Swish()
    y = swish(x)
    plt.plot(x,y)

    # save figure
    dirpath = Path(__file__).parent / "test_plot"
    dirpath.mkdir(exist_ok=True,parents=True)
    filepath = dirpath / "swish.png"
    plt.savefig(filepath)
