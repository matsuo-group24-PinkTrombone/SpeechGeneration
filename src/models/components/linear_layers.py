import torch


class LinearLayers(torch.nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, layers: int, output_size: int, bias: bool = True
    ) -> torch.nn.Module:
        """Linearの幅と深さを変更できるクラス.

        Args:
            input_size (int):
            hidden_size (int):
            layers (int):
            output_size (int):
            bias (bool):
        """
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.hidden_layers = torch.nn.Sequential()

        for i in range(layers):
            self.hidden_layers.add_module(
                f"linear_{i}", torch.nn.Linear(hidden_size, hidden_size, bias=bias)
            )
            self.hidden_layers.add_module(f"ReLU_{i}", torch.nn.ReLU())

        self.output_layer = torch.nn.Linear(hidden_size, output_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor

        Returns:
            output: output tensor
        """

        hidden_first = self.input_layer(x)

        hidden_last = self.hidden_layers(hidden_first)

        output = self.output_layer(hidden_last)

        return output
