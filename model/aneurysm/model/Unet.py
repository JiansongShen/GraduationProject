from torch import nn


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "unet"


    def get_name(self) -> str:
        return self.name