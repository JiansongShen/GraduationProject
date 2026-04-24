from torch import nn


def _map_norm_layer(norm_type: str, out_ch: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm3d(out_ch)
    elif norm_type == "instance":
        return nn.InstanceNorm3d(out_ch)
    elif norm_type == "group":
        return nn.GroupNorm(out_ch // 8, out_ch)
    else:
        raise ValueError("Unsupported normal type")

def _map_layer_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise ValueError("Unsupported activation type")

def _map_final_activation(final_activation: str) -> nn.Module:
    if final_activation == "sigmoid":
        return nn.Sigmoid()
    elif final_activation == "tanh":
        return nn.Tanh()
    elif final_activation == "softmax":
        return nn.Softmax(dim=1)
    elif final_activation == "none":
        return nn.Identity()

    raise ValueError("Unsupported final activation type")