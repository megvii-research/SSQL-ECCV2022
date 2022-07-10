import math
import torch


def get_state_dict(path):
    state_dict = torch.load(path, map_location="cpu")
    if state_dict.get("state_dict", None):
        state_dict = state_dict["state_dict"]
    return state_dict
