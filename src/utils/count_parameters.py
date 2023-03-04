import torch
from ..models.dreamer import Dreamer
from ..trainer import CheckPointNames

class ComponentNames:
    TRANSITION = "transition"
    PRIOR = "prior"
    OBS_ENCODER = "obs_encoder"
    OBS_DECODER = "obs_decoder"
    CONTROLLER = "controller"
    ALL_COMPONENTS = (TRANSITION, PRIOR, OBS_ENCODER, OBS_DECODER, CONTROLLER)


def count_parameters(param_path:str):
    param_dict = torch.load(param_path)
    model_params = param_dict[CheckPointNames.MODEL]
    num_params = {k: 0 for k in ComponentNames.ALL_COMPONENTS}
    num_params = _count(model_params, num_params)
    return num_params

def _count(model_parameters, counter):
    all_components = list(counter.keys())
    for k, parameters in model_parameters.items():
        for component_name in all_components:
            if component_name in k:
                num = 0
                try:
                    for p in parameters:
                        num += p.numel()
                except TypeError:
                    num += parameters.numel()
                counter[component_name] += num
    return counter

