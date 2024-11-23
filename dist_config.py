import torch 

class DistributionalConfig:
    def __init__(self, num_atoms, v_max, v_min, device):
        self.num_atoms = num_atoms 
        self.v_max = v_max 
        self.v_min = v_min
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, num_atoms, device=device)

