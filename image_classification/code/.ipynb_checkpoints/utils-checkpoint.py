import numpy as np

def save_torch(name:str, tensor):
    new = tensor.clone()
    np.save(name, new.detach().cpu().numpy())
    
def load_np_torch(name:str):
    return torch.from_numpy(np.load(name))