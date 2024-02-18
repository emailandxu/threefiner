import numpy as np
from kiui.cam import orbit_camera


class SimpleDataset():
    def __init__(self) -> None:
        pass

    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        ver = np.random.randint(-60, 30)
        hor = np.random.randint(-180, 180)
        radius = np.random.uniform() - 0.5 # [-0.5, 0.5]
        pose = orbit_camera(ver, hor, self.opt.radius + radius)
        return ver, hor, radius, pose