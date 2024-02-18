from typing import Any, Dict
import random
import math
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from kiui.cam import orbit_camera
import numpy as np
from itertools import starmap

def get_projection_matrix(
    fovy, aspect_wh: float, near: float, far: float
):
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(
    c2w, proj_mtx
):
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

class MVDataset():
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.n_view = opt.n_view
        self.elevation_range = [0.0, 30.0]
        self.azimuth_range = [-180.0, 180.0]
        self.fovy_range = [15, 65]
        self.camera_distance_range = [0.8, 1.0]
        self.zoom_range = [1.0, 1.0]
        self.relative_radius = True
        self.up_perturb = 0.0
        self.light_distance_range = [0.8, 1.6]
        self.light_sample_strategy = 'dreamfusion'
        self.width = opt.render_resolution
        self.height = opt.render_resolution
        self.camera_perturb = 0.0
        self.center_perturb = 0.0
        self.light_position_perturb = 1.0

    def __len__(self, ):
        return 1000000
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        assert self.batch_size % self.n_view == 0, f"batch_size ({self.batch_size}) must be dividable by n_view ({self.n_view})!"
        real_batch_size = self.batch_size // self.n_view

        # sample elevation angles
        elevation_deg = None
        elevation = None
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg = None
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1,1) + torch.arange(self.n_view).reshape(1,-1)
        ).reshape(-1) / self.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg = (
            torch.rand(real_batch_size)
            * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.n_view, dim=0)
        if self.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom = (
            torch.rand(real_batch_size)
            * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb = (
            torch.rand(real_batch_size, 3) * 2 * self.camera_perturb
            - self.camera_perturb
        ).repeat_interleave(self.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb = (
            torch.randn(real_batch_size, 3) * self.center_perturb
        ).repeat_interleave(self.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb = (
            torch.randn(real_batch_size, 3) * self.up_perturb
        ).repeat_interleave(self.n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances = (
            torch.rand(real_batch_size)
            * (self.light_distance_range[1] - self.light_distance_range[0])
            + self.light_distance_range[0]
        ).repeat_interleave(self.n_view, dim=0)

        if self.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(self.n_view, dim=0) * self.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions = (
                light_direction * light_distances[:, None]
            )
        elif self.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(self.n_view, dim=0)  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(self.n_view, dim=0)  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.light_sample_strategy}"
            )

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # Importance note: the returned rays_d MUST be normalized!
        proj_mtx = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)

        result = {
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "ver": elevation_deg,
            "hor": azimuth_deg,
            "radius": camera_distances + self.opt.radius,
            "height": self.height,
            "width": self.width,
            "fovy": fovy_deg,
        }

        vers, hors, radii = result['ver'], result['hor'], result['radius']
        to_pose = lambda ver, hor, radius: orbit_camera(ver, hor, self.opt.radius + radius)

        poses = np.array(list(starmap(to_pose, zip(vers, hors, radii))))

        return vers, hors, radii, poses


if __name__ == "__main__":
    dataset = MVDataset()
    for batch in dataset:
        print(batch.keys())
        print(batch['camera_positions'])
        break