"""Conservative online augmentation of lesion-centered MRI and dose crops."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.spatial.transform import Rotation


def uniform(low: float, high: float) -> float:
    """Use the DataLoader worker's seeded PyTorch random stream."""
    return float(torch.empty(()).uniform_(low, high))


@dataclass
class LesionAugmentation:
    """Share rigid geometry across a timeline; perturb only MRI intensities.

    Inputs are CPU float32 CZYX crops on RAS grids. Geometry is relative to
    each crop center, preserving the existing lesion-centered correspondence.
    This does not register follow-ups or simulate a different treatment plan.
    """

    spatial_probability: float = 0.5
    rotation_degrees: float = 180
    translation_mm: float = 2.0
    contrast_probability: float = 0.3
    contrast_range: tuple[float, float] = (0.9, 1.1)
    noise_probability: float = 0.3
    noise_std: float = 0.03
    blur_probability: float = 0.15
    blur_sigma_mm: tuple[float, float] = (0.3, 0.7)

    def __call__(
        self,
        images: list[torch.Tensor],
        dose: torch.Tensor,
        spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        spacing = np.asarray(spacing_xyz[::-1], dtype=np.float64)
        if uniform(0, 1) < self.spatial_probability:
            angles = [
                uniform(-self.rotation_degrees, self.rotation_degrees) for _ in range(3)
            ]
            rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            # scipy maps output to input. Convert XYZ physical rotation to ZYX
            # voxel coordinates, accounting for spacing before interpolation.
            inverse = rotation.T[::-1, ::-1]
            matrix = inverse * spacing[None, :] / spacing[:, None]
            shift = np.array(
                [uniform(-self.translation_mm, self.translation_mm) for _ in range(3)]
            )

            def resample(volume: torch.Tensor) -> torch.Tensor:
                center = (np.asarray(volume.shape[1:]) - 1) / 2
                offset = center - matrix @ center - inverse @ shift / spacing
                return torch.from_numpy(
                    np.stack(
                        [
                            affine_transform(
                                channel,
                                matrix,
                                offset,
                                order=1,
                                mode="constant",
                                cval=0,
                                prefilter=False,
                            )
                            for channel in volume.numpy()
                        ]
                    )
                )

            images = [resample(image) for image in images]
            dose = resample(dose)

        augmented = []
        for image in images:
            # Each acquisition can differ in contrast, resolution and noise.
            # Work out of place so callers and arrays on disk stay unchanged.
            if uniform(0, 1) < self.blur_probability:
                sigma = uniform(*self.blur_sigma_mm) / spacing
                image = torch.from_numpy(
                    gaussian_filter(image.numpy(), sigma=(0, *sigma), mode="nearest")
                )
            if uniform(0, 1) < self.contrast_probability:
                image = image * uniform(*self.contrast_range)
            if uniform(0, 1) < self.noise_probability:
                image = image + torch.randn_like(image) * uniform(0, self.noise_std)
            augmented.append(image)
        return augmented, dose
