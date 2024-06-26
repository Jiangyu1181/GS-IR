import os

import numpy as np
import nvdiffrast.torch as dr
import torch
from typing import Dict, Optional, Union

from .light import CubemapLight


# Lazarov 2013, "Getting More Physical in Call of Duty: Black Ops II"
# https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
def envBRDF_approx(roughness: torch.Tensor, NoV: torch.Tensor) -> torch.Tensor:
    c0 = torch.tensor([-1.0, -0.0275, -0.572, 0.022], device=roughness.device)
    c1 = torch.tensor([1.0, 0.0425, 1.04, -0.04], device=roughness.device)
    c2 = torch.tensor([-1.04, 1.04], device=roughness.device)
    r = roughness * c0 + c1
    a004 = (
            torch.minimum(torch.pow(r[..., (0,)], 2), torch.exp2(-9.28 * NoV)) * r[..., (0,)]
            + r[..., (1,)]
    )
    AB = (a004 * c2 + r[..., 2:]).clamp(min=0.0, max=1.0)
    return AB


def saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=1e-4, max=1.0)


# Tone Mapping
def aces_film(rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    EPS = 1e-6
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
    if isinstance(rgb, np.ndarray):
        return rgb.clip(min=0.0, max=1.0)
    elif isinstance(rgb, torch.Tensor):
        return rgb.clamp(min=0.0, max=1.0)


def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError


def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055
    )


def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _rgb_to_srgb(f)
    )
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4)
    )


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _srgb_to_rgb(f)
    )
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def get_brdf_lut() -> torch.Tensor:
    brdf_lut_path = os.path.join(os.path.dirname(__file__), "brdf_256_256.bin")
    brdf_lut = torch.from_numpy(
        np.fromfile(brdf_lut_path, dtype=np.float32).reshape(1, 256, 256, 2)
    )
    return brdf_lut


def get_material(class_feature, class_mask, height, width):
    # 将特征张量 reshape 为 (H*W, 1)
    # class_feature = class_feature.reshape(height * width, 1)
    #
    # # 将其在第一维上平均分割为 11(+1) 份, 2*4 + 3 计算每一份的均值
    # split_size = height * width // 15
    # class_features = [split.mean() for split in torch.split(class_feature, split_size)]
    # # print(class_features)
    # # exit()
    #
    # concrete_roughness = class_features[0]
    # concrete_specular_albedo = class_features[1]
    # concrete_metallic = class_features[2]
    #
    # glass_roughness = class_features[3]
    # glass_specular_albedo = class_features[4]
    # glass_metallic = class_features[5]
    #
    # metal_roughness = class_features[6]
    # metal_specular_albedo = class_features[7]
    # metal_metallic = class_features[8]
    #
    # wood_roughness = class_features[9]
    # wood_specular_albedo = class_features[10]
    # wood_metallic = class_features[11]
    #
    # other_roughness = class_features[12]
    # other_specular_albedo = class_features[13]
    # other_metallic = class_features[14]
    #
    # class_mask = class_mask.squeeze(-1)
    # # 创建 concrete, glass, metal, wood 的 mask
    # # print(f"Max value in the tensor: {class_feature.max().item()}")
    # # print(f"Min value in the tensor: {class_feature.min().item()}")
    #
    wood_mask = (class_mask >= 0.) & (class_mask <= 0.2)
    metal_mask = (class_mask > 0.2) & (class_mask <= 0.4)
    concrete_mask = (class_mask > 0.4) & (class_mask <= 0.6)
    glass_mask = (class_mask > 0.6) & (class_mask <= 0.8)
    other_mask = (class_mask > 0.8) & (class_mask <= 1.)

    # # 将布尔掩码转换为 0 和 255 的图像格式
    # concrete_image = (concrete_mask.cpu().numpy() * 255).astype(np.uint8)
    # glass_image = (glass_mask.cpu().numpy() * 255).astype(np.uint8)
    # metal_image = (metal_mask.cpu().numpy() * 255).astype(np.uint8)
    # wood_image = (wood_mask.cpu().numpy() * 255).astype(np.uint8)
    #
    # total = (np.ones_like(wood_mask.cpu().numpy()) * 255.).astype(np.uint8)
    # div = total - (concrete_image + glass_image + metal_image + wood_image)
    #
    # import cv2
    # # 使用 OpenCV 保存图像
    # cv2.imwrite('/mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/1-concrete_mask.png', concrete_image)
    # cv2.imwrite('/mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/1-glass_mask.png', glass_image)
    # cv2.imwrite('/mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/1-metal_mask.png', metal_image)
    # cv2.imwrite('/mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/1-wood_mask.png', wood_image)
    # cv2.imwrite('/mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/1-div.png', div)
    #
    # exit()
    # print(concrete_mask)
    # print(f"Max value in the tensor: {concrete_mask.max().item()}")
    # print(f"Min value in the tensor: {concrete_mask.min().item()}")
    # exit(0)

    # # 创建与 feature 形状相同的空张量，用于存储结果
    metallic = torch.zeros_like(class_mask)
    # roughness = torch.zeros_like(class_mask)
    # specular_albedo = torch.zeros_like(class_mask)
    #
    # 根据 mask 设置具体的材质属性值
    metallic[concrete_mask] = torch.tensor(0.0).cuda()
    # roughness[concrete_mask] = concrete_roughness
    # specular_albedo[concrete_mask] = concrete_specular_albedo
    #
    metallic[glass_mask] = torch.tensor(0.7).cuda()
    # roughness[glass_mask] = glass_roughness
    # specular_albedo[glass_mask] = glass_specular_albedo
    #
    metallic[metal_mask] = torch.tensor(1.0).cuda()
    # roughness[metal_mask] = metal_roughness
    # specular_albedo[metal_mask] = metal_specular_albedo
    #
    metallic[wood_mask] = torch.tensor(0.1).cuda()
    # roughness[wood_mask] = wood_roughness
    # specular_albedo[wood_mask] = wood_specular_albedo
    #
    metallic[other_mask] = torch.tensor(0.3).cuda()
    # roughness[other_mask] = other_roughness
    # specular_albedo[other_mask] = other_specular_albedo

    roughness = class_feature
    # metallic = class_mask
    specular_albedo = torch.ones_like(roughness) * torch.tensor(0.04).cuda()

    metallic = metallic.reshape(1, height, width)
    roughness = roughness.reshape(1, height, width)
    specular_albedo = specular_albedo.reshape(1, height, width)

    return roughness, specular_albedo, metallic


def pbr_shading(
        light: CubemapLight,
        normals: torch.Tensor,  # [H, W, 3]
        view_dirs: torch.Tensor,  # [H, W, 3]
        albedo: torch.Tensor,  # [H, W, 3]
        roughness: torch.Tensor,  # [H, W, 1]
        mask: torch.Tensor,  # [H, W, 1]
        tone: bool = False,
        gamma: bool = False,
        occlusion: Optional[torch.Tensor] = None,  # [H, W, 1]
        irradiance: Optional[torch.Tensor] = None,  # [H, W, 1]
        metallic: Optional[torch.Tensor] = None,  # [H, W, 1]
        specular_albedo: Optional[torch.Tensor] = None,  # [H, W, 1]
        brdf_lut: Optional[torch.Tensor] = None,  # [1, 256, 256, 2]
        background: Optional[torch.Tensor] = None,
) -> Dict:
    H, W, _ = normals.shape
    if background is None:
        background = torch.zeros_like(normals)  # [H, W, 3]

    # prepare
    normals = normals.reshape(1, H, W, 3)
    view_dirs = view_dirs.reshape(1, H, W, 3)
    albedo = albedo.reshape(1, H, W, 3)
    roughness = roughness.reshape(1, H, W, 1)
    metallic = metallic.reshape(1, H, W, 1)
    assert specular_albedo is not None
    specular_albedo = 0.04 if specular_albedo is None else specular_albedo

    results = {}
    # prepare
    ref_dirs = (
            2.0 * (normals * view_dirs).sum(-1, keepdims=True).clamp(min=0.0) * normals - view_dirs
    )  # [1, H, W, 3]

    # Diffuse lookup
    diffuse_light = dr.texture(
        light.diffuse[None, ...],  # [1, 6, 16, 16, 3]
        normals.contiguous(),  # [1, H, W, 3]
        filter_mode="linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    if occlusion is not None:
        diffuse_light = diffuse_light * occlusion[None] + (1 - occlusion[None]) * irradiance[None]

    results["diffuse_light"] = diffuse_light[0]
    diffuse_rgb = diffuse_light * albedo  # [1, H, W, 3]

    # specular
    # NoV = saturate_dot(normals, view_dirs)  # [1, H, W, 1]
    # fg_uv = torch.cat((NoV, roughness), dim=-1)  # [1, H, W, 2]
    # fg_lookup = dr.texture(
    #     brdf_lut,  # [1, 256, 256, 2]
    #     fg_uv.contiguous(),  # [1, H, W, 2]
    #     filter_mode="linear",
    #     boundary_mode="clamp",
    # )  # [1, H, W, 2]

    # Roughness adjusted specular env lookup
    miplevel = light.get_mip(roughness)  # [1, H, W, 1]
    spec = dr.texture(
        light.specular[0][None, ...],  # [1, 6, env_res, env_res, 3]
        ref_dirs.contiguous(),  # [1, H, W, 3]
        mip=list(m[None, ...] for m in light.specular[1:]),
        mip_level_bias=miplevel[..., 0],  # [1, H, W]
        filter_mode="linear-mipmap-linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    # Compute aggregate lighting
    if metallic is None:
        F0 = torch.ones_like(albedo) * specular_albedo  # [1, H, W, 3]
    else:
        F0 = (1.0 - metallic) * specular_albedo + albedo * metallic
    # reflectance = F0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]  # [1, H, W, 3]
    reflectance = F0
    specular_rgb = spec * reflectance  # [1, H, W, 3]

    render_rgb = diffuse_rgb + specular_rgb  # [1, H, W, 3]
    render_rgb = render_rgb.squeeze()  # [H, W, 3]

    if tone:  # Tone Mapping
        render_rgb = aces_film(render_rgb)
    else:
        render_rgb = render_rgb.clamp(min=0.0, max=1.0)

    ### NOTE: close `gamma` will cause better resuls in novel view synthesis but wrose relighting results.
    ### NOTE: it is worth to figure out a better way to handle both novel view synthesis and relighting
    if gamma:
        render_rgb = linear_to_srgb(render_rgb.squeeze())

    render_rgb = torch.where(mask, render_rgb, background)

    results.update(
        {
            "render_rgb": render_rgb,
            "diffuse_rgb": diffuse_rgb.squeeze(),
            "specular_rgb": specular_rgb.squeeze(),
        }
    )

    return results
