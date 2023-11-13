import warnings

import PIL
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")

# <editor-fold desc='numpy边界格式转换'>
CORNERSN = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])


def xywhN2xyxyN(xywhN: np.ndarray) -> np.ndarray:
    xcyc, wh_2 = xywhN[:2], xywhN[2:4] / 2
    return np.concatenate([xcyc - wh_2, xcyc + wh_2], axis=0)


def xywhsN2xyxysN(xywhsN: np.ndarray) -> np.ndarray:
    xcyc, wh_2 = xywhsN[..., :2], xywhsN[..., 2:4] / 2
    return np.concatenate([xcyc - wh_2, xcyc + wh_2], axis=-1)


def xyxyN2xywhN(xyxyN: np.ndarray) -> np.ndarray:
    x1y1, x2y2 = xyxyN[:2], xyxyN[2:4]
    return np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], axis=0)


def xyxysN2xywhsN(xyxysN: np.ndarray) -> np.ndarray:
    x1y1, x2y2 = xyxysN[..., :2], xyxysN[..., 2:4]
    return np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], axis=-1)


def xyxyN2xlylN(xyxyN: np.ndarray) -> np.ndarray:
    xlyl = np.stack([xyxyN[[0, 0, 2, 2]], xyxyN[[1, 3, 3, 1]]], axis=1)
    return xlyl


def xyxysN2xlylsN(xyxysN: np.ndarray) -> np.ndarray:
    xlyls = np.stack([xyxysN[..., [0, 0, 2, 2]], xyxysN[..., [1, 3, 3, 1]]], axis=-1)
    return xlyls


def xywhN2xlylN(xywhN: np.ndarray) -> np.ndarray:
    return xywhN[:2] + CORNERSN * xywhN[2:4] / 2


def xywhsN2xlylsN(xywhN: np.ndarray) -> np.ndarray:
    return xywhN[..., None, :2] + CORNERSN * xywhN[..., None, 2:4] / 2


def xywhaN2x1y1whaN(xywhaN: np.ndarray) -> np.ndarray:
    cos, sin = math.cos(xywhaN[4]), math.sin(xywhaN[4])
    mat = np.array([[cos, -sin], [sin, cos]])
    return np.concatenate([xywhaN[:2] - xywhaN[2:4] @ mat / 2, xywhaN[2:5]], axis=0)


def xywhasN2x1y1whasN(xywhasN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhasN[..., 4]), np.sin(xywhasN[..., 4])
    wh_2 = xywhasN[..., 2:4, None] / 2
    mat = np.stack([np.stack([cos, -sin], axis=-1), np.stack([sin, cos], axis=-1)], axis=-2)
    return np.concatenate([xywhasN[..., :2] - wh_2 @ mat, xywhasN[..., 2:5]], axis=-1)


def xywhaN2xlylN(xywhaN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhaN[4]), np.sin(xywhaN[4])
    mat = np.array([[cos, sin], [-sin, cos]])
    xlyl = xywhaN[:2] + (CORNERSN * xywhaN[2:4] / 2) @ mat
    return xlyl


def xywhuvN2xlylN(xywhuvN: np.ndarray) -> np.ndarray:
    cos, sin = xywhuvN[4], xywhuvN[5]
    mat = np.array([[cos, sin], [-sin, cos]])
    xlyl = xywhuvN[:2] + (CORNERSN * xywhuvN[2:4] / 2) @ mat
    return xlyl


def xywhasN2xlylsN(xywhasN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhasN[..., 4]), np.sin(xywhasN[..., 4])
    mat = np.stack([np.stack([cos, sin], axis=-1), np.stack([-sin, cos], axis=-1)], axis=-2)
    xlyls = xywhasN[..., None, :2] + (CORNERSN * xywhasN[..., None, 2:4] / 2) @ mat
    return xlyls


def xywhasN2xyxysN(xywhasN: np.ndarray) -> np.ndarray:
    xlylsN = xywhasN2xlylsN(xywhasN)
    x1y1 = np.min(xlylsN, axis=-2)
    x2y2 = np.max(xlylsN, axis=-2)
    xyxysN = np.concatenate([x1y1, x2y2], axis=-1)
    return xyxysN


def xywhabN2xlylN(xywhabN: np.ndarray) -> np.ndarray:
    cosa, sina = np.cos(xywhabN[4]), np.sin(xywhabN[4])
    cosb, sinb = np.cos(xywhabN[5]), np.sin(xywhabN[5])
    mat = np.array([[cosa, sina], [cosb, sinb]])
    xlyl = xywhabN[:2] + (CORNERSN * xywhabN[2:4] / 2) @ mat
    return xlyl


def xywhabsN2xlylsN(xywhabsN: np.ndarray) -> np.ndarray:
    cosa, sina = np.cos(xywhabsN[..., 4]), np.sin(xywhabsN[..., 4])
    cosb, sinb = np.cos(xywhabsN[..., 5]), np.sin(xywhabsN[..., 5])
    mat = np.stack([np.stack([cosa, sina], axis=-1), np.stack([cosb, sinb], axis=-1)], axis=-2)
    xlyls = xywhabsN[..., None, :2] + (CORNERSN * xywhabsN[..., None, 2:4] / 2) @ mat
    return xlyls


def xlylN2xyxyN(xlylN: np.ndarray) -> np.ndarray:
    x1y1 = np.min(xlylN, axis=0)
    x2y2 = np.max(xlylN, axis=0)
    return np.concatenate([x1y1, x2y2], axis=0)


def xywhaN2xyxyN(xywhaN: np.ndarray) -> np.ndarray:
    xlylN = xywhaN2xlylN(xywhaN)
    return xlylN2xyxyN(xlylN)


def xywhabN2xyxyN(xywhabN: np.ndarray) -> np.ndarray:
    xlylN = xywhabN2xlylN(xywhabN)
    return xlylN2xyxyN(xlylN)


def xyxyN2xywhaN(xyxyN: np.ndarray) -> np.ndarray:
    xywh = xyxyN2xywhN(xyxyN)
    return np.concatenate([xywh, [0]], axis=0)


def xywhN2xywhaN(xywhN: np.ndarray) -> np.ndarray:
    return np.concatenate([xywhN, [0]], axis=0)


def xlylN2xywhaN(xlylN: np.ndarray) -> np.ndarray:
    vh = xlylN[1] - xlylN[0]
    vw = xlylN[2] - xlylN[1]
    xy = np.mean(xlylN, axis=0)
    a = np.arctan2(vw[1], vw[0])
    w = np.sqrt(np.sum(vw ** 2))
    h = np.sqrt(np.sum(vh ** 2))
    return np.concatenate([xy, [w, h, a]], axis=0)


def xywhaN2xywhN(xywhaN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhaN[4]), np.sin(xywhaN[4])
    wh = np.abs(cos * xywhaN[2:4]) + np.abs(sin * xywhaN[3:1:-1])
    return np.concatenate([xywhaN[:2], wh], axis=0)


def xywhaN2xywhuvN(xywhaN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhaN[4:5]), np.sin(xywhaN[4:5])
    return np.concatenate([xywhaN[:4], cos, sin], axis=0)


def xywhuvN2xywhaN(xywhuvN: np.ndarray) -> np.ndarray:
    alpha = np.arccos(xywhuvN[4]) if xywhuvN[5] > 0 else -np.arccos(xywhuvN[4])
    return np.concatenate([xywhuvN[:4], [alpha]], axis=0)


def xywhasN2xywhuvsN(xywhasN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhasN[..., 4:5]), np.sin(xywhasN[..., 4:5])
    return np.concatenate([xywhasN[..., :4], cos, sin], axis=-1)


def xywhuvsN2xywhasN(xywhuvsN: np.ndarray) -> np.ndarray:
    alphas = np.arccos(xywhuvsN[..., 4:5])
    alphas = np.where(xywhuvsN[..., 5:6] > 0, alphas, -alphas)
    return np.concatenate([xywhuvsN[..., :4], alphas], axis=-1)


def xywhabN2xywhN(xywhabN: np.ndarray) -> np.ndarray:
    cosa, sina = np.cos(xywhabN[4]), np.sin(xywhabN[4])
    cosb, sinb = np.cos(xywhabN[5]), np.sin(xywhabN[5])
    w = np.abs(cosa * xywhabN[2]) + np.abs(cosb * xywhabN[3])
    h = np.abs(sinb * xywhabN[3]) + np.abs(sina * xywhabN[2])
    return np.concatenate([xywhabN[:2], w[None], h[None]], axis=0)


def xlylN2xywhN(xlylN: np.ndarray) -> np.ndarray:
    return xyxyN2xywhN(xlylN2xyxyN(xlylN))


def xyxyN2ixysN(xyxyN: np.ndarray, size: tuple) -> np.ndarray:
    xyxyN = np.round(xyxyN).astype(np.int32)
    ixs = np.arange(max(xyxyN[0], 0), min(xyxyN[2], size[0]))
    iys = np.arange(max(xyxyN[1], 0), min(xyxyN[3], size[1]))
    ixys = np.stack(np.meshgrid(ixs, iys), axis=2).reshape(-1, 2)
    return ixys


def ixysN2xyxyN(ixysN: np.ndarray) -> np.ndarray:
    if ixysN.shape[0] == 0:
        return np.zeros(shape=4, dtype=np.int32)
    else:
        ixys_min = np.min(ixysN, axis=0)
        ixys_max = np.max(ixysN, axis=0)
        xyxy = np.concatenate([ixys_min, ixys_max + 1], axis=0)
        return xyxy


def xywhN2ixysN(xywhN: np.ndarray, size: tuple) -> np.ndarray:
    return xyxyN2ixysN(xywhN2xyxyN(xywhN), size)


def ixysN2xywhN(ixysN: np.ndarray) -> np.ndarray:
    return xyxyN2xywhN(ixysN2xyxyN(ixysN))


def ixysN2xywhuvN(ixysN: np.ndarray, cosN: np.ndarray, sinN: np.ndarray) -> np.ndarray:
    if ixysN.shape[0] == 0:
        return np.zeros(shape=6, dtype=np.int32)
    else:
        mat = np.array([[cosN, -sinN], [sinN, cosN]])
        xys_proj = (ixysN + 0.5) @ mat
        xy_proj_min = np.min(xys_proj, axis=0) - 0.5
        xy_proj_max = np.max(xys_proj, axis=0) + 0.5
        xy_proj_cen = (xy_proj_min + xy_proj_max) / 2
        xy = xy_proj_cen @ mat.T
        wh = xy_proj_max - xy_proj_min
        xywhuv = np.concatenate([xy, wh, [cosN, sinN]], axis=0)
        return xywhuv


def ixysN2xywhaN(ixysN: np.ndarray, alphaN: np.ndarray) -> np.ndarray:
    xywhuvN = ixysN2xywhuvN(ixysN, np.cos(alphaN), np.sin(alphaN))
    xywha = np.concatenate([xywhuvN[:4], [alphaN]], axis=0)
    return xywha


def xlylN2abclN(xlylN: np.ndarray) -> np.ndarray:
    xlyl_ext = np.concatenate([xlylN, xlylN[0:1]], axis=0)
    As = xlyl_ext[:-1, 1] - xlyl_ext[1:, 1]
    Bs = xlyl_ext[1:, 0] - xlyl_ext[:-1, 0]
    Cs = xlyl_ext[1:, 1] * xlyl_ext[:-1, 0] - xlyl_ext[:-1, 1] * xlyl_ext[1:, 0]
    return np.stack([As, Bs, Cs], axis=1)


def xlylsN2abclsN(xlylsN: np.ndarray) -> np.ndarray:
    xlyls_ext = np.concatenate([xlylsN, xlylsN[..., 0:1, :]], axis=-2)
    As = xlyls_ext[..., :-1, 1] - xlyls_ext[..., 1:, 1]
    Bs = xlyls_ext[..., 1:, 0] - xlyls_ext[..., :-1, 0]
    Cs = xlyls_ext[..., 1:, 1] * xlyls_ext[..., :-1, 0] - xlyls_ext[..., :-1, 1] * xlyls_ext[..., 1:, 0]
    return np.stack([As, Bs, Cs], axis=-1)


def xlylN2areaN(xlylN: np.ndarray) -> np.ndarray:
    xlyl_ext = np.concatenate([xlylN, xlylN[0:1]], axis=0)
    area = xlyl_ext[1:, 0] * xlyl_ext[:-1, 1] - xlyl_ext[:-1, 0] * xlyl_ext[1:, 1]
    area = np.sum(area) / 2
    return area


def xlylsN2areasN(xlylsN: np.ndarray) -> np.ndarray:
    xlyls_ext = np.concatenate([xlylsN, xlylsN[..., 0:1, :]], axis=-2)
    areas = xlyls_ext[..., 1:, 0] * xlyls_ext[..., :-1, 1] - xlyls_ext[..., :-1, 0] * xlyls_ext[..., 1:, 1]
    areas = np.sum(areas, axis=-1) / 2
    return areas


def xywhN2maskNb(xywhN: np.ndarray, size: tuple) -> np.ndarray:
    xyxyN = xywhN2xyxyN(xywhN)
    return xyxyN2maskNb(xyxyN, size)


def xyxyN2maskNb(xyxyN: np.ndarray, size: tuple) -> np.ndarray:
    xs = np.arange(size[0])[None, :] + 0.5
    ys = np.arange(size[1])[:, None] + 0.5
    maskN = (xs > xyxyN[0]) * (xs < xyxyN[2]) * (ys > xyxyN[1]) * (ys < xyxyN[3])
    return maskN


def xywhsN2masksNb(xywhsN: np.ndarray, size: tuple) -> np.ndarray:
    return xyxysN2masksNb(xywhsN2xyxysN(xywhsN), size)


def xyxysN2masksNb(xyxysN: np.ndarray, size: tuple) -> np.ndarray:
    Wf, Hf = size
    Nb = xyxysN.shape[0]
    meshes = _create_meshesN(Nb, Hf, Wf)
    maskNb = np.all((meshes < xyxysN[:, None, 2:4]) * (meshes > xyxysN[:, None, :2]), axis=2)
    maskNb = maskNb.reshape(Nb, Hf, Wf)
    return maskNb


def xywhuvN2maskNb(xywhuvN: np.ndarray, size: tuple) -> np.ndarray:
    xlylN = xywhuvN2xlylN(xywhuvN)
    return xlylN2maskNb(xlylN, size)


def xywhaN2maskNb(xywhaN: np.ndarray, size: tuple) -> np.ndarray:
    xlylN = xywhaN2xlylN(xywhaN)
    return xlylN2maskNb(xlylN, size)


def xywhasN2masksN_guss(xywhasN: np.ndarray, size: tuple) -> np.ndarray:
    xywhuvsN = xywhasN2xywhuvsN(xywhasN)
    return xywhuvsN2masksN_guss(xywhuvsN, size)


def _create_meshesN(Nb: int, Hf: int, Wf: int) -> np.ndarray:
    xs = np.broadcast_to(np.arange(Wf)[None, None, :], (Nb, Hf, Wf))
    ys = np.broadcast_to(np.arange(Hf)[None, :, None], (Nb, Hf, Wf))
    meshes = np.stack([xs + 0.5, ys + 0.5], axis=3).reshape(Nb, Hf * Wf, 2)
    return meshes


def xywhuvsN2masksN_guss(xywhuvsN: np.ndarray, size: tuple) -> np.ndarray:
    Nb = xywhuvsN.shape[0]
    Wf, Hf = size
    xys, whs, coss, sins = xywhuvsN[:, None, 0:2], xywhuvsN[:, None, 2:4], xywhuvsN[:, 4], xywhuvsN[:, 5]
    mats = np.stack([np.stack([coss, sins], axis=1), np.stack([-sins, coss], axis=1)], axis=2)

    meshes = _create_meshesN(Nb, Hf, Wf) - xys
    meshes_proj = meshes @ mats
    pows_guss = np.exp(-np.sum((meshes_proj * 2 / whs) ** 2, axis=2) / 2)
    pows_guss = pows_guss.reshape(Nb, Hf, Wf)
    return pows_guss


def xyxysN2masksN_guss(xyxysN: np.ndarray, size: tuple) -> np.ndarray:
    return xywhsN2masksN_guss(xyxysN2xywhsN(xyxysN), size)


def xywhsN2masksN_guss(xywhsN: np.ndarray, size: tuple) -> np.ndarray:
    Nb = xywhsN.shape[0]
    Wf, Hf = size
    xys, whs = xywhsN[:, None, 0:2], xywhsN[:, None, 2:4]
    meshes = _create_meshesN(Nb, Hf, Wf) - xys
    pows_guss = np.exp(-np.sum((meshes * 2 / whs) ** 2, axis=2) / 2)
    pows_guss = pows_guss.reshape(Nb, Hf, Wf)
    return pows_guss


def xywhasN2masksN_cness(xywhasN: np.ndarray, size: tuple) -> np.ndarray:
    xywhuvsN = xywhasN2xywhuvsN(xywhasN)
    return xywhuvsN2masksN_cness(xywhuvsN, size)


def xywhuvsN2masksN_cness(xywhuvsN: np.ndarray, size: tuple) -> np.ndarray:
    Nb = xywhuvsN.shape[0]
    Wf, Hf = size
    xys, whs, coss, sins = xywhuvsN[:, None, 0:2], xywhuvsN[:, None, 2:4], xywhuvsN[:, 4], xywhuvsN[:, 5]
    mats = np.stack([np.stack([coss, sins], axis=1), np.stack([-sins, coss], axis=1)], axis=2)

    meshes = _create_meshesN(Nb, Hf, Wf) - xys
    meshes_proj = meshes @ mats
    meshes_proj = np.abs(meshes_proj)

    pows_cness = np.sqrt(np.prod(np.clip(whs / 2 - meshes_proj, a_min=0, a_max=None), axis=2)
                         / np.prod(whs / 2 + meshes_proj, axis=2))

    pows_cness = pows_cness.reshape(Nb, Hf, Wf)
    return pows_cness


def xywhasN2masksNb(xywhasN: np.ndarray, size: tuple) -> np.ndarray:
    xywhuvsN = xywhasN2xywhuvsN(xywhasN)
    return xywhuvsN2masksNb(xywhuvsN, size)


def xywhuvsN2masksNb(xywhuvsN: np.ndarray, size: tuple) -> np.ndarray:
    Nb = xywhuvsN.shape[0]
    Wf, Hf = size
    xys, whs, coss, sins = xywhuvsN[:, None, 0:2], xywhuvsN[:, None, 2:4], xywhuvsN[:, 4], xywhuvsN[:, 5]
    mats = np.stack([np.stack([coss, sins], axis=1), np.stack([-sins, coss], axis=1)], axis=2)

    meshes = _create_meshesN(Nb, Hf, Wf) - xys
    meshes_proj = meshes @ mats
    meshes_proj = np.abs(meshes_proj)
    masksNb = np.all(meshes_proj < whs / 2, axis=2)
    masksNb = masksNb.reshape(Nb, Hf, Wf)
    return masksNb


def xlylN2maskNb_convex(xlylN: np.ndarray, size: tuple) -> np.ndarray:
    maskN = np.zeros(shape=(size[1], size[0]), dtype=bool)
    if xlylN.shape[0] >= 3:
        abcl = xlylN2abclN(xlylN)
        xyxy = np.round(xlylN2xyxyN(xlylN)).astype(np.int32)
        xyxy = xyxyN_clip(xyxy, np.array([0, 0, size[0], size[1]]))
        xs = np.arange(xyxy[0], xyxy[2])[None, :, None] + 0.5
        ys = np.arange(xyxy[1], xyxy[3])[:, None, None] + 0.5
        maskN_ref = (xs * abcl[..., 0] + ys * abcl[..., 1] + abcl[..., 2]) >= 0
        maskN_ref = np.all(maskN_ref, axis=2) + np.all(~maskN_ref, axis=2)
        maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] += maskN_ref
    return maskN


def xlylN2maskNb(xlylN: np.ndarray, size: tuple) -> np.ndarray:
    maskNb = np.zeros(shape=(size[1], size[0]), dtype=np.float32)
    if xlylN.shape[0] >= 3:
        cv2.fillPoly(maskNb, [xlylN.astype(np.int32)], color=1.0)
    return maskNb.astype(bool)


def xlylNs2maskNb(xlylNs: list, size: tuple) -> np.ndarray:
    maskNb = np.zeros(shape=(size[1], size[0]), dtype=np.float32)
    xlylNs = [xlylN.astype(np.int32) for xlylN in xlylNs]
    cv2.fillPoly(maskNb, xlylNs, color=1.0)
    return maskNb.astype(bool)


def abclN_intersect(abcl1N: np.ndarray, abcl2N: np.ndarray) -> np.ndarray:
    x = (abcl1N[:, 1] * abcl2N[:, 2] - abcl1N[:, 2] * abcl2N[:, 1]) / \
        (abcl1N[:, 0] * abcl2N[:, 1] - abcl1N[:, 1] * abcl2N[:, 0])
    y = (abcl1N[:, 0] * abcl2N[:, 2] - abcl1N[:, 2] * abcl2N[:, 0]) / \
        (abcl1N[:, 1] * abcl2N[:, 0] - abcl1N[:, 0] * abcl2N[:, 1])
    return np.stack([x, y], axis=1)


def abclsN_intersect(abcls1N: np.ndarray, abcls2N: np.ndarray) -> np.ndarray:
    x = (abcls1N[..., 1] * abcls2N[..., 2] - abcls1N[..., 2] * abcls2N[..., 1]) / \
        (abcls1N[..., 0] * abcls2N[..., 1] - abcls1N[..., 1] * abcls2N[..., 0])
    y = (abcls1N[..., 0] * abcls2N[..., 2] - abcls1N[..., 2] * abcls2N[..., 0]) / \
        (abcls1N[..., 1] * abcls2N[..., 0] - abcls1N[..., 0] * abcls2N[..., 1])
    return np.stack([x, y], axis=-1)


def xlylN_intersect_coreN(xlyl1N: np.ndarray, xlyl2N: np.ndarray, eps: float = 1e-7) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    abcl1 = xlylN2abclN(xlyl1N)
    abcl2 = xlylN2abclN(xlyl2N)
    fltr_1pin2 = (xlyl1N[:, None, 0] * abcl2[None, :, 0] + xlyl1N[:, None, 1] * abcl2[None, :, 1]
                  + abcl2[None, :, 2]) <= eps
    msk_1pin2 = np.all(fltr_1pin2, axis=1)
    if np.all(msk_1pin2):
        return np.full_like(fltr_1pin2, fill_value=False), abcl1, abcl2, \
               msk_1pin2, np.full(shape=xlyl2N.shape[0], fill_value=False)
    fltr_2pin1 = (xlyl2N[None, :, 0] * abcl1[:, None, 0] + xlyl2N[None, :, 1] * abcl1[:, None, 1]
                  + abcl1[:, None, 2]) <= eps
    msk_2pin1 = np.all(fltr_2pin1, axis=0)
    if np.all(msk_2pin1):
        return np.full_like(fltr_1pin2, fill_value=False), abcl1, abcl2, \
               msk_1pin2, msk_2pin1
    fltr_1lin2 = np.concatenate([fltr_1pin2, fltr_1pin2[0:1]], axis=0)
    fltr_1lin2 = fltr_1lin2[1:] ^ fltr_1lin2[:-1]

    fltr_2lin1 = np.concatenate([fltr_2pin1, fltr_2pin1[:, 0:1]], axis=1)
    fltr_2lin1 = fltr_2lin1[:, 1:] ^ fltr_2lin1[:, :-1]
    fltr_int = fltr_2lin1 * fltr_1lin2
    return fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1


def xlylsN_intersect_coresN(xlyls1N: np.ndarray, xlyls2N: np.ndarray, eps: float = 1e-7) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    abcls1 = xlylsN2abclsN(xlyls1N)
    abcls2 = xlylsN2abclsN(xlyls2N)
    fltr_1pin2 = (xlyls1N[..., None, 0] * abcls2[..., None, :, 0] + xlyls1N[..., None, 1] * abcls2[..., None, :, 1]
                  + abcls2[..., None, :, 2]) < eps
    msk_1pin2 = np.all(fltr_1pin2, axis=-1)
    fltr_2pin1 = (xlyls2N[..., None, :, 0] * abcls1[..., None, 0] + xlyls2N[..., None, :, 1] * abcls1[..., None, 1]
                  + abcls1[..., None, 2]) < eps
    msk_2pin1 = np.all(fltr_2pin1, axis=-2)

    fltr_1lin2 = np.concatenate([fltr_1pin2, fltr_1pin2[..., 0:1, :]], axis=-2)
    fltr_1lin2 = fltr_1lin2[..., 1:, :] ^ fltr_1lin2[..., :-1, :]

    fltr_2lin1 = np.concatenate([fltr_2pin1, fltr_2pin1[..., 0:1]], axis=-1)
    fltr_2lin1 = fltr_2lin1[..., 1:] ^ fltr_2lin1[..., :-1]

    fltr_int = fltr_2lin1 * fltr_1lin2
    return fltr_int, abcls1, abcls2, msk_1pin2, msk_2pin1


def intersect_coreN2xlylN(xlyl1N: np.ndarray, xlyl2N: np.ndarray, fltr_int: np.ndarray,
                          abcl1: np.ndarray, abcl2: np.ndarray, msk_1pin2: np.ndarray,
                          msk_2pin1: np.ndarray) -> np.ndarray:
    if np.all(msk_1pin2):
        return xlyl1N
    elif np.all(msk_2pin1):
        return xlyl2N
    # 节点排序
    idls1, idls2 = np.nonzero(fltr_int)
    if idls1.shape[0] == 0:
        return np.zeros(shape=(0, 2), dtype=np.float32)
    num1 = xlyl1N.shape[0]
    num2 = xlyl2N.shape[0]
    num_int = len(idls1)

    xlyl_int = abclN_intersect(abcl1[idls1], abcl2[idls2])
    dists = np.sum(np.abs(xlyl_int - xlyl1N[idls1]), axis=1) + idls1 * np.sum(np.abs(abcl1[:, :2]))
    # dists = ((idls1 - idls1[0]) % num1) * num2 + (idls2 - idls2[0]) % num2
    order = np.argsort(dists)
    idls2 = idls2[order] + 1
    idls1 = idls1 + 1
    xlyl_int = xlyl_int[order]
    # 按序遍历

    idls1_nxt = np.concatenate([idls1[1:], idls1[0:1]])
    idls1_nxt = np.where(idls1_nxt < idls1, idls1_nxt + num1, idls1_nxt)
    idls2_nxt = np.concatenate([idls2[1:], idls2[0:1]])
    idls2_nxt = np.where(idls2_nxt < idls2, idls2_nxt + num2, idls2_nxt)
    ids = [np.zeros(shape=0, dtype=np.int32)]
    for i in range(num_int):
        ids.append([i + num1 + num2])
        ids.append(np.arange(idls1[i], idls1_nxt[i]) % num1)
        ids.append(np.arange(idls2[i], idls2_nxt[i]) % num2 + num1)
    ids = np.concatenate(ids, axis=0)
    pnts = np.concatenate([xlyl1N, xlyl2N, xlyl_int])
    msks = np.concatenate([msk_1pin2, msk_2pin1, np.full(shape=xlyl_int.shape[0], fill_value=True)])

    xlyl_final = pnts[ids][msks[ids]]
    return xlyl_final


def xyxyN_clip(xyxyN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    xyxyN[0:4:2] = np.clip(xyxyN[0:4:2], a_min=xyxyN_rgn[0], a_max=xyxyN_rgn[2])
    xyxyN[1:4:2] = np.clip(xyxyN[1:4:2], a_min=xyxyN_rgn[1], a_max=xyxyN_rgn[3])
    return xyxyN


def xywhN_clip(xywhN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    xyxy = xywhN2xyxyN(xywhN)
    xyxy = xyxyN_clip(xyxy, xyxyN_rgn=xyxyN_rgn)
    xywhN = xyxyN2xywhN(xyxy)
    return xywhN


def xlylN_intersect(xlyl1: np.ndarray, xlyl2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1 = xlylN_intersect_coreN(xlyl1, xlyl2, eps=eps)
    xlyl_final = intersect_coreN2xlylN(xlyl1, xlyl2, fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1)
    return xlyl_final


def xlylN_clip(xlylN: np.ndarray, xyxyN_rgn: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    xlyl_rgn = xyxyN2xlylN(xyxyN_rgn)
    xlylN = xlylN_intersect(xlylN, xlyl_rgn, eps=eps)
    return xlylN


def xywhaN_clip(xywhaN: np.ndarray, xyxyN_rgn: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    if np.any(xywhaN[2:4] == 0):
        return xywhaN
    xlyl = xywhaN2xlylN(xywhaN)
    xlyl_rgn = xyxyN2xlylN(xyxyN_rgn)
    fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1 = xlylN_intersect_coreN(xlyl, xlyl_rgn, eps=eps)
    idls1, idls2 = np.nonzero(fltr_int)
    xlyl_int = abclN_intersect(abcl1[idls1], abcl2[idls2])
    pnts = np.concatenate([xlyl_int, xlyl[msk_1pin2], xlyl_rgn[msk_2pin1]], axis=0)
    cos, sin = np.cos(xywhaN[4]), np.sin(xywhaN[4])
    mat = np.array([[cos, sin], [-sin, cos]])
    pnts_cast = (pnts - xywhaN[:2]) @ mat.T
    if len(pnts_cast) == 0:
        return np.array([xywhaN[0], xywhaN[1], 0, 0, xywhaN[4]])
    w_min = np.min(pnts_cast[:, 0])
    w_max = np.max(pnts_cast[:, 0])
    h_min = np.min(pnts_cast[:, 1])
    h_max = np.max(pnts_cast[:, 1])
    xy = np.array([(w_min + w_max) / 2, (h_min + h_max) / 2]) @ mat + xywhaN[:2]
    xywhaN_clp = np.concatenate([xy, [w_max - w_min, h_max - h_min, xywhaN[4]]])

    return xywhaN_clp


def xywhabN_clip(xywhabN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    cosa, sina = np.cos(xywhabN[4]), np.sin(xywhabN[4])
    cosb, sinb = np.cos(xywhabN[5]), np.sin(xywhabN[5])
    mat = np.array([[cosa, sina], [cosb, sinb]])
    pns_b = np.stack([xyxyN_rgn[:2], xyxyN_rgn[2:], [xyxyN_rgn[0], xyxyN_rgn[3]], [xyxyN_rgn[2], xyxyN_rgn[1]]], axis=0)
    pns_m = xywhabN2xlylN(xywhabN)
    pns_m[:, 0] = np.clip(pns_m[:, 0], a_min=xyxyN_rgn[0], a_max=xyxyN_rgn[2])
    pns_m[:, 1] = np.clip(pns_m[:, 1], a_min=xyxyN_rgn[1], a_max=xyxyN_rgn[3])
    pns_b_cast = (pns_b - xywhabN[:2]) @ mat.T
    pns_m_cast = (pns_m - xywhabN[:2]) @ mat.T
    ws_max = np.clip(np.array([-xywhabN[2], xywhabN[2]]) / 2, a_min=np.min(pns_b_cast[:, 0]),
                     a_max=np.max(pns_b_cast[:, 0]))
    hs_max = np.clip(np.array([-xywhabN[3], xywhabN[3]]) / 2, a_min=np.min(pns_b_cast[:, 1]),
                     a_max=np.max(pns_b_cast[:, 1]))
    ws_min = np.array([np.max(pns_m_cast[[0, 1], 0]), np.min(pns_m_cast[[2, 3], 0])])
    hs_min = np.array([np.max(pns_m_cast[[0, 3], 1]), np.min(pns_m_cast[[1, 2], 1])])
    ws = (ws_max + ws_min) / 2
    hs = (hs_max + hs_min) / 2
    xy = np.array([np.mean(ws), np.mean(hs)]) @ mat + xywhabN[:2]
    xywhabN = np.concatenate([xy, [ws[1] - ws[0], hs[1] - hs[0], xywhabN[4], xywhabN[5]]])
    return xywhabN


def xlylN2homography(xlylN_src: np.ndarray, xlylN_tgd: np.ndarray) -> np.ndarray:
    assert xlylN_src.shape[0] == xlylN_tgd.shape[0], 'len err'
    num_vert = xlylN_src.shape[0]
    xxp = xlylN_src * xlylN_tgd[:, 0:1]
    yyp = xlylN_src * xlylN_tgd[:, 1:2]
    Ax = np.concatenate([xlylN_src, np.ones((num_vert, 1)), np.zeros((num_vert, 3)), -xxp], axis=1)
    Ay = np.concatenate([np.zeros((num_vert, 3)), xlylN_src, np.ones((num_vert, 1)), -yyp], axis=1)
    A, b = np.concatenate([Ax, Ay], axis=0), np.concatenate([xlylN_tgd[:, 0:1], xlylN_tgd[:, 1:2]], axis=0)
    h = np.linalg.inv(A.T @ A) @ A.T @ b
    H = np.concatenate([h.reshape(-1), [1]], axis=0).reshape((3, 3))
    return H


def xlylN_perspective(xlylN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xlyl_ext = np.concatenate([xlylN, np.ones(shape=(xlylN.shape[0], 1))], axis=1)
    xlyl_td = xlyl_ext @ H.T
    return xlyl_td[:, :2] / xlyl_td[:, 2:]


def xyN_perspective(xyN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xyN_ext = np.concatenate([xyN, [1]])
    xyN_trd = H @ xyN_ext
    return xyN_trd[:2] / xyN_trd[2]


def xyxyN_perspective(xyxyN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xlylN = xyxyN2xlylN(xyxyN)
    xlylN = xlylN_perspective(xlylN, H=H)
    return xlylN2xyxyN(xlylN)


def xywhN_perspective(xywhN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xlylN = xywhN2xlylN(xywhN)
    xlylN = xlylN_perspective(xlylN, H=H)
    return xlylN2xywhN(xlylN)


def xywhaN_perspective(xywhaN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xlylN = xywhaN2xlylN(xywhaN)
    xlylN = xlylN_perspective(xlylN, H=H)
    return xlylN2xywhaN(xlylN)


# </editor-fold>

# <editor-fold desc='torch边界格式转换'>
CORNERST = torch.Tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]])


def cindT2chotT(cindT: torch.Tensor, num_cls: int) -> torch.Tensor:
    clso = torch.zeros(num_cls)
    clso[cindT] = 1
    return clso


def cindsT2chotsT(cindsT: torch.Tensor, num_cls: int) -> torch.Tensor:
    num = len(cindsT)
    clsos = torch.zeros(num, num_cls)
    clsos[range(num), cindsT] = 1
    return clsos


def xyxyT2xywhT(xyxyT: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = xyxyT[:2], xyxyT[2:4]
    return torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1], dim=0)


def xyxyT2xywhaT_align(xyxyT: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = xyxyT[:2], xyxyT[2:4]
    xyc, wh = (x1y1 + x2y2) / 2, x2y2 - x1y1
    if wh[0] > wh[1]:
        return torch.cat([xyc, wh, torch.Tensor([0]).to(xyxyT.device)], dim=0)
    else:
        return torch.cat([xyc, torch.Tensor([wh[1], wh[0], math.pi / 2]).to(xyxyT.device)], dim=0)


def xyxysT2xywhsT(xyxysT: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = xyxysT[..., :2], xyxysT[..., 2:4]
    return torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1], dim=-1)

def xyxysT2xywhasT(xyxysT: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = xyxysT[..., :2], xyxysT[..., 2:4]
    alphas=torch.zeros_like(xyxysT[..., :1]).to(xyxysT.device)
    return torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1,alphas], dim=-1)

def xywhT2xyxyT(xywhT: torch.Tensor) -> torch.Tensor:
    xcyc, wh_2 = xywhT[:2], xywhT[2:4] / 2
    return torch.cat([xcyc - wh_2, xcyc + wh_2], dim=0)


def xywhsT2xyxysT(xywhsT: torch.Tensor) -> torch.Tensor:
    xcyc, wh_2 = xywhsT[..., :2], xywhsT[..., 2:4] / 2
    return torch.cat([xcyc - wh_2, xcyc + wh_2], dim=-1)


def xyxyT2xlylT(xyxyT: torch.Tensor) -> torch.Tensor:
    xlyl = torch.stack([xyxyT[[0, 0, 2, 2]], xyxyT[[1, 3, 3, 1]]], dim=1)
    return xlyl


def xyxysT2xlylsT(xyxysT: torch.Tensor) -> torch.Tensor:
    xlyls = torch.stack([xyxysT[..., [0, 0, 2, 2]], xyxysT[..., [1, 3, 3, 1]]], dim=-1)
    return xlyls


def xywhaT2xlylT(xywhaT: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(xywhaT[4]), torch.sin(xywhaT[4])
    mat = torch.Tensor([[cos, sin], [-sin, cos]])
    xlyl = xywhaT[:2] + (CORNERST * xywhaT[2:4] / 2) @ mat
    return xlyl


def xywhasT2xlylsT(xywhasT: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(xywhasT[..., 4]), torch.sin(xywhasT[..., 4])
    mat = torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([-sin, cos], dim=-1)], dim=-2)
    xlyls = xywhasT[..., None, :2] + (CORNERST.to(xywhasT.device) * xywhasT[..., None, 2:4] / 2) @ mat
    return xlyls


def xywhabsT2xlylsT(xywhabsT: torch.Tensor) -> torch.Tensor:
    cosa, sina = torch.cos(xywhabsT[..., 4]), torch.sin(xywhabsT[..., 4])
    cosb, sinb = torch.cos(xywhabsT[..., 5]), torch.sin(xywhabsT[..., 5])
    mat = torch.stack([torch.stack([cosa, sina], dim=-1), torch.stack([cosb, sinb], dim=-1)], dim=-2)
    xlyls = xywhabsT[..., None, :2] + (CORNERST.to(xywhabsT.device) * xywhabsT[..., None, 2:4] / 2) @ mat
    return xlyls


def xlylsT2xyxysT(xlylsT: torch.Tensor) -> torch.Tensor:
    x1y1 = torch.min(xlylsT, dim=-2)[0]
    x2y2 = torch.max(xlylsT, dim=-2)[0]
    xyxys = torch.cat([x1y1, x2y2], dim=-1)
    return xyxys


def xywhasT2xyxysT(xywhasT: torch.Tensor) -> torch.Tensor:
    xlyls = xywhasT2xlylsT(xywhasT)
    xyxys = xlylsT2xyxysT(xlyls)
    return xyxys


def xyxysT_clip(xyxysT: torch.Tensor, xyxyN_rgn: np.ndarray) -> torch.Tensor:
    xyxysT[..., slice(0, 4, 2)] = torch.clamp(xyxysT[..., slice(0, 4, 2)], min=xyxyN_rgn[0], max=xyxyN_rgn[2])
    xyxysT[..., slice(1, 4, 2)] = torch.clamp(xyxysT[..., slice(1, 4, 2)], min=xyxyN_rgn[1], max=xyxyN_rgn[3])
    return xyxysT


def xywhsT_clip(xywhsT: torch.Tensor, xyxyN_rgn: np.ndarray) -> torch.Tensor:
    xyxysT = xywhsT2xyxysT(xywhsT)
    xyxysT = xyxysT_clip(xyxysT, xyxyN_rgn=xyxyN_rgn)
    xywhsT = xyxysT2xywhsT(xyxysT)
    return xywhsT


def _create_meshesT(Nb: int, Hf: int, Wf: int, device) -> torch.Tensor:
    ys = torch.arange(Hf, device=device)[:, None].expand(Hf, Wf)
    xs = torch.arange(Wf, device=device)[None, :].expand(Hf, Wf)
    meshes = torch.stack([xs + 0.5, ys + 0.5], dim=2).view(Wf * Hf, 2).expand(Nb, Wf * Hf, 2)
    return meshes


def xywhasT2masksT_guss(xywhasT: torch.Tensor, size: tuple) -> torch.Tensor:
    xywhuvsT = xywhasT2xywhuvsT(xywhasT)
    return xywhuvsT2masksT_guss(xywhuvsT, size)


def xywhuvsT2masksT_guss(xywhuvsT: torch.Tensor, size: tuple) -> torch.Tensor:
    Nb = xywhuvsT.size(0)
    Wf, Hf = size
    xys, whs, coss, sins = xywhuvsT[:, None, 0:2], xywhuvsT[:, None, 2:4], xywhuvsT[:, 4], xywhuvsT[:, 5]
    mats = torch.stack([torch.stack([coss, sins], dim=1), torch.stack([-sins, coss], dim=1)], dim=2)

    meshes = _create_meshesT(Nb, Hf, Wf, xywhuvsT.device) - xys
    meshes_proj = torch.bmm(meshes, mats)

    pows_guss = torch.exp(-torch.sum((meshes_proj * 2 / whs) ** 2, dim=2) / 2)
    pows_guss = pows_guss.view(Nb, Hf, Wf)
    return pows_guss


def xywhasT2masksT_cness(xywhasT: torch.Tensor, size: tuple) -> torch.Tensor:
    xywhuvsT = xywhasT2xywhuvsT(xywhasT)
    return xywhuvsT2masksT_cness(xywhuvsT, size)


def xywhuvsT2masksT_cness(xywhuvsT: torch.Tensor, size: tuple) -> torch.Tensor:
    Nb = xywhuvsT.size(0)
    Wf, Hf = size
    xys, whs, coss, sins = xywhuvsT[:, None, 0:2], xywhuvsT[:, None, 2:4], xywhuvsT[:, 4], xywhuvsT[:, 5]
    mats = torch.stack([torch.stack([coss, sins], dim=1), torch.stack([-sins, coss], dim=1)], dim=2)

    meshes = _create_meshesT(Nb, Hf, Wf, xywhuvsT.device) - xys
    meshes_proj = torch.bmm(meshes, mats)
    meshes_proj = torch.abs(meshes_proj)

    pows_cness = torch.sqrt(torch.prod(torch.clamp_(whs / 2 - meshes_proj, min=0), dim=2)
                            / torch.prod(whs / 2 + meshes_proj, dim=2))

    pows_cness = pows_cness.view(Nb, Hf, Wf)
    return pows_cness


def xywhasT2masksTb(xywhasT: torch.Tensor, size: tuple) -> torch.Tensor:
    xywhuvsT = xywhasT2xywhuvsT(xywhasT)
    return xywhuvsT2masksTb(xywhuvsT, size)


def xywhuvsT2masksTb(xywhuvsT: torch.Tensor, size: tuple) -> torch.Tensor:
    Nb = xywhuvsT.size(0)
    Wf, Hf = size
    xys, whs, coss, sins = xywhuvsT[:, None, 0:2], xywhuvsT[:, None, 2:4], xywhuvsT[:, 4], xywhuvsT[:, 5]
    mats = torch.stack([torch.stack([coss, sins], dim=1), torch.stack([-sins, coss], dim=1)], dim=2)

    meshes = _create_meshesT(Nb, Hf, Wf, xywhuvsT.device) - xys
    meshes_proj = torch.bmm(meshes, mats)
    meshes_proj = torch.abs(meshes_proj)
    masks = torch.all(meshes_proj < whs / 2, dim=2)
    masks = masks.view(Nb, Hf, Wf)
    return masks


def xywhuvsT2masksTb_border(xywhuvsT: torch.Tensor, size: tuple, expand_ratio=1.2) -> torch.Tensor:
    Nb = xywhuvsT.size(0)
    Wf, Hf = size
    xys, whs, coss, sins = xywhuvsT[:, None, 0:2], xywhuvsT[:, None, 2:4], xywhuvsT[:, 4], xywhuvsT[:, 5]
    mats = torch.stack([torch.stack([coss, sins], dim=1), torch.stack([-sins, coss], dim=1)], dim=2)

    meshes = _create_meshesT(Nb, Hf, Wf, xywhuvsT.device) - xys
    meshes_proj = torch.bmm(meshes, mats)
    meshes_proj = torch.abs(meshes_proj)
    masks = torch.all((meshes_proj >= whs / 2) * (meshes_proj < whs / 2 * expand_ratio), dim=2)
    masks = masks.view(Nb, Hf, Wf)
    return masks


def xyxysT2masksTb(xyxysT: torch.Tensor, size: tuple) -> torch.Tensor:
    Nb, _ = xyxysT.size()
    Wf, Hf = size
    x1y1s, x2y2s = xyxysT[:, None, 0:2], xyxysT[:, None, 2:4]
    meshes = _create_meshesT(Nb, Hf, Wf, xyxysT.device)
    masks = torch.all((meshes < x2y2s) * (meshes > x1y1s), dim=2)
    masks = masks.view(Nb, Hf, Wf)
    return masks


def _masksT_scatter_btch(masksT: torch.Tensor, ids_b: torch.Tensor, Nb: int) -> torch.Tensor:
    masks = torch.zeros((Nb, masksT.size(1), masksT.size(2)), device=masksT.device, dtype=masksT.dtype)
    masks.scatter_add_(dim=0, index=ids_b[:, None, None].expand(masksT.size()), src=masksT)
    return masks


def _masksTb_scatter_btch_cind(masksTb: torch.Tensor, ids_b: torch.Tensor, Nb: int, cinds: torch.Tensor,
                               num_cls: int) -> torch.Tensor:
    masks = torch.full((Nb, masksTb.size(1), masksTb.size(2)), device=masksTb.device, dtype=torch.long,
                       fill_value=num_cls)
    ib, ih, iw = torch.nonzero(masksTb, as_tuple=True)
    masks[ids_b[ib], ih, iw] = cinds[ib]
    return masks


def bxyxysT2masksTb(xyxys: torch.Tensor, ids_b: torch.Tensor, Nb: int, size: tuple) -> torch.Tensor:
    masks_bool = xyxysT2masksTb(xyxys, size)
    return _masksT_scatter_btch(masks_bool, ids_b, Nb)


def bxywhasT2masksTb(xywhas: torch.Tensor, ids_b: torch.Tensor, Nb: int, size: tuple) -> torch.Tensor:
    masks_bool = xywhasT2masksTb(xywhas, size)
    return _masksT_scatter_btch(masks_bool, ids_b, Nb)


def bcxyxysT2masksT_enc(xyxys: torch.Tensor, ids_b: torch.Tensor, Nb: int, cinds: torch.Tensor, num_cls: int,
                        size: tuple) -> torch.Tensor:
    masks_bool = xyxysT2masksTb(xyxys, size)
    return _masksTb_scatter_btch_cind(masks_bool, ids_b, Nb, cinds, num_cls)


def uvN2aN(uv: np.ndarray) -> np.ndarray:
    return np.arccos(uv[0]) if uv[1] > 0 else -np.arccos(uv[0])


def uvsN2asN(uvs: np.ndarray) -> np.ndarray:
    alphas = np.arccos(uvs[..., 0])
    return np.where(uvs[..., 1] > 0, alphas, -alphas)


def asN2uvsN(alphas: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(alphas), np.sin(alphas)], axis=-1)


def uvT2aT(uv: torch.Tensor) -> torch.Tensor:
    return torch.acos(uv[0]) if uv[1] > 0 else -torch.acos(uv[0])


def uvsT2asT(uvs: torch.Tensor) -> torch.Tensor:
    alphas = torch.acos(uvs[..., 0])
    return torch.where(uvs[..., 1] > 0, alphas, -alphas)


def asT2uvsT(alphas: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(alphas), torch.sin(alphas)], dim=-1)


def xywhuvT2xywhaT(xywhuv: torch.Tensor) -> torch.Tensor:
    alpha = torch.acos(xywhuv[4:5]) if xywhuv[5:6] > 0 else -torch.acos(xywhuv[4:5])
    return torch.cat([xywhuv[:4], alpha], dim=0)


def xywhuvsT2xywhasT(xywhuvs: torch.Tensor) -> torch.Tensor:
    alphas = torch.acos(xywhuvs[..., 4:5])
    alphas = torch.where(xywhuvs[..., 5:6] > 0, alphas, -alphas)
    return torch.cat([xywhuvs[..., :4], alphas], dim=-1)


def xywhaT2xywhuvT(xywha: torch.Tensor) -> torch.Tensor:
    return torch.cat([xywha[:4], torch.cos(xywha[4:5]), torch.sin(xywha[4:5])], dim=0)


def xywhasT2xywhuvsT(xywhas: torch.Tensor) -> torch.Tensor:
    alphas = xywhas[..., 4:5]
    return torch.cat([xywhas[..., :4], torch.cos(alphas), torch.sin(alphas)], dim=-1)


def abclT_intersect(abcl1T: torch.Tensor, abcl2T: torch.Tensor) -> torch.Tensor:
    x = (abcl1T[:, 1] * abcl2T[:, 2] - abcl1T[:, 2] * abcl2T[:, 1]) / \
        (abcl1T[:, 0] * abcl2T[:, 1] - abcl1T[:, 1] * abcl2T[:, 0])
    y = (abcl1T[:, 0] * abcl2T[:, 2] - abcl1T[:, 2] * abcl2T[:, 0]) / \
        (abcl1T[:, 1] * abcl2T[:, 0] - abcl1T[:, 0] * abcl2T[:, 1])
    return torch.stack([x, y], dim=1)


def abclsT_intersect(abcls1T: torch.Tensor, abcls2T: torch.Tensor) -> torch.Tensor:
    x = (abcls1T[..., 1] * abcls2T[..., 2] - abcls1T[..., 2] * abcls2T[..., 1]) / \
        (abcls1T[..., 0] * abcls2T[..., 1] - abcls1T[..., 1] * abcls2T[..., 0])
    y = (abcls1T[..., 0] * abcls2T[..., 2] - abcls1T[..., 2] * abcls2T[..., 0]) / \
        (abcls1T[..., 1] * abcls2T[..., 0] - abcls1T[..., 0] * abcls2T[..., 1])
    return torch.stack([x, y], dim=-1)


def xlylT2abclT(xlylT: torch.Tensor) -> torch.Tensor:
    xlyl_ext = torch.cat([xlylT, xlylT[0:1]], dim=0)
    As = xlyl_ext[:-1, 1] - xlyl_ext[1:, 1]
    Bs = xlyl_ext[1:, 0] - xlyl_ext[:-1, 0]
    Cs = xlyl_ext[1:, 1] * xlyl_ext[:-1, 0] - xlyl_ext[:-1, 1] * xlyl_ext[1:, 0]
    return torch.stack([As, Bs, Cs], dim=1)


def xlylsT2abclsT(xlylsT: torch.Tensor) -> torch.Tensor:
    xlyls_ext = torch.cat([xlylsT, xlylsT[..., 0:1, :]], dim=-2)
    As = xlyls_ext[..., :-1, 1] - xlyls_ext[..., 1:, 1]
    Bs = xlyls_ext[..., 1:, 0] - xlyls_ext[..., :-1, 0]
    Cs = xlyls_ext[..., 1:, 1] * xlyls_ext[..., :-1, 0] - xlyls_ext[..., :-1, 1] * xlyls_ext[..., 1:, 0]
    return torch.stack([As, Bs, Cs], dim=-1)


def xlylT2areaT(xlylT: torch.Tensor) -> torch.Tensor:
    xlyl_ext = torch.cat([xlylT, xlylT[0:1]], dim=0)
    area = xlyl_ext[1:, 0] * xlyl_ext[:-1, 1] - xlyl_ext[:-1, 0] * xlyl_ext[1:, 1]
    area = torch.sum(area) / 2
    return area


def xlylsT2areasT(xlylsT: torch.Tensor) -> torch.Tensor:
    xlyls_ext = torch.cat([xlylsT, xlylsT[..., 0:1, :]], dim=-2)
    areas = xlyls_ext[..., 1:, 0] * xlyls_ext[..., :-1, 1] - xlyls_ext[..., :-1, 0] * xlyls_ext[..., 1:, 1]
    areas = torch.sum(areas, dim=-1) / 2
    return areas


def xlylT_intersect_coreT(xlyl1T: torch.Tensor, xlyl2T: torch.Tensor, eps: float = 1e-7) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    abcl1 = xlylT2abclT(xlyl1T)
    abcl2 = xlylT2abclT(xlyl2T)
    fltr_1pin2 = (xlyl1T[:, None, 0] * abcl2[None, :, 0] + xlyl1T[:, None, 1] * abcl2[None, :, 1]
                  + abcl2[None, :, 2]) <= eps
    msk_1pin2 = torch.all(fltr_1pin2, dim=1)
    if torch.all(msk_1pin2):
        return torch.full_like(fltr_1pin2, fill_value=False), abcl1, abcl2, \
               msk_1pin2, torch.full(size=(xlyl2T.size(0),), fill_value=False)
    fltr_2pin1 = (xlyl2T[None, :, 0] * abcl1[:, None, 0] + xlyl2T[None, :, 1] * abcl1[:, None, 1]
                  + abcl1[:, None, 2]) <= eps
    msk_2pin1 = torch.all(fltr_2pin1, dim=0)
    if torch.all(msk_2pin1):
        return torch.full_like(fltr_1pin2, fill_value=False), abcl1, abcl2, \
               msk_1pin2, msk_2pin1
    fltr_1lin2 = torch.cat([fltr_1pin2, fltr_1pin2[0:1]], dim=0)
    fltr_1lin2 = fltr_1lin2[1:] ^ fltr_1lin2[:-1]

    fltr_2lin1 = torch.cat([fltr_2pin1, fltr_2pin1[:, 0:1]], dim=1)
    fltr_2lin1 = fltr_2lin1[:, 1:] ^ fltr_2lin1[:, :-1]
    fltr_int = fltr_2lin1 * fltr_1lin2
    return fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1


def xlylsT_intersect_coresT(xlyls1T: torch.Tensor, xlyls2T: torch.Tensor, eps: float = 1e-7) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    abcls1 = xlylsT2abclsT(xlyls1T)
    abcls2 = xlylsT2abclsT(xlyls2T)
    fltr_1pin2 = (xlyls1T[..., None, 0] * abcls2[..., None, :, 0] + xlyls1T[..., None, 1] * abcls2[..., None, :, 1]
                  + abcls2[..., None, :, 2]) < eps
    msk_1pin2 = torch.all(fltr_1pin2, dim=-1)
    fltr_2pin1 = (xlyls2T[..., None, :, 0] * abcls1[..., None, 0] + xlyls2T[..., None, :, 1] * abcls1[..., None, 1]
                  + abcls1[..., None, 2]) < eps
    msk_2pin1 = torch.all(fltr_2pin1, dim=-2)

    fltr_1lin2 = torch.cat([fltr_1pin2, fltr_1pin2[..., 0:1, :]], dim=-2)
    fltr_1lin2 = fltr_1lin2[..., 1:, :] ^ fltr_1lin2[..., :-1, :]

    fltr_2lin1 = torch.cat([fltr_2pin1, fltr_2pin1[..., 0:1]], dim=-1)
    fltr_2lin1 = fltr_2lin1[..., 1:] ^ fltr_2lin1[..., :-1]

    fltr_int = fltr_2lin1 * fltr_1lin2
    return fltr_int, abcls1, abcls2, msk_1pin2, msk_2pin1


def intersect_coreT2xlylT(xlyl1T: torch.Tensor, xlyl2T: torch.Tensor, fltr_int: torch.Tensor,
                          abcl1: torch.Tensor, abcl2: torch.Tensor, msk_1pin2: torch.Tensor,
                          msk_2pin1: torch.Tensor) -> torch.Tensor:
    if torch.all(msk_1pin2):
        return xlyl1T
    elif torch.all(msk_2pin1):
        return xlyl2T
    # 节点排序
    idls1, idls2 = torch.nonzero(fltr_int, as_tuple=True)
    xlyl_int = abclT_intersect(abcl1[idls1], abcl2[idls2])
    dists = torch.sum(torch.abs(xlyl_int - xlyl1T[idls1]), dim=1) + idls1 * torch.sum(torch.abs(abcl1[:, :2]))
    order = torch.argsort(dists)
    idls2 = idls2[order] + 1
    idls1 = idls1 + 1
    xlyl_int = xlyl_int[order]
    # 按序遍历
    num1 = xlyl1T.size(0)
    num2 = xlyl2T.size(0)
    num_int = len(idls1)
    idls1_nxt = torch.cat([idls1[1:], idls1[0:1]])
    idls1_nxt = torch.where(idls1_nxt < idls1, idls1_nxt + num1, idls1_nxt)
    idls2_nxt = torch.cat([idls2[1:], idls2[0:1]])
    idls2_nxt = torch.where(idls2_nxt < idls2, idls2_nxt + num2, idls2_nxt)
    ids = [torch.zeros(size=(0,), dtype=torch.long)]
    for i in range(num_int):
        ids.append(torch.Tensor([i + num1 + num2]).long())
        ids.append(torch.arange(int(idls1[i]), int(idls1_nxt[i]), dtype=torch.long) % num1)
        ids.append(torch.arange(int(idls2[i]), int(idls2_nxt[i]), dtype=torch.long) % num2 + num1)
    ids = torch.cat(ids, dim=0)
    pnts = torch.cat([xlyl1T, xlyl2T, xlyl_int])
    msks = torch.cat(
        [msk_1pin2, msk_2pin1, torch.full(size=(xlyl_int.size(0),), fill_value=True, device=xlyl_int.device)])
    xlyl_final = pnts[ids][msks[ids]]
    return xlyl_final


def asT2matsT(asT: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(asT), torch.sin(asT)
    mat = torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([-sin, cos], dim=-1)], dim=-2)
    return mat


def asN2matsN(asN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(asN), np.sin(asN)
    mat = np.stack([np.stack([cos, sin], axis=-1), np.stack([-sin, cos], axis=-1)], axis=-2)
    return mat


# </editor-fold>Z

# <editor-fold desc='list边界格式转换'>

def xyxyL2xywhL(xyxy: list) -> list:
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def xywhL2xyxyL(xywh: list) -> list:
    xc, yc, w_2, h_2 = xywh[0], xywh[1], xywh[2], xywh[3]
    return [xc - w_2, yc - h_2, xc + w_2, yc + h_2]


def xyxyL2xywhN(xyxy: list) -> np.ndarray:
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def xywhL2xyxyN(xywh: list) -> np.ndarray:
    xc, yc, w_2, h_2 = xywh[0], xywh[1], xywh[2], xywh[3]
    return np.array([xc - w_2, yc - h_2, xc + w_2, yc + h_2])


# </editor-fold>


# <editor-fold desc='图像格式转换'>


def imgP2imgN(imgP: PIL.Image.Image) -> np.ndarray:
    if imgP.size[0] == 0 or imgP.size[1] == 0:
        if imgP.mode == 'L':
            return np.zeros(shape=(imgP.size[1], imgP.size[0]))
        elif imgP.mode == 'RGB':
            return np.zeros(shape=(imgP.size[1], imgP.size[0], 3))
        elif imgP.mode == 'RGBA':
            return np.zeros(shape=(imgP.size[1], imgP.size[0], 4))
        else:
            raise Exception('err num ' + str(imgP.mode))
    imgN = np.array(imgP)
    return imgN


def imgN2imgP(imgN: np.ndarray) -> PIL.Image.Image:
    if len(imgN.shape) == 2:
        imgP_tp = 'L'
    elif len(imgN.shape) == 3 and imgN.shape[2] == 1:
        imgP_tp = 'L'
        imgN = imgN.squeeze(axis=2)
    elif imgN.shape[2] == 3:
        imgP_tp = 'RGB'
    elif imgN.shape[2] == 4:
        imgP_tp = 'RGBA'
    else:
        raise Exception('err num ' + str(imgN.shape))
    imgN = Image.fromarray(imgN.astype(np.uint8), mode=imgP_tp)
    return imgN


def imgN2imgT(imgN: np.ndarray) -> torch.Tensor:
    # imgN = np.transpose(imgN, (2, 0, 1))  # HWC转CHW
    imgT = torch.from_numpy(imgN).float()
    imgT = imgT.permute((2, 0, 1)) / 255
    imgT = imgT[None, :]
    return imgT


def imgT2imgN(imgT: torch.Tensor) -> np.ndarray:
    imgT = imgT * 255
    imgN = imgT.detach().cpu().numpy().astype(np.uint8)
    if len(imgN.shape) == 4 and imgN.shape[0] == 1:
        imgN = imgN.squeeze(axis=0)
    imgN = np.transpose(imgN, (1, 2, 0))  # CHW转为HWC
    return imgN


# from torchvision import transforms
# transforms.ToTensor

def imgP2imgT(imgP: PIL.Image.Image) -> torch.Tensor:
    # print('sdsd')
    imgT = torch.from_numpy(np.array(imgP)).float()
    imgT = imgT.permute((2, 0, 1)) / 255
    imgT = imgT[None, :]
    # imgN = imgP2imgN(imgP)
    # imgT = imgN2imgT(imgN)
    return imgT


def imgT2imgP(imgT: torch.Tensor) -> PIL.Image.Image:
    imgN = imgT2imgN(imgT)
    imgP = imgN2imgP(imgN)
    return imgP


def imgsN2imgsP(imgs: list) -> list:
    for i in range(len(imgs)):
        imgs[i] = Image.fromarray(imgs[i].astype('uint8')).convert('RGB')
    return imgs


def imgsP2imgsN(imgs: list) -> list:
    for i in range(len(imgs)):
        imgs[i] = np.array(imgs[i])
    return imgs


def img2imgT(img) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        return imgN2imgT(img)
    elif isinstance(img, PIL.Image.Image):
        return imgP2imgT(img)
    elif isinstance(img, torch.Tensor):
        return img
    else:
        raise Exception('err type ' + img.__class__.__name__)


def img2imgP(img) -> PIL.Image.Image:
    if isinstance(img, np.ndarray):
        return imgN2imgP(img)
    elif isinstance(img, PIL.Image.Image):
        return img
    elif isinstance(img, torch.Tensor):
        return imgT2imgP(img)
    else:
        raise Exception('err type ' + img.__class__.__name__)


def img2imgN(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, PIL.Image.Image):
        return imgP2imgN(img)
    elif isinstance(img, torch.Tensor):
        return imgT2imgN(img)
    else:
        raise Exception('err type ' + img.__class__.__name__)


# </editor-fold>

# <editor-fold desc='图像绘制处理'>
def xlylN2maskP(xlylN: np.ndarray, xyxy_rgn: np.ndarray) -> PIL.Image.Image:
    size = (xyxy_rgn[2:4] - xyxy_rgn[:2]).astype(np.int32)
    maskP = Image.new('1', tuple(size), 0)
    xlylN_ref = xlylN - xyxy_rgn[:2]
    PIL.ImageDraw.Draw(maskP).polygon(list(xlylN_ref.reshape(-1)), fill=1)
    return maskP


def xywhaN2maskP(xywhaN: np.ndarray, xyxy_rgn: np.ndarray) -> PIL.Image.Image:
    xlylN = xywhaN2xlylN(xywhaN)
    maskP = xlylN2maskP(xlylN=xlylN, xyxy_rgn=xyxy_rgn)
    return maskP


def xywhuvN2maskP(xywhuvN: np.ndarray, xyxy_rgn: np.ndarray) -> PIL.Image.Image:
    xlylN = xywhuvN2xlylN(xywhuvN)
    maskP = xlylN2maskP(xlylN=xlylN, xyxy_rgn=xyxy_rgn)
    return maskP


def imgP_fill_xlylN(imgP: PIL.Image.Image, xlylN: np.ndarray, color: tuple = (255, 255, 255)) -> PIL.Image.Image:
    draw = ImageDraw.Draw(imgP)
    draw.polygon(list(xlylN.reshape(-1)), outline=None, fill=color)
    return imgP


def imgP_fill_xywhaN(imgP: PIL.Image.Image, xywhaN: np.ndarray, color: tuple = (255, 255, 255)) -> PIL.Image.Image:
    xlylN = xywhaN2xlylN(xywhaN)
    imgP_fill_xlylN(imgP=imgP, xlylN=xlylN, color=color)
    return imgP


def imgP_crop_xywhaN(imgP: PIL.Image.Image, xywhaN: np.ndarray) -> (PIL.Image.Image, PIL.Image.Image):
    xlylN = xywhaN2xlylN(xywhaN)
    return imgP_crop_xlylN(imgP=imgP, xlylN=xlylN)


def imgP_crop_xlylN(imgP: PIL.Image.Image, xlylN: np.ndarray) -> (PIL.Image.Image, PIL.Image.Image):
    xyxyN = xlylN2xyxyN(xlylN).astype(np.int32)
    patch = imgP.crop(xyxyN)
    maskP = Image.new('1', size=patch.size, color=0)
    xlylN_ref = xlylN - xyxyN[:2]
    PIL.ImageDraw.Draw(maskP).polygon(list(xlylN_ref.reshape(-1)), fill=1)
    return patch, maskP


def _build_rflip_data(size: tuple, alpha: float, flip: bool = False, vflip: bool = False) -> np.ndarray:
    if not flip and not vflip:
        return np.array([1, 0, 0, 0, 1, 0])
    elif flip and vflip:
        pc = np.array(size) / 2
        return np.array([-1, 0, pc[0] * 2, 0, -1, pc[1] * 2])
    cos, sin = np.cos(alpha), np.sin(alpha)
    cos2, sin2, cossin = cos ** 2, sin ** 2, cos * sin
    pc = np.array(size) / 2
    mat = np.array([[cos2 - sin2, 2 * cossin], [2 * cossin, sin2 - cos2]])
    mat = mat if flip else -mat
    bias = pc - pc @ mat
    data = np.concatenate([mat[0], [bias[0]], mat[1], [bias[1]]])
    return data


def imgP_rflip(imgP: PIL.Image.Image, alpha: float, flip: bool = False, vflip: bool = False,
               resample=Image.BILINEAR) -> PIL.Image.Image:
    if not flip and not vflip:
        return imgP
    data = _build_rflip_data(size=imgP.size, alpha=alpha, flip=flip, vflip=vflip)
    imgP = imgP.transform(size=imgP.size, method=Image.AFFINE, data=data, resample=resample)
    return imgP


def imgP_rflip_paste_xywhaN(imgP: PIL.Image.Image, xywhaN: np.ndarray, flip: bool = False,
                            vflip: bool = False, resample=Image.BILINEAR) -> PIL.Image.Image:
    if not flip and not vflip:
        return imgP
    xlylN = xywhaN2xlylN(xywhaN)
    patch, maskP = imgP_crop_xlylN(imgP, xlylN)
    data = _build_rflip_data(size=patch.size, alpha=xywhaN[4], flip=flip, vflip=vflip)
    patch = patch.transform(size=patch.size, method=Image.AFFINE, data=data, resample=resample)
    maskP = maskP.transform(size=maskP.size, method=Image.AFFINE, data=data, resample=resample)
    pnt = np.min(xlylN, axis=0).astype(np.int32)
    imgP.paste(patch, box=tuple(pnt), mask=maskP)
    return imgP


# </editor-fold>

# <editor-fold desc='图像批次处理'>
def imgP_lmtsize(imgP: PIL.Image.Image, max_size: tuple, resample=Image.BILINEAR) -> (PIL.Image.Image, float):
    size = np.array(imgP.size)
    ratio = min(np.array(max_size) / size)
    if ratio == 1:
        return imgP, ratio
    imgP = imgP.resize(size=tuple((size * ratio).astype(np.int32)), resample=resample)
    return imgP, ratio


def imgP_lmtsize_pad(imgP: PIL.Image.Image, max_size: tuple, pad_val: int = 127, resample=Image.BILINEAR) \
        -> (PIL.Image.Image, float):
    if imgP.size == max_size:
        return imgP, 1
    imgN = imgP2imgN(imgP)
    aspect = max_size[0] / max_size[1]
    imgN = imgN_pad_aspect(imgN, aspect=aspect, pad_val=pad_val)
    imgP = imgN2imgP(imgN)
    ratio = min(max_size[0] / imgP.size[0], max_size[1] / imgP.size[1])
    imgP = imgP.resize(max_size, resample=resample)
    return imgP, ratio


def imgT_lmtsize_pad(imgT: torch.Tensor, max_size: tuple, pad_val: int = 127) \
        -> (torch.Tensor, float):
    aspect = max_size[0] / max_size[1]
    imgT = imgT_pad_aspect(imgT, aspect=aspect, pad_val=pad_val)
    ratio = min(max_size[0] / imgT.size(3), max_size[1] / imgT.size(2))
    imgT = F.interpolate(imgT, size=(max_size[1], max_size[0]))
    return imgT, ratio


def imgN_pad_aspect(imgN: np.ndarray, aspect: float, pad_val: int = 127) -> np.ndarray:
    h, w, _ = imgN.shape
    if w / h > aspect:
        imgN = np.pad(imgN, pad_width=((0, int(w / aspect - h)), (0, 0), (0, 0)), constant_values=pad_val)
    elif w / h < aspect:
        imgN = np.pad(imgN, pad_width=((0, 0), (0, int(h * aspect - w)), (0, 0)), constant_values=pad_val)
    return imgN


def imgT_pad_aspect(imgT: torch.Tensor, aspect: float, pad_val: int = 127) -> torch.Tensor:
    _, _, h, w = imgT.size()
    if w / h > aspect:
        imgT = F.pad(imgT, pad=(0, 0, 0, int(w / aspect - h)), value=pad_val / 255)
    elif w / h < aspect:
        imgT = F.pad(imgT, pad=(0, int(h * aspect - w), 0, 0), value=pad_val / 255)
    return imgT


def imgs2imgsT(imgs: list, img_size: tuple, pad_val: int = 127) -> (torch.Tensor, np.ndarray):
    if isinstance(imgs, torch.Tensor):
        if imgs.size(3) == img_size[0] and imgs.size(2) == img_size[1]:
            return imgs, np.ones(shape=len(imgs))
        else:
            imgsT, ratio = imgT_lmtsize_pad(imgT=imgs, max_size=img_size, pad_val=pad_val)
            return imgsT, np.full(shape=len(imgs), fill_value=ratio)
    imgsT = []
    ratios = []
    for img in imgs:
        imgT = img2imgT(img)
        imgT, ratio = imgT_lmtsize_pad(imgT=imgT, max_size=img_size, pad_val=pad_val)
        imgsT.append(imgT)
        ratios.append(ratio)
    imgsT = torch.cat(imgsT, dim=0)
    ratios = np.array(ratios)
    return imgsT, ratios


def img2size(img) -> tuple:
    if isinstance(img, PIL.Image.Image):
        return img.size
    elif isinstance(img, np.ndarray):
        return (img.shape[1], img.shape[0])
    elif isinstance(img, torch.Tensor):
        return (img.size(-1), img.size(-2))
    else:
        raise Exception('err type ' + img.__class__.__name__)

# </editor-fold>

# if __name__ == '__main__':
#     xywha = np.array([20, 20, 50, 40, 1])
#     xlyl = xywhaN2xlylN(xywha)
#     xywha2 = xlylN2xywhaN(xlyl)
