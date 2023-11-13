import torchvision

from utils.label import *


class OPR_TYPE:
    IOU = 'iou'
    GIOU = 'giou'
    CIOU = 'ciou'
    DIOU = 'diou'
    IAREA = 'iarea'
    IRATE = 'irate'
    KL = 'kl'
    KLIOU = 'kliou'


class IOU_TYPE:
    IOU = OPR_TYPE.IOU
    GIOU = OPR_TYPE.GIOU
    CIOU = OPR_TYPE.CIOU
    DIOU = OPR_TYPE.DIOU
    KLIOU = OPR_TYPE.KLIOU


# <editor-fold desc='numpy形状运算'>
def _align_shapeN(arr1, arr2, last_axis=1):
    assert arr1.shape[-last_axis:] == arr2.shape[-last_axis:], 'shape err'
    shape1_ = arr1.shape[:-last_axis]
    shape2_ = arr2.shape[:-last_axis]
    val_shape_len = len(shape1_) + len(shape2_)
    full_shape_len = val_shape_len + last_axis
    trans = list(range(len(shape2_), val_shape_len)) + \
            list(range(len(shape2_))) + \
            list(range(val_shape_len, full_shape_len))
    arr1 = np.broadcast_to(arr1, list(shape2_) + list(arr1.shape)).transpose(trans)
    arr2 = np.broadcast_to(arr2, list(shape1_) + list(arr2.shape))
    return arr1, arr2


def ropr_arr_xywhsN(xywhs1, xywhs2, opr_type=OPR_TYPE.IOU):
    xymin1, xymax1 = xywhs1[..., :2] - xywhs1[..., 2:4] / 2, xywhs1[..., :2] + xywhs1[..., 2:4] / 2
    xymin2, xymax2 = xywhs2[..., :2] - xywhs2[..., 2:4] / 2, xywhs2[..., :2] + xywhs2[..., 2:4] / 2
    xymax_min = np.minimum(xymax1, xymax2)
    xymin_max = np.maximum(xymin1, xymin2)
    whi = np.maximum(xymax_min - xymin_max, 0)
    areai = np.prod(whi, axis=-1)
    if opr_type == OPR_TYPE.IAREA:
        return areai
    area1 = np.prod(xywhs1[..., 2:4], axis=-1)
    area2 = np.prod(xywhs2[..., 2:4], axis=-1)
    if opr_type == OPR_TYPE.IRATE:
        return areai / area2
    areau = area1 + area2 - areai
    iou = areai / areau
    if opr_type is None or opr_type == OPR_TYPE.IOU:
        return iou
    xymax_max = np.maximum(xymax1, xymax2)
    xymin_min = np.minimum(xymin1, xymin2)
    whb = xymax_max - xymin_min
    areab = np.prod(whb, axis=-1)
    if opr_type == OPR_TYPE.GIOU:
        return iou - (areab - areau) / areab
    diagb = np.sum(whb ** 2, axis=-1)
    diagc = np.sum((xywhs1[..., :2] - xywhs2[..., :2]) ** 2, axis=-1)
    diou = iou - diagc / diagb
    if opr_type == OPR_TYPE.DIOU:
        return diou
    r1 = np.arctan(xywhs1[..., 2] / xywhs1[..., 3])
    r2 = np.arctan(xywhs2[..., 2] / xywhs2[..., 3])
    v = ((r1 - r2) * 2 / math.pi) ** 2
    alpha = v * 2 / (1 - areai / areau + v + 1e-7)
    if opr_type == OPR_TYPE.CIOU:
        return diou - v * alpha
    raise Exception('err iou type ' + str(opr_type))


def ropr_arr_xyxysN(xyxys1, xyxys2, opr_type=OPR_TYPE.IOU):
    xymax_min = np.minimum(xyxys1[..., 2:4], xyxys2[..., 2:4])
    xymin_max = np.maximum(xyxys1[..., :2], xyxys2[..., :2])
    whi = np.maximum(xymax_min - xymin_max, 0)
    areai = np.prod(whi, axis=-1)
    if opr_type == OPR_TYPE.IAREA:
        return areai
    area1 = np.prod(xyxys1[..., 2:4] - xyxys1[..., :2], axis=-1)
    area2 = np.prod(xyxys2[..., 2:4] - xyxys2[..., :2], axis=-1)
    if opr_type == OPR_TYPE.IRATE:
        return areai / area2
    areau = area1 + area2 - areai
    iou = areai / areau
    if opr_type is None or opr_type == OPR_TYPE.IOU:
        return iou
    xymax_max = np.maximum(xyxys1[..., 2:4], xyxys2[..., 2:4])
    xymin_min = np.minimum(xyxys1[..., :2], xyxys2[..., :2])
    whb = xymax_max - xymin_min
    areab = np.prod(whb, axis=-1)
    if opr_type == OPR_TYPE.GIOU:
        return iou - (areab - areau) / areab
    xyc1 = (xyxys1[..., 2:4] + xyxys1[..., :2]) / 2
    xyc2 = (xyxys2[..., 2:4] + xyxys2[..., :2]) / 2
    diagb = np.sum(whb ** 2, axis=-1)
    diagc = np.sum((xyc1 - xyc2) ** 2, axis=-1)
    diou = iou - diagc / diagb
    if opr_type == OPR_TYPE.DIOU:
        return diou
    wh1 = xyxys1[..., 2:4] - xyxys1[..., :2]
    wh2 = xyxys2[..., 2:4] - xyxys2[..., :2]
    r1 = np.arctan(wh1[..., 0] / wh1[..., 1])
    r2 = np.arctan(wh2[..., 0] / wh2[..., 0])
    v = ((r1 - r2) * 2 / math.pi) ** 2
    alpha = v * 2 / (1 - areai / areau + v + 1e-7)
    if opr_type == OPR_TYPE.CIOU:
        return diou - v * alpha
    raise Exception('err iou type ' + str(opr_type))


def _iarea_arr_xlylsN_shapely(xlyls1, xlyls2):
    assert xlyls1.shape == xlyls2.shape, 'shape err'
    num = xlyls1.shape[0]
    if len(xlyls1.shape) == 3:
        num_samp = xlyls1.shape[0]
        iareas = np.zeros(shape=num_samp)
        fltrs_int, abcls1, abcls2, msks_1pin2, msks_2pin1 = xlylsN_intersect_coresN(xlyls1, xlyls2, eps=1e-7)
        for m, (xlyl1, xlyl2, fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1) \
                in enumerate(zip(xlyls1, xlyls2, fltrs_int, abcls1, abcls2, msks_1pin2, msks_2pin1)):
            xlyl_int = intersect_coreN2xlylN(xlyl1, xlyl2, fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1)
            area = xlylN2areaN(xlyl_int)
            iareas[m] = area
        return iareas
    else:
        iareas = [np.zeros(shape=([0] + list(xlyls1.shape[1:-2])))]
        for i in range(num):
            iareas.append(_iarea_arr_xlylsN_shapely(xlyls1[i], xlyls2[i])[None, :])
        iareas = np.concatenate(iareas, axis=0)
        return iareas


def ropr_arr_xlylsN(xlyls1, xlyls2, opr_type=OPR_TYPE.IOU):
    iareas = _iarea_arr_xlylsN_shapely(xlyls1, xlyls2)
    if opr_type == OPR_TYPE.IAREA:
        return iareas
    areas1 = xlylsN2areasN(xlyls1)
    areas2 = xlylsN2areasN(xlyls2)
    uareas = areas1 + areas2 - iareas
    ious = iareas / (uareas + 1e-16)
    if opr_type == OPR_TYPE.IOU:
        return ious
    raise Exception('err iou type ' + str(opr_type))


def ropr_arr_xlyls(xlyls1, xlyls2, opr_type=OPR_TYPE.IOU):
    res = []
    for xlyl1, xlyl2 in zip(xlyls1, xlyls2):
        fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1 = xlylN_intersect_coreN(xlyl1, xlyl2, eps=1e-7)
        xlyl_int = intersect_coreN2xlylN(xlyl1, xlyl2, fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1)
        iarea = xlylN2areaN(xlyl_int)
        if opr_type == OPR_TYPE.IAREA:
            res.append(iarea)
            continue
        area1 = xlylN2areaN(xlyl1)
        area2 = xlylN2areaN(xlyl2)
        uarea = area1 + area2 - iarea
        iou = iarea / (uarea + 1e-16)
        if opr_type == OPR_TYPE.IOU:
            res.append(iou)
            continue
        raise Exception('err iou type ' + str(opr_type))
    res = np.array(res)
    return res


def ropr_arr_xywhasN(xywhas1, xywhas2, opr_type=OPR_TYPE.KL):
    if opr_type == OPR_TYPE.IAREA or opr_type == OPR_TYPE.IOU:
        xlyls1 = xywhasN2xlylsN(xywhas1)
        xlyls2 = xywhasN2xlylsN(xywhas2)
        iarea = _iarea_arr_xlylsN_shapely(xlyls1, xlyls2)
        if opr_type == OPR_TYPE.IAREA:
            return iarea
        area1 = xywhas1[..., 2] * xywhas1[..., 3]
        area2 = xywhas2[..., 2] * xywhas2[..., 3]
        uarea = area1 + area2 - iarea
        iou = iarea / (uarea + 1e-16)
        return iou
    w1, h1, a1, = xywhas1[..., 2], xywhas1[..., 3], xywhas1[..., 4]
    w2, h2, a2, = xywhas2[..., 2], xywhas2[..., 3], xywhas2[..., 4]
    x_dt, y_dt = xywhas2[..., 0] - xywhas1[..., 0], xywhas2[..., 1] - xywhas1[..., 1]
    wr, hr, a_dt = w2 / w1, h2 / h1, a2 - a1
    wh21, wh12, hw12, hw21 = w2 / h1, w1 / h2, h1 / w2, h2 / w1
    cos1, sin1 = np.cos(a1), np.sin(a1)
    cos2, sin2 = np.cos(a2), np.sin(a2)
    cos_dt, sin_dt = np.cos(a_dt), np.sin(a_dt)
    p1 = ((x_dt * cos1 + y_dt * sin1) / w1) ** 2 + ((y_dt * cos1 - x_dt * sin1) / h1) ** 2 \
         + ((x_dt * cos2 + y_dt * sin2) / w2) ** 2 + ((y_dt * cos2 - x_dt * sin2) / h2) ** 2
    p2 = (wr ** 2 + 1 / wr ** 2 + hr ** 2 + 1 / hr ** 2) * cos_dt ** 2 \
         + (wh21 ** 2 + wh12 ** 2 + hw12 ** 2 + hw21 ** 2) * sin_dt ** 2
    kl_dist = p1 + p2 / 4 - 1
    if opr_type == OPR_TYPE.KL:
        return kl_dist
    kliou = 1 / (kl_dist + 1)
    if opr_type == OPR_TYPE.KLIOU:
        return kliou
    raise Exception('err iou type ' + str(opr_type))


def ropr_arr_xywhuvsN(xywhuvs1, xywhuvs2, opr_type=OPR_TYPE.KL):
    w1, h1, cos1, sin1, = xywhuvs1[..., 2], xywhuvs1[..., 3], xywhuvs1[..., 4], xywhuvs1[..., 5]
    w2, h2, cos2, sin2, = xywhuvs2[..., 2], xywhuvs2[..., 3], xywhuvs2[..., 4], xywhuvs2[..., 5]
    x_dt, y_dt = xywhuvs2[..., 0] - xywhuvs1[..., 0], xywhuvs2[..., 1] - xywhuvs1[..., 1]
    wr, hr = w2 / w1, h2 / h1
    wh21, wh12, hw12, hw21 = w2 / h1, w1 / h2, h1 / w2, h2 / w1
    cos_dt, sin_dt = cos1 * cos2 + sin1 * sin2, sin2 * cos1 - sin1 * cos2
    p1 = ((x_dt * cos1 + y_dt * sin1) / w1) ** 2 + ((y_dt * cos1 - x_dt * sin1) / h1) ** 2 \
         + ((x_dt * cos2 + y_dt * sin2) / w2) ** 2 + ((y_dt * cos2 - x_dt * sin2) / h2) ** 2
    p2 = (wr ** 2 + 1 / wr ** 2 + hr ** 2 + 1 / hr ** 2) * cos_dt ** 2 \
         + (wh21 ** 2 + wh12 ** 2 + hw12 ** 2 + hw21 ** 2) * sin_dt ** 2
    kl_dist = p1 + p2 / 4 - 1
    if opr_type == OPR_TYPE.KL:
        return kl_dist
    kliou = 1 / (kl_dist + 1)
    if opr_type == OPR_TYPE.KLIOU:
        return kliou
    raise Exception('err iou type ' + str(opr_type))


def ropr_arr_xywhabsN(xywhabs1, xywhabs2, iou_type=OPR_TYPE.IOU):
    xlyls1 = xywhasN2xlylsN(xywhabs1)
    xlyls2 = xywhasN2xlylsN(xywhabs2)
    iarea = _iarea_arr_xlylsN_shapely(xlyls1, xlyls2)
    if iou_type == OPR_TYPE.IAREA:
        return iarea
    area1 = xywhabs1[..., 2] * xywhabs1[..., 3] * np.abs(np.sin(xywhabs1[..., 5] - xywhabs1[..., 4]))
    area2 = xywhabs2[..., 2] * xywhabs2[..., 3] * np.abs(np.sin(xywhabs2[..., 5] - xywhabs2[..., 4]))
    uarea = area1 + area2 - iarea
    iou = iarea / (uarea + 1e-16)
    if iou_type == OPR_TYPE.IOU:
        return iou
    raise Exception('err iou type ' + str(iou_type))


def ropr_mat_xyxysN(xyxys1, xyxys2, opr_type=OPR_TYPE.IOU):
    xyxys1, xyxys2 = _align_shapeN(xyxys1, xyxys2, last_axis=1)
    return ropr_arr_xyxysN(xyxys1, xyxys2, opr_type)


def ropr_mat_xywhsN(xywhs1, xywhs2, opr_type=OPR_TYPE.IOU):
    xywhs1, xywhs2 = _align_shapeN(xywhs1, xywhs2, last_axis=1)
    return ropr_arr_xywhsN(xywhs1, xywhs2, opr_type)


def ropr_mat_xywhasN(xywhas1, xywhas2, opr_type=OPR_TYPE.IOU):
    xywhas1, xywhas2 = _align_shapeN(xywhas1, xywhas2, last_axis=1)
    return ropr_arr_xywhasN(xywhas1, xywhas2, opr_type)


def ropr_mat_xlylsN(xlyls1, xlyls2, opr_type=OPR_TYPE.IOU):
    xlyls1, xlyls2 = _align_shapeN(xlyls1, xlyls2, last_axis=2)
    return ropr_arr_xlylsN(xlyls1, xlyls2, opr_type)


# </editor-fold>

# <editor-fold desc='numpy形状运算扩展'>
def ropr_arr_xyxysN_maskNs(xyxys1, masks1, xyxys2, masks2, opr_type=OPR_TYPE.IOU):
    assert opr_type in [OPR_TYPE.IAREA, OPR_TYPE.IOU], 'opr err ' + str(opr_type)
    iarea = ropr_arr_xyxysN(xyxys1, xyxys2)
    assert len(iarea.shape) == 1, 'dim err ' + str(iarea.shape)
    xys1_min, xys1_max = xyxys1[..., :2].astype(np.int32), xyxys1[..., 2:].astype(np.int32)
    xys2_min, xys2_max = xyxys2[..., :2].astype(np.int32), xyxys2[..., 2:].astype(np.int32)
    xys_min = np.minimum(xys1_min, xys2_min)
    xys_max = np.maximum(xys1_max, xys2_max)
    pds1_min, pds1_max = xys1_min - xys_min, xys_max - xys1_max
    pds2_min, pds2_max = xys2_min - xys_min, xys_max - xys2_max
    for i in range(iarea.shape[0]):
        if iarea[i] == 0:
            continue
        mask1 = np.pad(masks1[i], ((pds1_min[i, 1], pds1_max[i, 1]), (pds1_min[i, 0], pds1_max[i, 0])), 'constant')
        mask2 = np.pad(masks2[i], ((pds2_min[i, 1], pds2_max[i, 1]), (pds2_min[i, 0], pds2_max[i, 0])), 'constant')
        iarea[i] = np.sum(mask1 * mask2)
    if opr_type == OPR_TYPE.IAREA:
        return iarea
    area1 = np.array([np.sum(mask1) for mask1 in masks1])
    area2 = np.array([np.sum(mask2) for mask2 in masks2])
    uarea = area1 + area2 - iarea
    iou = iarea / (uarea + 1e-7)
    if opr_type == OPR_TYPE.IOU:
        return iou
    raise Exception('err iou type ' + str(opr_type))


def ropr_mat_xyxysN_maskNs(xyxys1, masks1, xyxys2, masks2, opr_type=OPR_TYPE.IOU):
    assert opr_type in [OPR_TYPE.IAREA, OPR_TYPE.IOU], 'opr err ' + str(opr_type)
    xyxys1, xyxys2 = _align_shapeN(xyxys1, xyxys2, last_axis=1)
    iarea = ropr_arr_xyxysN(xyxys1, xyxys2)
    assert len(iarea.shape) == 2, 'dim err ' + str(iarea.shape)
    xys1_min, xys1_max = xyxys1[..., :2].astype(np.int32), xyxys1[..., 2:].astype(np.int32)
    xys2_min, xys2_max = xyxys2[..., :2].astype(np.int32), xyxys2[..., 2:].astype(np.int32)
    xys_min = np.minimum(xys1_min, xys2_min)
    xys_max = np.maximum(xys1_max, xys2_max)
    pds1_min, pds1_max = xys1_min - xys_min, xys_max - xys1_max
    pds2_min, pds2_max = xys2_min - xys_min, xys_max - xys2_max
    for i in range(iarea.shape[0]):
        for j in range(iarea.shape[1]):
            if iarea[i, j] == 0:
                continue
            mask1 = np.pad(masks1[i], ((pds1_min[i, j, 1], pds1_max[i, j, 1]), (pds1_min[i, j, 0], pds1_max[i, j, 0])),
                           'constant')
            mask2 = np.pad(masks2[j], ((pds2_min[i, j, 1], pds2_max[i, j, 1]), (pds2_min[i, j, 0], pds2_max[i, j, 0])),
                           'constant')
            iarea[i, j] = np.sum(mask1 * mask2)
    if opr_type == OPR_TYPE.IAREA:
        return iarea
    area1 = np.array([np.sum(mask1) for mask1 in masks1])[:, None]
    area2 = np.array([np.sum(mask2) for mask2 in masks2])[None, :]
    uarea = area1 + area2 - iarea
    iou = iarea / (uarea + 1e-7)
    if opr_type == OPR_TYPE.IOU:
        return iou
    raise Exception('err iou type ' + str(opr_type))


# </editor-fold>

# <editor-fold desc='torch形状运算'>
def _align_shapeT(ten1, ten2, last_axis=1):
    assert list(ten1.size())[-last_axis:] == list(ten2.size())[-last_axis:], 'shape err'
    shape1_ = list(ten1.size())[:-last_axis]
    shape2_ = list(ten2.size())[:-last_axis]
    val_shape_len = len(shape1_) + len(shape2_)
    full_shape_len = val_shape_len + last_axis
    trans = list(range(len(shape2_), val_shape_len)) + \
            list(range(len(shape2_))) + \
            list(range(val_shape_len, full_shape_len))
    ten1 = torch.broadcast_to(ten1, list(shape2_) + list(ten1.shape)).permute(*trans)
    ten2 = torch.broadcast_to(ten2, list(shape1_) + list(ten2.shape))
    return ten1, ten2


def ropr_arr_xywhsT(xywhs1, xywhs2, opr_type=OPR_TYPE.IOU):
    xymin1, xymax1 = xywhs1[..., :2] - xywhs1[..., 2:4] / 2, xywhs1[..., :2] + xywhs1[..., 2:4] / 2
    xymin2, xymax2 = xywhs2[..., :2] - xywhs2[..., 2:4] / 2, xywhs2[..., :2] + xywhs2[..., 2:4] / 2
    xymax_min = torch.minimum(xymax1, xymax2)
    xymin_max = torch.maximum(xymin1, xymin2)
    whi = torch.clamp(xymax_min - xymin_max, min=0)
    areai = torch.prod(whi, dim=-1)
    if opr_type == OPR_TYPE.IAREA:
        return areai
    area1 = torch.prod(xywhs1[..., 2:4], dim=-1)
    area2 = torch.prod(xywhs2[..., 2:4], dim=-1)
    areau = area1 + area2 - areai
    iou = areai / areau
    if opr_type is None or opr_type == OPR_TYPE.IOU:
        return iou
    xymax_max = torch.maximum(xymax1, xymax2)
    xymin_min = torch.minimum(xymin1, xymin2)
    whb = xymax_max - xymin_min
    areab = torch.prod(whb, dim=-1)
    if opr_type == OPR_TYPE.GIOU:
        return iou - (areab - areau) / areab
    diagb = torch.sum(whb ** 2, dim=-1)
    diagc = torch.sum((xywhs1[..., :2] - xywhs2[..., :2]) ** 2, dim=-1)
    diou = iou - diagc / diagb
    if opr_type == OPR_TYPE.DIOU:
        return diou
    r1 = torch.arctan(xywhs1[..., 2] / xywhs1[..., 3])
    r2 = torch.arctan(xywhs2[..., 2] / xywhs2[..., 3])
    v = ((r1 - r2) * 2 / math.pi) ** 2
    alpha = v * 2 / (1 - areai / areau + v + 1e-7)
    if opr_type == OPR_TYPE.CIOU:
        return diou - v * alpha
    raise Exception('err iou type ' + str(opr_type))


def ropr_arr_xyxysT(xyxys1, xyxys2, opr_type=OPR_TYPE.IOU):
    xymax_min = torch.minimum(xyxys1[..., 2:4], xyxys2[..., 2:4])
    xymin_max = torch.maximum(xyxys1[..., :2], xyxys2[..., :2])
    whi = torch.clamp(xymax_min - xymin_max, min=0)
    areai = torch.prod(whi, dim=-1)
    if opr_type == OPR_TYPE.IAREA:
        return areai
    area1 = torch.prod(xyxys1[..., 2:4] - xyxys1[..., :2], dim=-1)
    area2 = torch.prod(xyxys2[..., 2:4] - xyxys2[..., :2], dim=-1)
    areau = area1 + area2 - areai
    iou = areai / areau
    if opr_type is None or opr_type == OPR_TYPE.IOU:
        return iou
    xymax_max = torch.maximum(xyxys1[..., 2:4], xyxys2[..., 2:4])
    xymin_min = torch.minimum(xyxys1[..., :2], xyxys2[..., :2])
    whb = xymax_max - xymin_min
    areab = torch.prod(whb, dim=-1)
    if opr_type == OPR_TYPE.GIOU:
        return iou - (areab - areau) / areab
    xyc1 = (xyxys1[..., 2:4] + xyxys1[..., :2]) / 2
    xyc2 = (xyxys2[..., 2:4] + xyxys2[..., :2]) / 2
    diagb = torch.sum(whb ** 2, dim=-1)
    diagc = torch.sum((xyc1 - xyc2) ** 2, dim=-1)
    diou = iou - diagc / diagb
    if opr_type == OPR_TYPE.DIOU:
        return diou
    wh1 = xyxys1[..., 2:4] - xyxys1[..., :2]
    wh2 = xyxys2[..., 2:4] - xyxys2[..., :2]
    r1 = torch.arctan(wh1[..., 0] / wh1[..., 1])
    r2 = torch.arctan(wh2[..., 0] / wh2[..., 0])
    v = ((r1 - r2) * 2 / math.pi) ** 2
    alpha = v * 2 / (1 - areai / areau + v + 1e-7)
    if opr_type == OPR_TYPE.CIOU:
        return diou - v * alpha
    raise Exception('err iou type ' + str(opr_type))


def _iarea_arr_xlylsT_shapely(xlyls1, xlyls2):
    assert xlyls1.size() == xlyls2.size(), 'shape err'
    num = xlyls1.size(0)
    if len(xlyls1.size()) == 3:
        iareas = torch.zeros(size=(num,), device=xlyls1.device)
        # time1 = time.time()
        fltrs_int, abcls1, abcls2, msks_1pin2, msks_2pin1 = xlylsT_intersect_coresT(xlyls1, xlyls2, eps=1e-7)
        # time2 = time.time()
        for m, (xlyl1, xlyl2, fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1) \
                in enumerate(zip(xlyls1, xlyls2, fltrs_int, abcls1, abcls2, msks_1pin2, msks_2pin1)):
            xlyl_int = intersect_coreT2xlylT(xlyl1, xlyl2, fltr_int, abcl1, abcl2, msk_1pin2, msk_2pin1)
            area = xlylT2areaT(xlyl_int)
            iareas[m] = area
        # time3 = time.time()
        # print(time2 - time1, time3 - time2)
        return iareas
    else:
        iareas = [torch.zeros(size=([0] + list(xlyls1.shape[1:-2])), device=xlyls1.device)]
        for i in range(num):
            iareas.append(_iarea_arr_xlylsT_shapely(xlyls1[i], xlyls2[i])[None, :])
        iareas = torch.cat(iareas, dim=0)
        return iareas


def ropr_arr_xywhasT(xywhas1, xywhas2, opr_type=OPR_TYPE.KL):
    if opr_type == OPR_TYPE.IAREA or opr_type == OPR_TYPE.IOU:
        xlyls1 = xywhasT2xlylsT(xywhas1)
        xlyls2 = xywhasT2xlylsT(xywhas2)
        iarea = _iarea_arr_xlylsT_shapely(xlyls1, xlyls2)
        if opr_type == OPR_TYPE.IAREA:
            return iarea
        area1 = xywhas1[..., 2] * xywhas1[..., 3]
        area2 = xywhas2[..., 2] * xywhas2[..., 3]
        uarea = area1 + area2 - iarea
        iou = iarea / (uarea + 1e-16)
        return iou
    w1, h1, a1, = xywhas1[..., 2] + 1e-7, xywhas1[..., 3] + 1e-7, xywhas1[..., 4]
    w2, h2, a2, = xywhas2[..., 2] + 1e-7, xywhas2[..., 3] + 1e-7, xywhas2[..., 4]
    x_dt, y_dt = xywhas2[..., 0] - xywhas1[..., 0], xywhas2[..., 1] - xywhas1[..., 1]
    wr, hr, a_dt = w2 / w1, h2 / h1, a2 - a1
    wh21, wh12, hw12, hw21 = w2 / h1, w1 / h2, h1 / w2, h2 / w1
    cos1, sin1 = torch.cos(a1), torch.sin(a1)
    cos2, sin2 = torch.cos(a2), torch.sin(a2)
    cos_dt, sin_dt = torch.cos(a_dt), torch.sin(a_dt)
    p1 = ((x_dt * cos1 + y_dt * sin1) / w1) ** 2 + ((y_dt * cos1 - x_dt * sin1) / h1) ** 2 \
         + ((x_dt * cos2 + y_dt * sin2) / w2) ** 2 + ((y_dt * cos2 - x_dt * sin2) / h2) ** 2
    p2 = (wr ** 2 + 1 / wr ** 2 + hr ** 2 + 1 / hr ** 2) * cos_dt ** 2 \
         + (wh21 ** 2 + wh12 ** 2 + hw12 ** 2 + hw21 ** 2) * sin_dt ** 2
    kl_dist = p1 + p2 / 4 - 1
    if opr_type == OPR_TYPE.KL:
        return kl_dist
    kliou = 1 / (kl_dist + 1)
    if opr_type == OPR_TYPE.KLIOU:
        return kliou
    raise Exception('err iou type ' + str(opr_type))


def ropr_arr_xywhuvsT(xywhuvs1, xywhuvs2, opr_type=OPR_TYPE.KL):
    w1, h1, cos1, sin1, = xywhuvs1[..., 2], xywhuvs1[..., 3], xywhuvs1[..., 4], xywhuvs1[..., 5]
    w2, h2, cos2, sin2, = xywhuvs2[..., 2], xywhuvs2[..., 3], xywhuvs2[..., 4], xywhuvs2[..., 5]
    x_dt, y_dt = xywhuvs2[..., 0] - xywhuvs1[..., 0], xywhuvs2[..., 1] - xywhuvs1[..., 1]
    wr, hr = w2 / w1, h2 / h1
    wh21, wh12, hw12, hw21 = w2 / h1, w1 / h2, h1 / w2, h2 / w1
    cos_dt, sin_dt = cos1 * cos2 + sin1 * sin2, sin2 * cos1 - sin1 * cos2
    p1 = ((x_dt * cos1 + y_dt * sin1) / w1) ** 2 + ((y_dt * cos1 - x_dt * sin1) / h1) ** 2 \
         + ((x_dt * cos2 + y_dt * sin2) / w2) ** 2 + ((y_dt * cos2 - x_dt * sin2) / h2) ** 2
    p2 = (wr ** 2 + 1 / wr ** 2 + hr ** 2 + 1 / hr ** 2) * cos_dt ** 2 \
         + (wh21 ** 2 + wh12 ** 2 + hw12 ** 2 + hw21 ** 2) * sin_dt ** 2
    kl_dist = p1 + p2 / 4 - 1
    if opr_type == OPR_TYPE.KL:
        return kl_dist
    kliou = 1 / (kl_dist + 1)
    if opr_type == OPR_TYPE.KLIOU:
        return kliou
    raise Exception('err iou type ' + str(opr_type))


def ropr_arr_xlylsT(xlyls1, xlyls2, opr_type=OPR_TYPE.IOU):
    # torch实现较慢
    # iareas = _iarea_arr_xlylsT_shapely(xlyls1, xlyls2)
    # if opr_type == OPR_TYPE.IAREA:
    #     return iareas
    # areas1 = xlylsT2areasT(xlyls1)
    # areas2 = xlylsT2areasT(xlyls2)
    # uareas = areas1 + areas2 - iareas
    # ious = iareas / (uareas + 1e-16)
    # if opr_type == OPR_TYPE.IOU:
    #     return ious
    # raise Exception('err iou type ' + str(opr_type))

    device = xlyls1.device
    xlyls1 = xlyls1.detach().cpu().numpy()
    xlyls2 = xlyls2.detach().cpu().numpy()
    res = ropr_arr_xlylsN(xlyls1, xlyls2, opr_type)
    res = torch.from_numpy(res).to(device)
    return res


def ropr_mat_xyxysT(xyxys1, xyxys2, opr_type=OPR_TYPE.IOU):
    xyxys1, xyxys2 = _align_shapeT(xyxys1, xyxys2, last_axis=1)
    return ropr_arr_xyxysT(xyxys1, xyxys2, opr_type)


def ropr_mat_xywhsT(xywhs1, xywhs2, opr_type=OPR_TYPE.IOU):
    xywhs1, xywhs2 = _align_shapeT(xywhs1, xywhs2, last_axis=1)
    return ropr_arr_xywhsT(xywhs1, xywhs2, opr_type)


def ropr_mat_xywhasT(xywhas1, xywhas2, opr_type=OPR_TYPE.IOU):
    xywhas1, xywhas2 = _align_shapeT(xywhas1, xywhas2, last_axis=1)
    return ropr_arr_xywhasT(xywhas1, xywhas2, opr_type)


def ropr_mat_xlylsT(xlyls1, xlyls2, opr_type=OPR_TYPE.IOU):
    xlyls1, xlyls2 = _align_shapeT(xlyls1, xlyls2, last_axis=2)
    return ropr_arr_xlylsT(xlyls1, xlyls2, opr_type)


# </editor-fold>


# <editor-fold desc='nms原型'>

class NMS_TYPE:
    HARD = 'hard'
    SOFT = 'soft'
    NONE = 'none'


def _nms_softN(bordersN, confsN, roprN, iou_thres=0.45, iou_type=IOU_TYPE.IOU):
    confsN = copy.deepcopy(confsN)  # 隔离
    conf_thres = np.min(confsN)  # 阈值
    prsv_inds = []
    for i in range(len(bordersN)):
        ind = np.argmax(confsN)
        if confsN[ind] <= conf_thres:
            break
        prsv_inds.append(ind)
        res_inds = np.nonzero(confsN > conf_thres)[0]
        bordersN1 = np.repeat(bordersN[ind:ind + 1], repeats=len(res_inds), axis=0)
        ious = roprN(bordersN1, bordersN[res_inds], opr_type=iou_type)
        confsN[res_inds[ious > iou_thres]] *= (1 - ious)
    prsv_inds = np.array(prsv_inds)
    return prsv_inds


def _nms_hardN(bordersN, confsN, roprN, iou_thres=0.45, iou_type=IOU_TYPE.IOU):
    confsN = copy.deepcopy(confsN)  # 隔离
    order = np.argsort(-confsN)
    bordersN, confsN = bordersN[order], confsN[order]
    prsv_inds = []
    for i in range(bordersN.shape[0]):
        if confsN[i] == 0:
            continue
        prsv_inds.append(order[i])
        res_inds = i + 1 + np.nonzero(confsN[i + 1:] > 0)[0]
        bordersN1 = np.repeat(bordersN[i:i + 1], repeats=len(res_inds), axis=0)
        ious = roprN(bordersN1, bordersN[res_inds], opr_type=iou_type)
        confsN[res_inds[ious > iou_thres]] = 0
    prsv_inds = np.array(prsv_inds)
    return prsv_inds


def _nmsN(bordersN, confsN, roprN, cindsN=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU):
    if cindsN is None:
        if nms_type == NMS_TYPE.SOFT:
            return _nms_softN(bordersN, confsN, roprN, iou_thres=iou_thres, iou_type=iou_type)
        elif nms_type == NMS_TYPE.HARD:
            return _nms_hardN(bordersN, confsN, roprN, iou_thres=iou_thres, iou_type=iou_type)
        elif nms_type == NMS_TYPE.NONE or nms_type is None:
            return np.arange(bordersN.shape[0])
        else:
            raise Exception('nms type err')
    else:
        prsv_inds = []
        num_cls = int(np.max(cindsN))
        for i in range(num_cls + 1):
            inds = cindsN == i
            if np.any(inds):
                prsv_inds_cls = _nmsN(bordersN[inds], confsN[inds], roprN, cindsN=None, iou_thres=iou_thres,
                                      nms_type=nms_type, iou_type=iou_type)
                inds = np.nonzero(inds)[0]
                prsv_inds.append(inds[prsv_inds_cls])
        prsv_inds = np.concatenate(prsv_inds, axis=0)
    return prsv_inds


def nms_xyxysN(xyxysN, confsN, cindsN=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU):
    return _nmsN(xyxysN, confsN, ropr_arr_xyxysN, cindsN, iou_thres, nms_type, iou_type)


def nms_xywhsN(xywhsN, confsN, cindsN=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU):
    return _nmsN(xywhsN, confsN, ropr_arr_xywhsN, cindsN, iou_thres, nms_type, iou_type)


def nms_xywhasN(xywhasN, confsN, cindsN=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU):
    return _nmsN(xywhasN, confsN, ropr_arr_xywhasN, cindsN, iou_thres, nms_type, iou_type)


def nms_xlyls(xlyls, confsN, cindsN=None, iou_thres=0.45, iou_type=IOU_TYPE.IOU):
    order = np.argsort(-confsN)
    xlyls = [xlyls[ind] for ind in order]
    confsN = confsN[order]
    prsv_inds = []
    for i in range(len(xlyls)):
        if confsN[i] == 0:
            continue
        prsv_inds.append(order[i])
        res_inds = i + 1 + np.nonzero(confsN[i + 1:] > 0)[0]
        xlyls1 = xlyls[i:i + 1] * len(res_inds)
        xlyls2 = [xlyls[ind] for ind in res_inds]
        ious = ropr_arr_xlyls(xlyls1, xlyls2, opr_type=iou_type)
        confsN[res_inds[ious > iou_thres]] = 0
    prsv_inds = np.array(prsv_inds)
    return prsv_inds
    # boxes2


# SOFT NMS
def _nms_softT(bordersT, confsT, roprT, iou_thres=0.45, iou_type=IOU_TYPE.IOU):
    confsT = copy.deepcopy(confsT)  # 隔离
    conf_thres = torch.min(confsT)  # 阈值
    prsv_inds = []
    for i in range(bordersT.size(0)):
        ind = torch.argmax(confsT)
        if confsT[ind] < conf_thres:
            break
        prsv_inds.append(ind)
        res_inds = torch.nonzero(confsT > conf_thres, as_tuple=True)[0]
        bordersT1 = bordersT[ind:ind + 1].repeat(len(res_inds), 1)
        ious = roprT(bordersT1, bordersT[res_inds], opr_type=iou_type)
        confsT[res_inds[ious > iou_thres]] *= (1 - ious)
    prsv_inds = torch.Tensor(prsv_inds).long()
    return prsv_inds


# HARD NMS
def _nms_hardT(bordersT, confsT, roprT, iou_thres=0.45, iou_type=IOU_TYPE.IOU):
    confsT = copy.deepcopy(confsT)  # 隔离
    order = torch.argsort(confsT, descending=True)
    bordersT, confsT = bordersT[order], confsT[order]
    prsv_inds = []
    for i in range(bordersT.size(0)):
        if confsT[i] == 0:
            continue
        prsv_inds.append(order[i])
        res_inds = i + 1 + torch.nonzero(confsT[i + 1:] > 0, as_tuple=True)[0]
        boxesT1 = bordersT[i:i + 1].repeat(len(res_inds), 1)
        ious = roprT(boxesT1, bordersT[res_inds], opr_type=iou_type)
        confsT[res_inds[ious > iou_thres]] = 0

    prsv_inds = torch.Tensor(prsv_inds).long()
    return prsv_inds


# NMS
def _nmsT(bordersT, confsT, roprT, cindsT=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU):
    if cindsT is None:
        if nms_type == NMS_TYPE.SOFT:
            return _nms_softT(bordersT, confsT, roprT, iou_thres=iou_thres, iou_type=iou_type)
        elif nms_type == NMS_TYPE.HARD:
            if iou_type == IOU_TYPE.IOU and roprT == ropr_arr_xyxysT:
                return torchvision.ops.nms(bordersT, confsT, iou_threshold=iou_thres)
            else:
                return _nms_hardT(bordersT, confsT, roprT, iou_thres=iou_thres, iou_type=iou_type)
        elif nms_type == NMS_TYPE.NONE or nms_type is None:
            return torch.arange(bordersT.size(0))
        else:
            raise Exception('nms type err')
    else:
        prsv_inds = []
        num_cls = int(torch.max(cindsT).item())
        for i in range(num_cls + 1):
            inds = cindsT == i
            if torch.any(inds):
                prsv_inds_cls = _nmsT(bordersT[inds], confsT[inds], roprT, cindsT=None, iou_thres=iou_thres,
                                      nms_type=nms_type, iou_type=iou_type)
                inds = torch.nonzero(inds, as_tuple=False).squeeze(dim=1)
                prsv_inds.append(inds[prsv_inds_cls])
        prsv_inds = torch.cat(prsv_inds, dim=0)
    return prsv_inds


def nms_xyxysT(xyxysT, confsT, cindsT=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU):
    return _nmsT(xyxysT, confsT, ropr_arr_xyxysT, cindsT, iou_thres, nms_type, iou_type)


def nms_xywhsT(xywhsT, confsT, cindsT=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU):
    xyxysT = xywhsT2xyxysT(xywhsT)
    return _nmsT(xyxysT, confsT, ropr_arr_xyxysT, cindsT, iou_thres, nms_type, iou_type)


def nms_xywhasT(xywhasT, confsT, cindsT=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU):
    return _nmsT(xywhasT, confsT, ropr_arr_xywhasT, cindsT, iou_thres, nms_type, iou_type)

# </editor-fold>
