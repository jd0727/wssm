import os
from collections import OrderedDict

os.environ["DISPLAY"] = "localhost:10.0"
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.interactive(True)
from utils.label import *
from PIL import ImageFont

FONT_DICT_SMALL = dict(fontfamily='DejaVu Sans', fontsize='small', weight='bold')
FONT_DICT_XLARGE = dict(fontfamily='DejaVu Sans', fontsize='x-large', weight='bold')

FONT_PTH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'times.ttf')
IMAGE_SIZE_DEFAULT = (256, 256)
CMAP = plt.get_cmap('jet')

AXPLT_MAPPER = OrderedDict()


def axplt_registor(*item_types):
    def wrapper(_axplt):
        for item_type in item_types:
            AXPLT_MAPPER[item_type] = _axplt
        return _axplt

    return wrapper


def _axplt_item(item, axis, **kwargs):
    pltor = AXPLT_MAPPER[item.__class__]
    pltor(item, axis=axis, **kwargs)
    return axis


# <editor-fold desc='坐标轴设置'>
def get_axis(axis=None, tick=True, **kwargs):
    if axis is None:
        fig, axis = plt.subplots()
        fig.show()
    if not tick:
        axis.set_xticks([])
        axis.set_yticks([])
    axis.xaxis.set_ticks_position('top')
    axis.set_aspect('equal', 'box')
    return axis


def init_axis(axis, img_size, **kwargs):
    axis.set_xlim(0, img_size[0])
    axis.set_ylim(0, img_size[1])
    if not axis.yaxis_inverted():
        axis.invert_yaxis()
    return axis


# </editor-fold>


# <editor-fold desc='颜色处理'>
def random_color(index, low=30, high=200, unit=False):
    radius = (high - low) / 2
    color = np.cos([index * 7, index * 8 + math.pi, index * 9 - math.pi]) * radius + radius + low
    color = (color / 255) if unit else tuple(color.astype(np.int32))
    return color


def mcol2pcol(color):
    color = matplotlib.colors.to_rgb(color)
    return tuple((np.array(color) * 255).astype(np.int32))


def pcol2mcol(color):
    return np.array(color) / 255


# </editor-fold>


# <editor-fold desc='标签展示'>
def cate_cont2str(cate_cont):
    cate = IndexCategory.convert(cate_cont.category)
    name = cate_cont['name'] if 'name' in cate_cont.keys() else '<%d>' % cate.cindN
    cate_str = name + ' %.2f' % cate.conf if cate.conf < 1 else name
    return cate_str


@axplt_registor(Image.Image, np.ndarray, torch.Tensor)
def _axplt_img(img, axis, alpha=0.3, **kwargs):
    imgN = img2imgN(img)
    extent = (0, imgN.shape[1], imgN.shape[0], 0)
    if len(imgN.shape) == 2:
        axis.imshow(imgN, extent=extent)
    elif imgN.shape[2] == 4:
        imgN, maskN = imgN[:, :, :3], imgN[:, :, 3]
        axis.imshow(imgN, extent=extent)
        axis.imshow(maskN, cmap=CMAP, alpha=alpha, extent=extent)
    else:
        axis.imshow(imgN, extent=extent)
    return axis


@axplt_registor(XYXYBorder, XYWHABorder, XYWHBorder, XLYLBorder)
def _axplt_border(border, axis, color='r', linewidth=1.0, **kwargs):
    init_axis(axis, img_size=border.size, **kwargs)
    border = XLYLBorder.convert(border)
    xlyl = np.concatenate([border.xlylN, border.xlylN[:1]], axis=0)
    axis.plot(xlyl[:, 0], xlyl[:, 1], '-', linewidth=linewidth, color=color)
    return axis


@axplt_registor(CategoryLabel)
def _axplt_cate(cate, axis, color=None, **kwargs):
    axis = init_axis(axis, img_size=cate.img_size, **kwargs)
    color = random_color(IndexCategory.convert(cate.category).cindN, unit=True) if color is None else color
    axis.set_title(label=cate_cont2str(cate), pad=4, color=color, fontdict=FONT_DICT_XLARGE)
    _axplt_border(border=cate.ctx_border, axis=axis, linewidth=2, color='k')
    return axis


@axplt_registor(PointItem)
def _axplt_pnt(pnt, axis, color=None, **kwargs):
    init_axis(axis, img_size=pnt.size, **kwargs)
    color = random_color(IndexCategory.convert(pnt.category).cindN, unit=True) if color is None else color
    axis.plot(pnt.xyN[0], pnt.xyN[1], 'o', ms=2, color=color)
    axis.text(pnt.xyN[0], pnt.xyN[1], cate_cont2str(pnt), color=color, fontdict=FONT_DICT_SMALL)
    return axis


def _axplt_border_text(border, axis, text, color, text_color, linewidth=1, alpha=0.7, with_text=True, **kwargs):
    init_axis(axis, img_size=border.size, **kwargs)
    border = XLYLBorder.convert(border)
    xlyl = np.concatenate([border.xlylN, border.xlylN[:1]], axis=0)
    axis.plot(xlyl[:, 0], xlyl[:, 1], '-', linewidth=linewidth, color=color)
    if with_text:
        idx = np.argmin(xlyl[:, 1] + xlyl[:, 0] * 0.2)
        x = np.clip(xlyl[idx, 0], a_min=min(axis.get_xlim()), a_max=max(axis.get_xlim()))
        y = np.clip(xlyl[idx, 1], a_min=min(axis.get_ylim()), a_max=max(axis.get_ylim()))
        axis.text(x, y, text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@axplt_registor(BoxItem)
def _axplt_box(box, axis, color=None, text_color='w', **kwargs):
    color = random_color(IndexCategory.convert(box.category).cindN, unit=True) if color is None else color
    _axplt_border_text(border=box.border, axis=axis, text=cate_cont2str(box), color=color,
                       text_color=text_color, linewidth=1, alpha=0.7, **kwargs)
    return axis


@axplt_registor(BoxRefItem)
def axplt_box_ref(box, axis, color=None, text_color='w', **kwargs):
    color = random_color(IndexCategory.convert(box.category).cindN, unit=True) if color is None else color
    linestyle = '-' if 'long' in box.keys() and box['long'] else '--'
    _axplt_border(border=box.border_ref, axis=axis, color='k', linestyle=linestyle, linewidth=1, **kwargs)
    _axplt_border_text(border=box.border, axis=axis, text=cate_cont2str(box), color=color,
                       text_color=text_color, linewidth=1, alpha=0.7, **kwargs)
    return axis


@axplt_registor(AbsBoolRegion)
def _axplt_region_abs(rgn, axis, color=None, alpha=0.7, **kwargs):
    init_axis(axis, img_size=rgn.size, **kwargs)
    color = random_color(0, unit=True) if color is None else color
    maskN = rgn.maskNb[..., None].astype(np.float32)
    color = maskN * np.broadcast_to(color, (maskN.shape[0], maskN.shape[1], 3))
    maskN_merge = np.concatenate([color, maskN], axis=2)
    axis.imshow(maskN_merge, alpha=alpha, extent=(0, maskN_merge.shape[1], maskN_merge.shape[0], 0))
    return axis


@axplt_registor(RefValRegion)
def _axplt_region_ref(rgn, axis, color=None, alpha=0.7, **kwargs):
    init_axis(axis, img_size=rgn.size, **kwargs)
    color = random_color(0, unit=True) if color is None else color
    rgn = RefValRegion.convert(rgn)
    xyxy_rgn = rgn.xyxyN
    maskN_ref = rgn.maskNb_ref[..., None].astype(np.float32)
    color = np.broadcast_to(color, (maskN_ref.shape[0], maskN_ref.shape[1], 3))
    maskN_merge = np.concatenate([color, maskN_ref], axis=2)
    axis.imshow(maskN_merge, alpha=alpha, extent=(xyxy_rgn[0], xyxy_rgn[2], xyxy_rgn[3], xyxy_rgn[1]))
    return axis


@axplt_registor(SegItem)
def _axplt_seg(seg, axis, color=None, text_color='w', alpha=0.7, **kwargs):
    color = random_color(IndexCategory.convert(seg.category).cindN, unit=True) if color is None else color

    rgn = AbsBoolRegion.convert(seg.rgn)
    _axplt_region_abs(rgn, axis, color=color, alpha=alpha, **kwargs)

    posi = np.where(rgn.maskNb)
    axis.text(posi[1][0], posi[0][0], cate_cont2str(seg), color=text_color, fontdict=FONT_DICT_SMALL,
              bbox=dict(facecolor=color, alpha=0.7, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@axplt_registor(InstItem)
def _axplt_inst(inst, axis, color=None, text_color='w', alpha=0.7, **kwargs):
    color = random_color(IndexCategory.convert(inst.category).cindN, unit=True) if color is None else color
    _axplt_border_text(border=inst.border, text=cate_cont2str(inst), axis=axis, color=color, text_color=text_color,
                       linewidth=1, alpha=alpha, **kwargs)
    _axplt_region_ref(rgn=inst.rgn, axis=axis, color=color, alpha=alpha, **kwargs)
    return axis


@axplt_registor(InstRefItem)
def _axplt_inst_ref(inst, axis, color=None, text_color='w', alpha=0.7, **kwargs):
    color = random_color(IndexCategory.convert(inst.category).cindN, unit=True) if color is None else color
    linestyle = '-' if 'long' in inst.keys() and inst['long'] else '--'
    _axplt_border(border=inst.border_ref, axis=axis, color='k', linestyle=linestyle, linewidth=1.5, **kwargs)
    _axplt_border_text(border=inst.border, text=cate_cont2str(inst), axis=axis, color=color, text_color=text_color,
                       linewidth=1, alpha=alpha, **kwargs)
    _axplt_region_ref(rgn=inst.rgn, axis=axis, color=color, alpha=alpha, **kwargs)
    return axis


@axplt_registor(list)
def _axplt_list(items, axis, **kwargs):
    for item in items:
        _axplt_item(item=item, axis=axis, **kwargs)
    return axis


@axplt_registor(PointsLabel, BoxesLabel, SegsLabel, InstsLabel)
def _axplt_items(items, axis, **kwargs):
    _axplt_border(items.ctx_border, axis, color='k', linewidth=2, **kwargs)
    for item in items:
        _axplt_item(item=item, axis=axis, **kwargs)
    return axis


def _axplt_label(*items, axis=None, tick=True, **kwargs):
    axis = get_axis(axis=axis, tick=tick)
    for item in items:
        if item is not None:
            _axplt_item(item=item, axis=axis, **kwargs)
    return axis


# </editor-fold>
PILRND_MAPPER = OrderedDict()


def pilrnd_registor(*item_types):
    def wrapper(_pilrnd):
        for item_type in item_types:
            PILRND_MAPPER[item_type] = _pilrnd
        return _pilrnd

    return wrapper


@pilrnd_registor(PointsLabel, BoxesLabel, SegsLabel, InstsLabel)
def _pilrnd_items(items, img, with_ctx=True, **kwargs):
    if with_ctx:
        img = _pilrnd_border(items.ctx_border, img=img, color=(0, 0, 0), **kwargs)
    for item in items:
        img = _pilrnd_item(item=item, img=img, **kwargs)
    return img


def _pilrnd_item(item, img, **kwargs):
    rndor = PILRND_MAPPER[item.__class__]
    img = rndor(item, img=img, **kwargs)
    return img


def _pilrnd_label(*items, **kwargs):
    img = None
    for item in items:
        if item is not None:
            img = _pilrnd_item(item=item, img=img, **kwargs)
    return img


# <editor-fold desc='基础元素绘制'>
@pilrnd_registor(Image.Image, np.ndarray, torch.Tensor)
def _pilrnd_img(img_rnd, img=None, alpha=0.3, **kwargs):
    if isinstance(img_rnd, Image.Image) and (img_rnd.mode == 'RGB' or img_rnd.mode == 'L'):
        return img_rnd
    imgN = img2imgN(img_rnd)
    imgN, maskN = imgN[:, :, :3], imgN[:, :, 3]
    mapper = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=np.min(maskN), vmax=np.max(maskN)))
    maskN_mapped = mapper.to_rgba(maskN)[:, :, :3] * 255
    imgN_mixed = imgN * (1 - alpha) + maskN_mapped * alpha
    return imgN2imgP(imgN_mixed)


@pilrnd_registor(XYXYBorder, XYWHBorder, XYWHABorder, XLYLBorder)
def _pilrnd_border(border, img=None, color=None, line_width=2, **kwargs):
    img = Image.new(mode='RGB', size=border.size) if img is None else img
    color = random_color(0, unit=False) if color is None else color
    draw = PIL.ImageDraw.Draw(img)
    xlyl = XLYLBorder.convert(border).xlylN
    xlyl = np.concatenate([xlyl, xlyl[:1]], axis=0)
    draw.line(list(xlyl.reshape(-1)), fill=color, width=line_width)
    return img


@pilrnd_registor(PointItem)
def _pilrnd_pnt(pnt, img=None, color=None, radius=3, font_size=20, **kwargs):
    img = Image.new(mode='RGB', size=pnt.size) if img is None else img
    color = random_color(IndexCategory.convert(pnt.category).cindN, unit=False) if color is None else color
    draw = PIL.ImageDraw.Draw(img)
    box_str = cate_cont2str(pnt)
    draw.ellipse([tuple(pnt.xyN - radius), tuple(pnt.xyN + radius)], fill=color)
    font = ImageFont.truetype(FONT_PTH, size=font_size)
    textsize = draw.textbbox(xy=(0, 0), text=box_str, font=font)[2:]
    draw.text((pnt.xyN[0], pnt.xyN[1] - textsize[1]), box_str, fill=color, font=font)
    return img


def _pilrnd_text(text, img, x, y, color=None, text_color=(255, 255, 255), font_size=20):
    color = random_color(0, unit=False) if color is None else color
    draw = PIL.ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PTH, size=font_size)
    textsize = draw.textbbox(xy=(0, 0), text=text, font=font)[2:]
    draw.rectangle((x, y - textsize[1], x + textsize[0], y), fill=color)
    draw.text((x, y - textsize[1]), text, fill=text_color, font=font)
    return img


@pilrnd_registor(AbsBoolRegion, AbsValRegion)
def _pilrnd_region(rgn, img, color=None, alpha=0.7, rgn_bdr_width=0, rgn_bdr_color=(255, 255, 255), **kwargs):
    color = random_color(0, unit=False) if color is None else color
    maskN = rgn.maskNb.astype(np.float32) * 255

    mask = imgN2imgP(maskN * alpha)
    mask_color = Image.new(mode='RGB', size=mask.size, color=color)
    img = Image.composite(mask_color, img, mask)

    if rgn_bdr_width > 0:
        kernel = np.ones(shape=(rgn_bdr_width * 2 + 1, rgn_bdr_width * 2 + 1))
        maskN_expd = cv2.dilate(maskN, kernel)

        mask_bdr = imgN2imgP((maskN_expd - maskN) * alpha)
        mask_bdr_color = Image.new(mode='RGB', size=mask.size, color=rgn_bdr_color)
        img = Image.composite(mask_bdr_color, img, mask_bdr)
    return img


@pilrnd_registor(RefValRegion)
def _pilrnd_region_ref(rgn, img, color=None, alpha=0.7, **kwargs):
    color = random_color(0, unit=False) if color is None else color
    rgn = RefValRegion.convert(rgn)
    maskP_ref = imgN2imgP(imgP2imgN(rgn.maskNb_ref.convert('L')) * alpha)
    maskP_color = Image.new(mode='RGB', size=maskP_ref.size, color=color)
    maskP_merge = Image.merge(mode='RGBA', bands=[*maskP_color.split(), maskP_ref])
    img.paste(maskP_merge, (int(rgn.xyxyN[0]), int(rgn.xyxyN[1])))
    return img


@pilrnd_registor(BoxItem)
def _pilrnd_box(box, img=None, color=None, text_color=(255, 255, 255), font_size=20, with_text=True, line_width=2,
                with_bdr=True, **kwargs):
    img = Image.new(mode='RGB', size=box.size) if img is None else img
    color = random_color(IndexCategory.convert(box.category).cindN, unit=False) if color is None else color
    draw = PIL.ImageDraw.Draw(img)
    xlyl = XLYLBorder.convert(box.border).xlylN
    if with_bdr:
        xlyl = np.concatenate([xlyl, xlyl[:1]], axis=0)
        draw.line(list(xlyl.reshape(-1)), fill=color, width=line_width)

    if with_text:
        box_str = cate_cont2str(box)
        idx = np.argmin(xlyl[:, 1] + xlyl[:, 0] * 0.2)
        x = np.clip(xlyl[idx, 0], a_min=0, a_max=img.size[0])
        y = np.clip(xlyl[idx, 1], a_min=0, a_max=img.size[1])
        img = _pilrnd_text(box_str, img, x, y, color=color, text_color=text_color, font_size=font_size)
    return img


@pilrnd_registor(SegItem)
def _pilrnd_seg(seg, img=None, color=None, alpha=0.3, text_color=(255, 255, 255), font_size=20, **kwargs):
    img = Image.new(mode='RGB', size=seg.size) if img is None else img
    color = random_color(IndexCategory.convert(seg.category).cindN, unit=False) if color is None else color
    img = _pilrnd_region(seg.rgn, img, color=color, alpha=alpha)
    posi = np.where(np.array(seg.rgn.maskP))
    img = _pilrnd_text(cate_cont2str(seg), img, posi[0][0], posi[1][0], color=color, text_color=text_color,
                       font_size=font_size)
    return img


@pilrnd_registor(InstItem, InstRefItem)
def _pilrnd_inst(inst, img=None, color=None, alpha=0.3, text_color=(255, 255, 255), font_size=20, with_text=True,
                 with_bdr=True, line_width=2, **kwargs):
    img = Image.new(mode='RGB', size=inst.size) if img is None else img
    color = random_color(IndexCategory.convert(inst.category).cindN, unit=False) if color is None else color
    draw = PIL.ImageDraw.Draw(img)
    xlyl = XLYLBorder.convert(inst.border).xlylN
    if with_bdr:
        xlyl = np.concatenate([xlyl, xlyl[:1]], axis=0)
        draw.line(list(xlyl.reshape(-1)), fill=color, width=line_width)
    if with_text:
        idx = np.argmin(xlyl[:, 1] + xlyl[:, 0] * 0.2)
        x = np.clip(xlyl[idx, 0], a_min=0, a_max=img.size[0])
        y = np.clip(xlyl[idx, 1], a_min=0, a_max=img.size[1])
        img = _pilrnd_text(cate_cont2str(inst), img, x, y, color=color, text_color=text_color, font_size=font_size)
    img = _pilrnd_region(inst.rgn, img, color=color, alpha=alpha, **kwargs)
    return img


def show_label(img=None, label=None, use_pil=False, axis=None, tick=True, **kwargs):
    axis = get_axis(axis=axis, tick=tick)
    if use_pil:
        img = _pilrnd_label(img, label, **kwargs)
        axis.imshow(img)
        return axis
    else:
        return _axplt_label(img, label, axis=axis, **kwargs)


# 检查数据分布
def show_distribute(data, quant_step=None, num_quant=20, axis=None):
    std = np.std(data)
    mean = np.average(data)
    if quant_step == None:
        quant_step = std / 5
    # 检查是否有值
    if std == 0:
        vals_axis = [mean - 1, mean, mean + 1]
        nums = [0, len(data), 0]
        quant_step = 0.1
    else:
        vals_axis = (np.arange(-num_quant, num_quant) + 0.5) * quant_step + mean
        nums = np.zeros(shape=num_quant * 2)
        for i in range(len(data)):
            ind = np.floor((data[i] - mean) / quant_step) + num_quant
            if ind < num_quant * 2:
                nums[int(ind)] += 1
        # 归一化
        nums = nums / np.sum(nums) / quant_step
    if axis is None:
        fig, axis = plt.subplots()
    axis.bar(vals_axis, nums, width=quant_step * 0.8, color='k')
    return axis


# 显示图片
def show_img(img, axis=None, **kwargs):
    img = img2imgN(img)
    if axis is None:
        fig, axis = plt.subplots()
    axis.imshow(img, extent=(0, img.shape[1], img.shape[0], 0), **kwargs)
    # axis.axis('off')
    return axis


# 显示曲线
def show_curve(data, axis=None, **kwargs):
    if axis is None:
        fig, axis = plt.subplots()
    axis.plot(data, color='k', **kwargs)
    return axis


# def pilrnd_obj(obj,img,  camera=None, focus=100, light=None, img_size=IMAGE_SIZE_DEFAULT):
#     if obj is not None:
#         img_size = img.size
#         if camera is None:
#             R = axis_angle_to_matrix(torch.Tensor([0, 0, math.pi]))[None, :]
#             T = torch.Tensor([img_size[0] / 2, img_size[1] / 2, focus])[None, :]
#             camera = FoVPerspectiveCameras(
#                 R=R, T=T, aspect_ratio=1, znear=0.01, zfar=100,
#                 fov=np.arctan(img_size[0] / 2 / focus) * 2, degrees=False)
#         raster_settings = RasterizationSettings(image_size=(img_size[1], img_size[0]), blur_radius=0.0)
#         if light is None:
#             light = PointLights(location=[[0.0, 0.0, -focus]])
#         renderer = MeshRenderer(
#             rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
#             shader=HardPhongShader(cameras=camera, lights=light)
#         )
#         mesh = obj.transed_mesh
#         img_mesh = renderer(mesh)
#         img_mesh = torch.squeeze(img_mesh, dim=0).detach().numpy() * 255
#         img_mesh, mask = img_mesh[..., :3], img_mesh[..., 3:]
#         img_mesh = img2imgP(img_mesh)
#         mask = img2imgP(mask)
#         img = Image.composite(img_mesh, img, mask)
#     return img


# </editor-fold>

# <editor-fold desc='子图排列'>

# 按数列展示
def arrange_arr(datas, shower=None, show_inds=True, **kwargs):
    # 确定长宽
    s0 = datas.size()
    area = 10 * 8
    num_wid = math.ceil(np.sqrt(s0))
    num_hei = math.ceil(s0 / num_wid)
    rate = num_wid / num_hei
    hei = round(math.sqrt(area / rate))
    wid = round(area / hei)
    fig = plt.figure(figsize=(wid, hei))
    # 画图
    ind = 0
    for i in range(num_wid):
        for j in range(num_hei):
            if ind == s0:
                break
            axis = fig.add_subplot(num_hei, num_wid, ind + 1)
            if show_inds:
                axis.set_title(str(ind), fontdict=FONT_DICT_XLARGE)
            shower(axis=axis, **datas[ind], **kwargs)
            ind += 1
    fig.subplots_adjust(wspace=0.15, hspace=0.2)
    return fig


# 按矩阵展示
def arrange_mat(datas, shower=None, show_inds=True, **kwargs):
    s0, s1 = datas.size()
    rate = s0 / s1
    area = 10 * 8
    wid = round(math.sqrt(area / rate))
    hei = round(area / wid)
    fig = plt.figure(figsize=(wid, hei))
    if show_inds:
        # 设置刻度
        axis = fig.add_subplot(1, 1, 1)
        # x
        axis.set_xlim(0, s1)
        axis.set_xticks(np.arange(s1) + 0.5)
        axis.set_xticklabels([str(i) for i in range(s1)], fontdict=FONT_DICT_XLARGE)
        axis.xaxis.tick_top()
        # y
        axis.set_ylim(0, s0)
        axis.set_yticks(np.arange(s0) + 0.5)
        axis.set_yticklabels([str(i) for i in range(s0)], fontdict=FONT_DICT_XLARGE)
        axis.invert_yaxis()
        # 边框
        axis.spines['bottom'].set_linewidth(0)
        axis.spines['left'].set_linewidth(0)
        axis.spines['right'].set_linewidth(0)
        axis.spines['top'].set_linewidth(0)
        axis.tick_params(axis=u'both', which=u'both', length=0)
    # 画图
    ind = 1
    for i in range(s0):
        for j in range(s1):
            axis = fig.add_subplot(s0, s1, ind)
            shower(axis=axis, **datas[i, j], **kwargs)
            ind += 1
    # fig.subplots_adjust(wspace=0.3, hspace=0.6)
    return fig


# </editor-fold>

# <editor-fold desc='数据封装'>

class Datas():
    def size(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class Datas1dArrs(Datas):
    def __init__(self, arrs, para_name='img'):
        self.arrs = np.array(arrs)
        self.para_name = para_name

    def size(self):
        return self.arrs.shape[0]

    def __getitem__(self, item):
        return dict(((self.para_name, self.arrs[item]),))


class Datas2dArrs(Datas):
    def __init__(self, arrs, para_name='img'):
        self.arrs = arrs
        self.para_name = para_name

    def size(self):
        return self.arrs.shape[0], self.arrs.shape[1]

    def __getitem__(self, item):
        ind0, ind1 = item
        return dict(((self.para_name, self.arrs[ind0, ind1]),))


class Datas1dImgLab(Datas):
    def __init__(self, imgs, labels=None):
        self.imgs = imgs
        self.labels = labels

    def size(self):
        return len(self.imgs)

    def __getitem__(self, item):
        return dict(img=self.imgs[item], label=self.labels[item] if self.labels is not None else None, tick=False)


class Datas2dImgGtBx(Datas):
    def __init__(self, imgs, gtss, boxes):
        self.imgs = imgs
        self.boxes = boxes
        self.gtss = gtss

    def size(self):
        return len(self.gtss), 2

    def __getitem__(self, item):
        ind0, ind1 = item
        if ind1 == 0:
            return dict(img=self.imgs[ind0], label=self.gtss[ind0], tick=False)
        else:
            return dict(img=self.imgs[ind0], label=self.boxes[ind0], tick=False)


# </editor-fold>

# <editor-fold desc='数据展示接口'>

def show_arrs(arrs):
    if isinstance(arrs, torch.Tensor):
        arrs = arrs.detach().cpu().numpy()
    vmin = np.min(arrs)
    vmax = np.max(arrs)
    if len(arrs.shape) == 2:
        fig, axis = plt.subplots()
        show_img(img=arrs, vmin=vmin, vmax=vmax, axis=axis)
    elif len(arrs.shape) == 3:
        datas = Datas1dArrs(arrs, para_name='img')
        fig = arrange_arr(datas, shower=show_img, vmin=vmin, vmax=vmax)
    elif len(arrs.shape) == 4:
        datas = Datas2dArrs(arrs, para_name='img')
        fig = arrange_mat(datas, shower=show_img, vmin=vmin, vmax=vmax)
    else:
        raise Exception('err shape')
    return fig


def show_labels(imgs, labels=None, **kwargs):
    if imgs is not None and isinstance(imgs, torch.Tensor) and len(imgs.size()) == 3:
        return show_label(imgs, labels)
    else:
        datas = Datas1dImgLab(imgs, labels)
        fig = arrange_arr(datas, shower=show_label, **kwargs)
    # plt.show()
    return fig


def show_labelscmp(imgs, labels, labels_cmp):
    if isinstance(imgs, torch.Tensor) and len(imgs.size()) == 3:
        labels = [labels]
        labels_cmp = [labels_cmp]
        imgs = torch.unsqueeze(imgs, dim=0)
    datas = Datas2dImgGtBx(imgs, labels, labels_cmp)
    fig = arrange_mat(datas, shower=show_label, show_inds=False)
    return fig

# </editor-fold>


# if __name__ == '__main__':
#     img = Image.open('../res/cube.png')
#
#     # cate_lb = CategoryLabel(category=5, name='xx')
#     box_lb = BoxItem(category=5, border=[10, 10, 50, 50], name='car')
#     boxes_lb = BoxesLabel([box_lb, BoxItem(category=4, border=[40, 10, 70, 20], name='peo')], norm=(512, 512))
#
#     rgn = np.zeros(shape=(512, 512, 1))
#     rgn[20:300, 50:200] = 255
#     mask_lb = MaskLabel(rgn=rgn, category=5)
#
#     mask2 = np.zeros(shape=(512, 512, 1))
#     mask2[300:400, 100:150] = 255
#     mask2_lb = MaskLabel(rgn=mask2, category=1)
#
#     masks_lb = MaskesLabel(mask_lb, mask2_lb, norm=(512, 512))
#     # axplt_img_boxes(img, boxes_lb)
#     axplt_label(None, box_lb, use_pil=False)

# if __name__ == '__main__':
#     img = Image.open('../res/anchor.png')
#     mesh = load_objs_as_meshes(['../res/insu.obj'])
#
#     # mesh = anchor(1)
#
#     mesh_lb = StereoObjLabel(mesh=mesh, mesh_posi=[128, 384, 0], mesh_axang=[0, 0, 0], scale=20, category=3)
#     mesh_lb2 = StereoObjLabel(mesh=mesh, mesh_posi=[128, 128, 0], mesh_axang=[math.pi / 2, 0, 0], scale=20, category=3)
#
#     meshes_lb = StereoObjItem([mesh_lb, mesh_lb2], focus=5000, img_size=(384, 512))
#
#     # imgr = pilrnd_meshes(img, meshes_lb)
#     # imgr.show()
#     show_label(None, meshes_lb, use_pil=False)
