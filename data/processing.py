import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from utils import *


def encode_meta_xyxy(meta, xyxy):
    return meta + '_' + '_'.join(['%04d' % v for v in xyxy])


def decode_meta_xyxy(meta):
    meta_p = meta.split('_')
    xyxy = np.array([int(v) for v in meta_p[-4:]])
    meta = '_'.join(meta_p[:-4])
    return meta, xyxy


# 检测框样本区域扩展策略
def _xyxy_expend_with_cxt(xyxys, cinds, index, img_size, expend_ratio=1.1, as_square=False,
                          avoid_overlap=True, with_clip=False, min_expand=3):
    xywh_ori = xyxyN2xywhN(xyxys[index])
    xywh = copy.deepcopy(xywh_ori)
    xywh[2:4] = np.maximum(xywh_ori[2:4] * expend_ratio, xywh_ori[2:4] + 2 * min_expand)
    if as_square and not avoid_overlap:
        xywh[2:4] = np.max(xywh[2:4])
    elif as_square and avoid_overlap:
        xywh_try = copy.deepcopy(xywh)
        xywh_try[2:4] = np.max(xywh_try[2:4])
        xyxy_try = xywhN2xyxyN(xywh_try)
        # 获取所有同类标签
        fliter = np.full_like(cinds, fill_value=True, dtype=bool)
        fliter[index] = False
        fliter *= (cinds == cinds[index])
        xyxys_cxt = xyxys[fliter]
        # 计算重叠面积
        iarea = ropr_arr_xyxysN(np.broadcast_to(xyxy_try, xyxys_cxt.shape), xyxys_cxt, opr_type=OPR_TYPE.IAREA)
        xywh = xywh if np.any(iarea > 0) else xywh_try
    xywh[2:4] = np.ceil(xywh[2:4])
    xyxy = xywhN2xyxyN(xywhN=xywh)
    if with_clip:
        xyxy = xyxyN_clip(xyxy, xyxyN_rgn=np.array([0, 0, img_size[0], img_size[1]]))
    return xyxy


def _items_flit(items, cls_names=None, difficult=None, thres=-1):
    items_fltd = items.__class__(img_size=items.img_size, meta=items.meta)
    for item in items:
        if cls_names is not None and item['name'] not in cls_names:
            continue
        if difficult is not None and not item.get('difficult', False) == difficult:
            continue
        if thres > 0 and item.measure() < thres:
            continue
        items_fltd.append(item)
    return items_fltd


def img2piece_persize(img, items, piece_size=(640, 640), thres=5, over_lap=(100, 100), ignore_empty=True,
                      with_clip=False, cls_names=None, meta_encoder=None):
    pieces = []
    plabels = []
    img = img2imgP(img)
    items = _items_flit(items, cls_names=cls_names, difficult=None)
    piece_size = np.array(piece_size)
    over_lap = np.array(over_lap)
    img_size = np.array(img.size)
    step_size = piece_size - over_lap
    assert np.all(step_size > 0), 'size err'
    nwh = np.ceil((img_size - over_lap) / step_size)
    meta_encoder = encode_meta_xyxy if meta_encoder is None else meta_encoder
    for i in range(int(nwh[0])):
        for j in range(int(nwh[1])):
            offset = np.array([i * step_size[0], j * step_size[1]])
            piece_rgn = np.concatenate([offset, offset + piece_size]).astype(np.int32)
            if with_clip:
                piece_rgn[2:4] = np.minimum(piece_rgn[2:4], img_size)
                piece_rgn[:2] = np.maximum(piece_rgn[2:4] - piece_size, np.array([0, 0]))
                # piece_rgn = xyxyN_clip(piece_rgn, xyxyN_rgn=np.array([0, 0, img_size[0], img_size[1]]))
            meta_piece = meta_encoder(items.meta, piece_rgn)
            piece_size_j = tuple(piece_rgn[2:] - piece_rgn[:2])
            assert isinstance(items, ImageItemsLabel), 'fmt err ' + items.__class__.__name__
            pitems = items.__class__(img_size=piece_size_j, meta=meta_piece)
            for item in copy.deepcopy(items):
                if thres > 0 and item.clip(piece_rgn).measure() < thres:
                    continue
                item.linear(bias=-piece_rgn[:2], size=piece_size_j)
                pitems.append(item)
            if isinstance(pitems, InstsLabel):
                pitems.align()
                pitems.avoid_overlap()
                for item in pitems:
                    item.rgn = RefValRegion.from_maskNb_xyxyN(item.rgn.maskNb, XYXYBorder.convert(item.border).xyxyN)

            if ignore_empty and len(pitems) == 0:
                continue
            piece = img.crop(list(piece_rgn))
            # print(items.meta,i,j,len(pitems))
            plabels.append(pitems)
            pieces.append(piece)
    return pieces, plabels


def img2piece_perbox(img, items, expend_ratio=1.2, with_clip=False, as_square=False, avoid_overlap=True,
                     cls_names=None, only_one=True, thres=5, ratio_thres=0.2, ignore_empty=True, difficult=None,
                     meta_encoder=None):
    pieces = []
    plabels = []
    items_fltd = _items_flit(items, cls_names=cls_names, difficult=difficult)
    xyxys, cinds = items_fltd.export_xyxysN(), items_fltd.export_cindsN()
    img = img2imgP(img)
    meta_encoder = encode_meta_xyxy if meta_encoder is None else meta_encoder
    for j, item in enumerate(items_fltd):
        # if items.meta=='P0001' and j==53:
        #     print('Testing')
        if thres > 0 and item.measure() < thres:
            continue
        xyxy_ext = _xyxy_expend_with_cxt(
            xyxys, cinds, index=j, img_size=img.size, expend_ratio=expend_ratio, as_square=as_square,
            avoid_overlap=avoid_overlap, with_clip=with_clip).astype(np.int32)

        meta = meta_encoder(items_fltd.meta, xyxy_ext)
        patch_size = tuple(xyxy_ext[2:4] - xyxy_ext[:2])

        pitems = items_fltd.__class__(img_size=patch_size, meta=meta, xyxy_rgn=xyxy_ext, meta_ori=items_fltd.meta)
        if only_one:
            item.linear(bias=-xyxy_ext[:2], size=patch_size)
            pitems.append(item)
        else:
            for item_cp in copy.deepcopy(items_fltd):
                item_mea = item_cp.clip(xyxy_ext).measure()
                if item_mea < max(np.max(patch_size) * ratio_thres, thres):
                    continue
                item_cp.linear(bias=-xyxy_ext[:2], size=patch_size)
                # if isinstance(item_cp, InstItem):
                #     item_cp.align()
                pitems.append(item_cp)
        # if isinstance(pitems, InstsLabel):
        #     pitems.avoid_overlap()
        #     pitems.align()
        if ignore_empty and len(pitems) == 0:
            continue
        piece = img.crop(list(xyxy_ext))
        plabels.append(pitems)
        pieces.append(piece)
    return pieces, plabels


def img2img_flit(img, items, thres=-1, cls_names=None, difficult=None, ignore_empty=True):
    items_fltd = _items_flit(items, cls_names=cls_names, difficult=difficult, thres=thres)
    if ignore_empty and len(items_fltd) == 0:
        return [], []
    else:
        img = img2imgP(img)
        return [img], [items_fltd]


def img2background(img, items, min_size=0, max_size=16, repeat_num=1.0, cls_names=None):
    pieces = []
    plabels = []
    items_fltd = _items_flit(items, cls_names=cls_names, difficult=None)
    xyxys_fltd = items_fltd.export_xyxysN()
    img = img2imgP(img)
    for j in range(int(np.ceil(repeat_num))):
        if j + np.random.rand() > repeat_num: continue
        size_ratio = np.random.rand()
        cur_size = int(size_ratio * max_size + (1 - size_ratio) * min_size)
        offset = ((np.array(img.size) - cur_size) * np.random.rand(2)).astype(np.int32)
        xyxy = np.concatenate([offset, offset + cur_size], axis=0)
        iareas = ropr_arr_xyxysN(np.repeat(xyxy[None, :], axis=0, repeats=xyxys_fltd.shape[0]), xyxys_fltd,
                                 opr_type=OPR_TYPE.IAREA)
        if np.any(iareas > 0): continue

        piece = img.crop(xyxy)
        meta = encode_meta_xyxy(items.meta, xyxy)
        pitems = items.__class__(img_size=(cur_size, cur_size), meta=meta)
        plabels.append(pitems)
        pieces.append(piece)
    return pieces, plabels


def dataset2background(dataset, bkgd_dir, min_size=0, max_size=16, repeat_num=1.0, cls_names=None):
    ensure_folder_pth(bkgd_dir)
    print('Start convert [ ' + dataset.__class__.__name__ + ' ]')
    for i, (img, items) in MEnumerate(dataset, prefix='Converting ', with_eta=True):
        pieces_i, plabels_i = img2background(img, items, min_size=min_size, max_size=max_size,
                                             repeat_num=repeat_num, cls_names=cls_names)
        for piece_i, plabel_i in zip(pieces_i, plabels_i):
            piece_pth = os.path.join(bkgd_dir, plabel_i.meta + '.jpg')
            piece_i.save(piece_pth)

    print('Convert complete')
    return None


def dataset2piece_persize(dataset, piece_size=(640, 640), thres=5, over_lap=(100, 100), with_clip=False,
                          cls_names=None, ignore_empty=True):
    pieces = []
    plabels = []
    print('Start convert [ ' + dataset.__class__.__name__ + ' ]')
    for i, (img, items) in MEnumerate(dataset, prefix='Converting ', with_eta=True):
        pieces_i, plabels_i = img2piece_persize(img, items, piece_size=piece_size, thres=thres,
                                                over_lap=over_lap, cls_names=cls_names, ignore_empty=ignore_empty,
                                                with_clip=with_clip)
        pieces += pieces_i
        plabels += plabels_i
    print('Convert complete')
    return pieces, plabels


def piece_merge(plabels, iou_thres=0.0, meta_decoder=None):
    meta_dict = {}
    meta_decoder = decode_meta_xyxy if meta_decoder is None else meta_decoder
    print('Cluster label for every image')
    for plabel in plabels:
        meta, xyxy_rgn = meta_decoder(plabel.meta)
        piece = (xyxy_rgn, plabel)
        if meta in meta_dict.keys():
            meta_dict[meta].append(piece)
        else:
            meta_dict[meta] = [piece]
    labels = []
    print('Merge label for every image')
    for i, (meta, pieces) in MEnumerate(meta_dict.items(), prefix='Merging'):
        label = []
        max_size = np.array([-np.inf, -np.inf])
        for xyxy_rgn, plabel in pieces:
            max_size = np.maximum(max_size, xyxy_rgn[2:])
            plabel.linear(bias=xyxy_rgn[:2], size=tuple(xyxy_rgn[2:]))
            label += plabel

        img_size = tuple(max_size.astype(np.int32))
        if iou_thres > 0:
            boxes_label = BoxesLabel(label, img_size=img_size, meta=meta)
            cindsN, confsN = boxes_label.export_cindsN_confsN()
            xlyls = boxes_label.export_xlyls()
            prsv_inds = nms_xlyls(xlyls, confsN, cindsN, iou_thres=iou_thres, iou_type=IOU_TYPE.IOU)
            label = [label[i] for i in prsv_inds]

        # if meta=='P0001':
        #     print('testing')
        label = ImageItemsLabel(label, img_size=img_size, meta=meta)
        labels.append(label)
    return labels
