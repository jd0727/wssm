import os
import sys

PROJECT_DIR = os.path.abspath(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
import argparse
from data import InsulatorObj, InsulatorDI

from data.voc import *


def imgP_affine(imgP, scale=1.0, angle=0.0, shear=0.0, resample=Image.BICUBIC):
    img_size = np.array(imgP.size)
    img_size_scled = (img_size * scale).astype(np.int32)
    A = np.array([[np.cos(angle + shear), np.sin(angle + shear)],
                  [-np.sin(angle - shear), np.cos(angle - shear)]]) * scale
    Ai = np.linalg.inv(A)
    bi = img_size / 2 - Ai @ img_size_scled / 2
    data = [Ai[0, 0], Ai[0, 1], bi[0], Ai[1, 0], Ai[1, 1], bi[1]]
    imgP = imgP.transform(size=tuple(img_size_scled), data=data,
                          method=Image.AFFINE, resample=resample, )
    return imgP


def gen_data(dataset_src, dataset_tar, num_gen, num_samp=5, scale=(0.5, 2), rotate=(0, 360), thres=32,
             samp_dct=None, with_mask=True, ensure_one=True):
    num_tar = len(dataset_tar)
    imgs = []
    labels = []

    print('Fliter valid samples')
    plabels = dataset_src.labels
    inds_valid = []
    powers = []
    for j, plabel in enumerate(plabels):
        pinst = plabel[0]
        name = pinst['name']
        if samp_dct is None:
            power = 1
        else:
            if name not in samp_dct.keys():
                continue
            else:
                power = samp_dct[name]
        if thres > 0 and pinst.border.measure() < thres:
            continue
        inds_valid.append(j)
        powers.append(power)
    powers = np.array(powers)
    powers = powers / np.sum(powers)
    inds_valid = np.array(inds_valid)
    print('Get valid samples %d' % len(inds_valid))
    num_gen = num_gen if isinstance(num_gen, int) else int(num_gen * num_tar)
    print('Generate %d samples' % num_gen + ' to target %d' % num_tar)
    inds_tar = np.random.choice(a=num_tar, size=num_gen, replace=num_gen > num_tar)

    for i, ind_tar in MEnumerate(inds_tar, prefix='Pasting ', with_eta=True):
        img, label = dataset_tar[ind_tar]
        xyxys = label.export_xyxysN()
        num_cur = 0
        while (ensure_one and len(xyxys) == 0) or num_cur < num_samp:
            num_cur += 1
            ind_src = np.random.choice(inds_valid, size=1, p=powers)[0]
            piece, plabel = dataset_src[ind_src]
            pinst = plabel[0]
            piece = img2imgP(piece)

            ratio = np.random.uniform(low=scale[0], high=scale[1])
            psize_scld = np.array((int(ratio * piece.size[0]), int(ratio * piece.size[1])))
            if np.any(psize_scld >= np.array(img.size)) or np.sqrt(np.prod(psize_scld)) < thres:
                continue
            if with_mask:
                angle = np.random.uniform(low=rotate[0], high=rotate[1])
                piece = imgP_affine(piece, scale=ratio, angle=angle)
                mask = imgN2imgP(pinst.rgn.maskNb.astype(np.float32) * 255)
                mask = imgP_affine(mask, scale=ratio, angle=angle)
            else:
                mask = imgN2imgP(pinst.border.maskNb.astype(np.float32) * 255)
                piece = imgP_affine(piece, scale=ratio)
                mask = imgP_affine(mask, scale=ratio)
            maskNb = np.array(mask) > 128
            xy = [np.random.randint(low=0, high=img.size[0] - psize_scld[0]),
                  np.random.randint(low=0, high=img.size[1] - psize_scld[1])]
            xyxy_samp = RefValRegion._maskNb2xyxyN(maskNb)
            xyxy_samp = xyxy_samp + np.array([xy[0], xy[1], xy[0], xy[1]])
            iareas = ropr_arr_xyxysN(np.repeat(xyxy_samp[None, :], axis=0, repeats=xyxys.shape[0]), xyxys,
                                     opr_type=OPR_TYPE.IAREA)

            if np.any(iareas > 0): continue
            xyxys = np.concatenate([xyxys, xyxy_samp[None, :]], axis=0)

            measure = np.sqrt(np.sum(maskNb))

            if with_mask:
                kernel = np.ones(shape=(int(measure / 3), int(measure / 3)))  # 扩大选取范围
                mask = cv2.dilate(np.array(mask), kernel)
                fltr = np.any(np.array(piece) > 20, axis=2)
                mask = mask * fltr
                mask = gaussian_filter(mask, sigma=measure / 10)
                mask = mask * fltr
                mask = imgN2imgP(mask)
            img.paste(piece, xy, mask=mask)
            pbox = BoxItem.convert(pinst)
            pbox.border = XYXYBorder(xyxy_samp, size=img.size)
            label.append(pbox)

        imgs.append(img)
        # label.meta = label.meta + '_A%6d' % np.random.randint(low=0, high=1e6)
        label.meta = 'AM_' + label.meta
        labels.append(label)
    return imgs, labels


def update_metas(metas, metas_updt):
    meta_dct = dict([(meta.split('_')[1], meta) for meta in metas_updt])
    for i in range(len(metas)):
        if metas[i] in meta_dct.keys():
            metas[i] = meta_dct[metas[i]]
    return metas


def get_parser():
    parser = argparse.ArgumentParser(description='infer')

    parser.add_argument('--wei_pth', default=None, help='weight path')
    parser.add_argument('--root', default='', help='VOC dataset root:...root')
    parser.add_argument('--set_name_train', default='trainval_example', help='test set')
    parser.add_argument('--img_folder', default='JPEGImages', help='JPEGImages')
    parser.add_argument('--anno_folder', default='Annotations', help='Annotations')
    parser.add_argument('--inst_folder_train', default='InstancesTrain', help='InstancesTrain')

    parser.add_argument('--root_dst', default='', help='Dist VOC dataset root:...root')
    parser.add_argument('--img_folder_dst', default='', help='JPEGImages')
    parser.add_argument('--anno_folder_dst', default='', help='Annotations')
    parser.add_argument('--set_name_dst', default='bkgd', help='background set')
    parser.add_argument('--set_name_app', default='app', help='append set')

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    root = args.root
    inst_folder = args.inst_folder
    inst_folder_train = args.inst_folder_train
    set_name_train = args.set_name_train
    anno_folder = args.anno_folder
    img_folder = args.img_folder

    root_dst = args.root_dst
    img_folder_dst = args.img_folder_dst
    anno_folder_dst = args.anno_folder_dst
    set_name_dst = args.set_name_dst
    set_name_app = args.set_name_app

    ds_src = InsulatorObj(
        root=root,
        task_type=TASK_TYPE.INSTANCE,
        cls_names=InsulatorDI.CLS_NAMES,
        anno_folder=anno_folder,
        inst_folder=inst_folder_train,
    )
    ds_tar = InsulatorDI(
        root=root_dst,
        task_type=TASK_TYPE.DETECTION,
        cls_names=InsulatorDI.CLS_NAMES,
        anno_folder=anno_folder_dst,
        img_folder=img_folder_dst
    )
    dataset_src = ds_src.dataset(set_name_train)
    with_mask = True

    dataset_dst = ds_tar.dataset(set_name_dst)

    samp_dct = {'insulator_normal': 1, 'insulator_blast': 1}
    imgs, labels = gen_data(dataset_src, dataset_dst, num_gen=1.0, samp_dct=samp_dct, ensure_one=True,
                            num_samp=5, scale=(1 / 2, 2), rotate=(-0, 0), thres=64, with_mask=with_mask)
    dataset_dst.append(imgs, labels, anno_folder=anno_folder_dst, img_folder=img_folder_dst, set_name=set_name_app,
                       with_recover=True)
