import os
import sys

PROJECT_DIR = os.path.abspath(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
import argparse

from utils import *
from data import *
import shutil


def decode_meta_xyxy_isaid(meta, width_def=800, height_def=800):
    if '_' not in meta:
        return meta, [0, 0, width_def, height_def]
    meta_p = meta.split('_')
    xyxy = np.array([int(v) for v in meta_p[-4:]])
    meta = '_'.join(meta_p[:-4])
    xyxy = [xyxy[2], xyxy[0], xyxy[3], xyxy[1]]
    return meta, xyxy


def meta_encoder(meta, xyxy):
    meta, xyxy_base = decode_meta_xyxy_isaid(meta)
    xyxy = xyxy + np.array([xyxy_base[0], xyxy_base[1], xyxy_base[0], xyxy_base[1]])
    return meta + '_' + '_'.join(['%04d' % v for v in xyxy])


def get_parser():
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--root_isaid', default=root, help='ISAID dataset root:...root')
    parser.add_argument('--json_name_isaid', default='instances_train', help='instances_train.json')
    parser.add_argument('--img_folder_isaid', default='images_train', help='images_train')

    parser.add_argument('--root', default=root, help='VOC dataset root:...root')
    parser.add_argument('--img_folder', default='JPEGImages', help='JPEGImages')
    parser.add_argument('--anno_folder', default='Annotations', help='Annotations')
    parser.add_argument('--set_folder', default='ImageSets/Main', help='ImageSets/Main')
    parser.add_argument('--inst_folder', default='Instances', help='Instances')
    return parser


# DOTA数据集裁剪
if __name__ == '__main__':
    args = get_parser().parse_args()
    root_isaid = args.root_isaid
    json_name_isaid = args.json_name_isaid
    img_folder_isaid = args.img_folder_isaid

    root = args.root
    img_folder = args.img_folder
    anno_folder = args.anno_folder
    set_folder = args.set_folder
    inst_folder = args.inst_folder

    ds = ISAIDPatch(root=root_isaid, task_type=TASK_TYPE.INSTANCE)

    # ds_tar = ISAIDObj.SEV_NEW1()
    # thres = 0
    # ds_tar = ISAIDObj.SEV_NEW2()
    # thres=0
    ds_tar = ISAIDObj(root=root, cls_names=ISAIDObj.CLS_NAMES3, )
    thres = 0

    cls_names = ds_tar.cls_names
    if os.path.exists(root):
        print('Remove old')
        shutil.rmtree(root)
    colors = Voc.COLORS

    dataset = ds.dataset(set_name='train', json_name=json_name_isaid, img_folder=img_folder_isaid)
    metas = datasetI2vocI_perbox(dataset, 'train', colors, root, set_folder=set_folder, patch_folder=img_folder,
                                 inst_folder=inst_folder, expend_ratio=1.2, anno_folder=anno_folder,
                                 with_clip=False, as_square=True, avoid_overlap=False, thres=thres, cls_names=cls_names,
                                 only_one=True, meta_encoder=meta_encoder)
