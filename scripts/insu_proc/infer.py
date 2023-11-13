import os
import sys

PROJECT_DIR = os.path.abspath(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
import argparse

from models.box2seg import UInstV1
from utils import *
from data.voc import *

wei_pth = 'E:\研究封存\弱监督绝缘子分割\Public\ckpt\insu_buff\insu_res50.pth'
set_pth = 'D:\Datasets\InsulatorObj\ImageSets\Main//val.txt'
img_dir = 'D:\Datasets\InsulatorObj\JPEGImages'
anno_dir = 'D:\Datasets\InsulatorObj\PatchAnnotations'
inst_dir = 'D:\Datasets\InsulatorObj\Instances'


def get_parser():
    parser = argparse.ArgumentParser(description='infer')

    parser.add_argument('--wei_pth', default=wei_pth, help='weight path')
    parser.add_argument('--img_dir', default=img_dir, help='images directory:...root\JPEGImages')
    parser.add_argument('--anno_dir', default=anno_dir, help='annotation directory:...root\Annotations')
    parser.add_argument('--inst_dir', default=inst_dir, help='instance directory:...root\Instances')
    parser.add_argument('--device', default=0, help='infer device')
    parser.add_argument('--img_size', default=(224, 224), help='img_size')
    parser.add_argument('--num_cls', default=2, help='num cls')
    parser.add_argument('--set_pth', default=None, help='set path:...root\ImageSets\Main//val.txt')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    img_dir = args.img_dir
    anno_dir = args.anno_dir
    inst_dir = args.inst_dir
    wei_pth = args.wei_pth
    device = args.device
    num_cls = args.num_cls
    img_size = args.img_size
    set_pth = args.set_pth

    model = UInstV1.ResUNetR50(device=device, pack=PACK.AUTO, num_cls=num_cls, img_size=img_size, )
    model.load(wei_pth, power=1)

    if set_pth is not None:
        metas = load_txt(set_pth)
    else:
        metas = [os.path.splitext(img_name)[0] for img_name in os.listdir(img_dir)]

    ensure_folder_pth(inst_dir)
    for i, meta in MEnumerate(metas):
        img_pth = os.path.join(img_dir, meta + '.jpg')
        anno_pth = os.path.join(anno_dir, meta + '.xml')
        inst_pth = os.path.join(inst_dir, meta + '.png')

        img = Image.open(img_pth).convert('RGB')
        label = VocDDataset.prase_anno(anno_pth, num_cls=num_cls, name2cind=None)

        label_md = model.imgs_labels2labels(imgs=[img], labels=[label], conf_thres=0.3, num_infer=0,
                                            cind2name=None, only_inner=True, new_border=True)[0]
        VocIDataset.create_inst(inst_pth, colors=Voc.COLORS, insts=label_md)
