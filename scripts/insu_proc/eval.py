import os
import sys

from data import InsulatorObj
from tools import Metric, CRITERION

PROJECT_DIR = os.path.abspath(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
import argparse

from models.box2seg import UInstV1
from utils import *
from data.voc import *

wei_pth = 'E:\研究封存\弱监督绝缘子分割\Public\ckpt\insu_buff\insu_res50.pth'
root = 'D:\Datasets\InsulatorObj'


def get_parser():
    parser = argparse.ArgumentParser(description='infer')

    parser.add_argument('--wei_pth', default=wei_pth, help='weight path')
    parser.add_argument('--root', default=root, help='VOC dataset root:...root')
    parser.add_argument('--set_name_test', default='val', help='test set')
    parser.add_argument('--device', default=0, help='infer device')
    parser.add_argument('--img_size', default=(224, 224), help='img_size')
    parser.add_argument('--img_folder', default='JPEGImages', help='JPEGImages')
    parser.add_argument('--anno_folder', default='Annotations', help='Annotations')
    parser.add_argument('--inst_folder', default='Instances', help='Instances')
    parser.add_argument('--set_folder', default='ImageSets/Main', help='ImageSets/Main')
    parser.add_argument('--num_workers', default=5, help='num_workers')
    parser.add_argument('--batch_size', default=16, help='batch_size')
    parser.add_argument('--conf_thres', default=0.5, help='conf_thres')
    parser.add_argument('--save_pth', default='./eval.xlsx', help='eval result save pth')
    parser.add_argument('--metric', default='miou', help='miou or instap')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    root = args.root
    inst_folder = args.inst_folder
    anno_folder = args.anno_folder
    img_folder = args.img_folder
    img_size = args.img_size
    set_folder = args.set_folder
    set_name_test = args.set_name_test
    num_workers = args.num_workers
    wei_pth = args.wei_pth
    device = args.device
    batch_size = args.batch_size
    conf_thres = args.conf_thres
    save_pth = args.save_pth
    metric = args.metric

    cls_names = ('insulator_glass',)
    ds = InsulatorObj(root=root, task_type=TASK_TYPE.INSTANCE, inst_folder=inst_folder, anno_folder=anno_folder,
                      img_folder=img_folder, set_folder=set_folder, cls_names=cls_names)

    test_loader = ds.loader(set_name=set_name_test, batch_size=batch_size, pin_memory=False, shuffle=True,
                            num_workers=num_workers,
                            aug_seq=AugUInstNorm(img_size=img_size, thres=0, mode=MODE.REFLECT),
                            )

    kwargs_eval = dict(conf_thres=conf_thres, num_infer=0, only_inner=True)
    model = UInstV1.ResUNetR50(device=device, pack=PACK.AUTO, num_cls=test_loader.num_cls, img_size=img_size)
    model.load(wei_pth, power=1)

    assert metric in ['miou', 'instap']
    if metric == 'miou':
        metric = Metric.MIOU(loader=test_loader, save_pth=save_pth, **kwargs_eval)
    else:
        metric = Metric.InstMAP(loader=test_loader, criterion=CRITERION.COCO_STD, save_pth=save_pth, **kwargs_eval)

    metric(model)
