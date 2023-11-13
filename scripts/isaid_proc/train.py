import os
import sys

PROJECT_DIR = os.path.abspath(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
import argparse

from data import InsulatorObj, ISAIDObj
from tools import Metric, CRITERION, OneStageTrainer, LRScheduler
from models.box2seg import UInstV6
from utils import *
from data.voc import *


def get_parser():
    parser = argparse.ArgumentParser(description='infer')

    parser.add_argument('--wei_pth', default=None, help='weight path')
    parser.add_argument('--root', default=root, help='VOC dataset root:...root')
    parser.add_argument('--set_name_test', default='val_example', help='test set')
    parser.add_argument('--set_name_train', default='trainval_example', help='test set')

    parser.add_argument('--device', default=0, help='infer device')
    parser.add_argument('--img_size', default=(224, 224), help='img_size')
    parser.add_argument('--img_folder', default='JPEGImages', help='JPEGImages')
    parser.add_argument('--anno_folder', default='Annotations', help='Annotations')
    parser.add_argument('--inst_folder_train', default='InstancesTrain', help='InstancesTrain')
    parser.add_argument('--inst_folder', default='Instances', help='Instances')

    parser.add_argument('--set_folder', default='ImageSets/Main', help='ImageSets/Main')
    parser.add_argument('--num_workers', default=0, help='num_workers')
    parser.add_argument('--batch_size', default=4, help='batch_size')
    parser.add_argument('--conf_thres', default=0.5, help='conf_thres')

    parser.add_argument('--save_pth', default='./ckpt/buff', help='weight save pth')
    parser.add_argument('--save_pth_tea', default='./ckpt/buff_tea', help='weight save pth ema')
    parser.add_argument('--save_pth_eval', default='./eval.xlsx', help='eval result save pth')

    parser.add_argument('--metric', default='miou', help='miou or instap')

    parser.add_argument('--total_epoch', default=3, help='total_epoch')
    parser.add_argument('--total_stage', default=25, help='total_stage')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    root = args.root
    inst_folder = args.inst_folder
    inst_folder_train = args.inst_folder_train
    anno_folder = args.anno_folder
    img_folder = args.img_folder
    img_size = args.img_size
    set_folder = args.set_folder
    set_name_test = args.set_name_test
    set_name_train = args.set_name_train
    num_workers = args.num_workers
    wei_pth = args.wei_pth
    device = args.device
    batch_size = args.batch_size
    conf_thres = args.conf_thres
    save_pth = args.save_pth
    save_pth_tea = args.save_pth_tea
    save_pth_eval = args.save_pth_eval
    metric = args.metric
    total_epoch = args.total_epoch
    total_stage = args.total_stage

    cls_names = ISAIDObj.CLS_NAMES3
    ds = ISAIDObj(root=root, task_type=TASK_TYPE.DETECTION, anno_folder=anno_folder, inst_folder=inst_folder_train,
                      img_folder=img_folder, set_folder=set_folder, cls_names=cls_names)

    dataset = ds.dataset(set_name=set_name_train)
    vocD2vocI(dataset, colors=Voc.COLORS, root=ds.root, inst_folder=inst_folder_train)
    ds.task_type = TASK_TYPE.INSTANCE

    test_loader = ds.loader(set_name=set_name_test, batch_size=batch_size, pin_memory=False, shuffle=False,
                            num_workers=num_workers,
                            aug_seq=AugUInstNorm(img_size=img_size, thres=0, mode=MODE.REFLECT),
                            pre_type=PRE_TYPE.NONE,
                            anno_folder=anno_folder,
                            inst_folder=inst_folder, )

    model = UInstV6.ResUNetR34(device=device, pack=PACK.AUTO, num_cls=test_loader.num_cls,
                               img_size=img_size, )

    if wei_pth is not None and len(wei_pth) > 0:
        model.load(wei_pth, power=1)

    kwargs_eval = dict(conf_thres=conf_thres, num_infer=0, only_inner=True)
    kwargs_anno = dict(conf_thres=None, num_infer=0, only_inner=True)

    assert metric in ['miou', 'instap']
    if metric == 'miou':
        metric = Metric.MIOU(loader=test_loader, **kwargs_eval)
    else:
        metric = Metric.InstMAP(loader=test_loader, criterion=CRITERION.COCO_STD, **kwargs_eval)

    imscheduler = img_size
    trainner = OneStageTrainer(model=model, optimizer='sgd', metric=metric, best_pfrmce=0, interval=50)
    aug_seq = AugUInstV4(img_size=img_size, thres=0, p=0.3, mode=MODE.REFLECT, prob=0.3, with_exchange=True)
    for i in range(0, total_stage):
        print('Stage', i)
        lrscheduler = LRScheduler.WARM_COS(lr=0.002, warm_epoch=0, total_epoch=total_epoch)
        train_loader = ds.loader(set_name=set_name_train, batch_size=batch_size, pin_memory=False, shuffle=True,
                                 num_workers=num_workers,
                                 aug_seq=aug_seq,
                                 anno_folder=anno_folder,
                                 inst_folder=inst_folder_train,
                                 )
        anno_loader = ds.loader(set_name=set_name_train, batch_size=batch_size, pin_memory=False, shuffle=True,
                                num_workers=num_workers,
                                aug_seq=AugUInstAnno(img_size=img_size, thres=0, p=1, mode=MODE.REFLECT),
                                anno_folder=anno_folder,
                                inst_folder=inst_folder_train,
                                )
        trainner.train(loader=train_loader, lrscheduler=lrscheduler, imscheduler=imscheduler,
                       save_pth=save_pth, new_proc=True, accu_step=1, test_step=total_epoch, save_step=1,
                       with_frgd=True)

        if i > 0:
            model.load(save_pth_tea, power=0.5)
        model.save(save_pth_tea)
        labels_anno = model.loader2labels_with_surp(loader=anno_loader, conf_thres=None, num_infer=0,
                                                    only_inner=False, )
        for label in labels_anno:
            label.recover()
        anno_loader.dataset.update(labels_anno, with_recover=False,
                                   anno_folder=anno_folder, inst_folder=inst_folder_train, )
        model.load(save_pth)
        del labels_anno
        del anno_loader

    sys.exit(0)
