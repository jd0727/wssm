import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from data.voc import Voc
from data.folder import FolderDataset
from utils.frame import *


class InsulatorC(DataSource):
    ROOT_SEV_NEW = '/ses-data/JD//InsulatorC//'
    ROOT_SEV_NEW_RAWC = '//home//data-storage//JD//RawC//unknown//'
    ROOT_SEV_NEW_BKGD = '//ses-img//JD//Bkgd//'
    ROOT_SEV_OLD = '//home//user1//JD//Datasets//InsulatorC//'
    ROOT_DES = 'D://Datasets//InsulatorC//'

    @staticmethod
    def SEV_NEW(**kwargs):
        return InsulatorC(root=InsulatorC.ROOT_SEV_NEW, resample={'abnormal': 3}, **kwargs)

    @staticmethod
    def SEV_OLD(**kwargs):
        return InsulatorC(root=InsulatorC.ROOT_SEV_OLD, resample={'abnormal': 3}, **kwargs)

    @staticmethod
    def DES(**kwargs):
        return InsulatorC(root=InsulatorC.ROOT_DES, resample={'abnormal': 3}, **kwargs)

    def __init__(self, root, resample=None, pre_aug_seq=None, **kwargs):
        super().__init__(root=root, set_names=('train', 'test'))
        self.pth = root
        self.resample = resample
        self.pre_aug_seq = pre_aug_seq

    def dataset(self, set_name, **kwargs):
        dataset = FolderDataset(root=os.path.join(self.pth, set_name), resample=self.resample,
                                pre_aug_seq=self.pre_aug_seq,
                                cls_names=('abnormal', 'normal'))
        return dataset


class InsulatorD(Voc):
    CLS_NAMES = ('insulator_comp', 'insulator_normal', 'insulator_blast')
    CLS_NAMES_BORDER = ('insulator',)
    CLS_NAMES_MERGE = ('insulator_comp', 'insulator_glass')
    COLORS = Voc.COLORS

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = Voc.MASK_FOLDER
    INST_FOLDER = Voc.INST_FOLDER
    SET_FOLDER = Voc.SET_FOLDER_DET

    ROOT_SEV_OLD = '/home/user/JD/Datasets/InsulatorD'
    ROOT_SEV_NEW_ORI = '//ses-img//CY//jueyuanzi'
    ROOT_SEV_NEW = '//home//data-storage//JD//InsulatorD'
    ROOT_BOARD = '/home/jd/data/DataSets/InsulatorD'
    ROOT_DES = 'D://Datasets//InsulatorD//'

    @staticmethod
    def DES(**kwargs):
        return InsulatorD(root=InsulatorD.ROOT_DES, **kwargs)

    @staticmethod
    def BOARD(**kwargs):
        return InsulatorD(root=InsulatorD.ROOT_BOARD, **kwargs)

    @staticmethod
    def SEV_OLD(**kwargs):
        return InsulatorD(root=InsulatorD.ROOT_SEV_OLD, **kwargs)

    @staticmethod
    def SEV_NEW_ORI(**kwargs):
        return InsulatorD(root=InsulatorD.ROOT_SEV_NEW_ORI, **kwargs)

    @staticmethod
    def SEV_NEW(**kwargs):
        return InsulatorD(root=InsulatorD.ROOT_SEV_NEW, **kwargs)

    def __init__(self, root, pre_aug_seq=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval', 'example'), **kwargs):
        super().__init__(root=root, pre_aug_seq=pre_aug_seq, cls_names=cls_names, colors=colors, task_type=task_type,
                         mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                         img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)

    def dataset(self, set_name='train', task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = Voc.BUILDER_MAPPER[task_type]
        kwargs_update = dict(img_folder=self.img_folder, cls_names=self.cls_names,
                             anno_folder=self.anno_folder, set_folder=self.set_folder, set_name=set_name,
                             mask_folder=self.mask_folder, inst_folder=self.inst_folder, colors=self.colors)
        kwargs_update.update(kwargs)
        kwargs_update.update(self.kwargs)
        dataset = builder(root=self.root, fmt=set_name if set_name is not None else 'all' + '_%d', **kwargs_update)
        return dataset


class InsulatorDI(Voc):
    CLS_NAMES = ('insulator_normal', 'insulator_blast')
    COLORS = Voc.COLORS

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = Voc.MASK_FOLDER
    INST_FOLDER = Voc.INST_FOLDER
    SET_FOLDER = Voc.SET_FOLDER_DET

    ROOT_SEV_NEW = '//home//data-storage//JD//InsulatorDI'

    @staticmethod
    def SEV_NEW(**kwargs):
        return InsulatorDI(root=InsulatorDI.ROOT_SEV_NEW, **kwargs)

    def __init__(self, root, pre_aug_seq=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=None, **kwargs):
        super().__init__(root=root, pre_aug_seq=pre_aug_seq, cls_names=cls_names, colors=colors, task_type=task_type,
                         mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                         img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)

    def dataset(self, set_name='train', task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = Voc.BUILDER_MAPPER[task_type]
        kwargs_update = dict(img_folder=self.img_folder, cls_names=self.cls_names,
                             anno_folder=self.anno_folder, set_folder=self.set_folder, set_name=set_name,
                             mask_folder=self.mask_folder, inst_folder=self.inst_folder, colors=self.colors)
        kwargs_update.update(kwargs)
        kwargs_update.update(self.kwargs)
        dataset = builder(root=self.root, fmt=set_name + '_%d', **kwargs_update)
        return dataset


class InsulatorObj(Voc):
    ROOT_SEV_NEW = '//ses-data//JD//InsulatorObj'
    ROOT_DES = 'D://Datasets//InsulatorObj//'
    COLORS = Voc.COLORS
    CLS_NAMES = ('insulator_glass',)
    IMG_FOLDER = 'Patches'
    SET_FOLDER = 'ImageSets/Patch'
    ANNO_FOLDER = 'PatchAnnotations'
    INST_FOLDER = 'PatchInstance'

    def __init__(self, root, cls_names=CLS_NAMES, colors=COLORS,
                 task_type=TASK_TYPE.DETECTION, img_folder=IMG_FOLDER, anno_folder=ANNO_FOLDER,
                 mask_folder=Voc.MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, **kwargs):
        super().__init__(root, cls_names, colors, task_type, mask_folder, inst_folder, set_folder,
                         img_folder, anno_folder, **kwargs)

    def dataset(self, set_name='train', task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = InsulatorObj.BUILDER_MAPPER[task_type]
        kwargs_update = dict(img_folder=self.img_folder, cls_names=self.cls_names,
                             anno_folder=self.anno_folder, set_folder=self.set_folder, set_name=set_name,
                             mask_folder=self.mask_folder, inst_folder=self.inst_folder, colors=self.colors)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = builder(root=self.root, fmt=set_name + '_%d' if set_name is not None else 'all_%d', **kwargs_update)
        return dataset
