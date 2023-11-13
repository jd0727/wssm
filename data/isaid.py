import os

from data.coco import CoCo
from data.voc import Voc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from utils import *


class ISAID(CoCo):
    ROOT_SEV_NEW = '//home//data-storage//ISAID'
    ROOT_DES = 'D://Datasets//ISAID//'

    IMG_PREFIX = 'images_'
    JSON_PREFIX = 'instances_'
    JSON_FOLDER = 'annotation'

    CLS_NAMES = ('storage_tank', 'Large_Vehicle', 'Small_Vehicle', 'plane', 'ship',
                 'Swimming_pool', 'Harbor', 'tennis_court', 'Ground_Track_Field', 'Soccer_ball_field',
                 'baseball_diamond', 'Bridge', 'basketball_court', 'Roundabout', 'Helicopter')
    # CLS_NAMES = ('bridge', 'ground-track-field', 'harbor', 'helicopter', 'large-vehicle',
    #              'roundabout', 'small-vehicle', 'soccer-ball-field', 'swimming-pool', 'baseball-diamond',
    #              'basketball-court', 'plane', 'ship', 'storage-tank', 'tennis-court')

    # CIND2NAME_REMAPPER_DICT = {
    #     2: 'storage-tank', 8: 'large-vehicle', 9: 'small-vehicle', 1: 'ship', 15: 'harbor', 3: 'baseball-diamond',
    #     6: 'ground-track-field', 13: 'soccer-ball-field', 11: 'swimming-pool', 12: 'roundabout', 4: 'tennis-court',
    #     5: 'basketball-court', 14: 'plane', 10: 'helicopter', 7: 'bridge'}

    # CIND2NAME_REMAPPER_DICT = {
    #     1: 'storage-tank', 2: 'large-vehicle', 3: 'small-vehicle', 4: 'plane', 5: 'ship', 6: 'swimming-pool',
    #     7: 'harbor', 8: 'tennis-court', 9: 'ground-track-field', 10: 'soccer-ball-field', 11: 'baseball-diamond',
    #     12: 'bridge', 13: 'basketball-court', 14: 'roundabout', 15: 'helicopter'}

    CIND2NAME_REMAPPER = None

    @staticmethod
    def SEV_NEW(**kwargs):
        return ISAID(root=ISAID.ROOT_SEV_NEW, **kwargs)

    @staticmethod
    def DES(**kwargs):
        return ISAID(root=ISAID.ROOT_DES, **kwargs)

    def __init__(self, root, json_prefix=JSON_PREFIX, img_prefix=IMG_PREFIX, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES, cind2name_remapper=None,
                 set_names=None, **kwargs):
        super().__init__(root, json_prefix=json_prefix, img_prefix=img_prefix, json_folder=json_folder,
                         cind2name_remapper=cind2name_remapper, task_type=task_type, cls_names=cls_names,
                         set_names=set_names, **kwargs)


class ISAIDPatch(CoCo):
    ROOT_SEV_NEW = '//home//data-storage//ISAID'
    ROOT_DES = 'D://Datasets//ISAID//'

    IMG_PREFIX = 'patches_'
    JSON_PREFIX = 'instances_'
    JSON_FOLDER = 'annotation_ptch'

    CLS_NAMES = ISAID.CLS_NAMES

    CIND2NAME_REMAPPER_DICT = {
        9: 'Small_Vehicle', 8: 'Large_Vehicle', 14: 'plane', 2: 'storage_tank', 1: 'ship',
        11: 'Swimming_pool', 15: 'Harbor', 4: 'tennis_court', 6: 'Ground_Track_Field', 13: 'Soccer_ball_field',
        3: 'baseball_diamond', 7: 'Bridge', 5: 'basketball_court', 12: 'Roundabout', 10: 'Helicopter'}
    NAME2CIND_REMAPPER_DICT = dict([(name, cind) for cind, name in CIND2NAME_REMAPPER_DICT.items()])
    CIND2NAME_REMAPPER = lambda cind: ISAIDPatch.CIND2NAME_REMAPPER_DICT[cind]
    NAME2CIND_REMAPPER = lambda cind: ISAIDPatch.NAME2CIND_REMAPPER_DICT[cind]

    @staticmethod
    def SEV_NEW(**kwargs):
        return ISAIDPatch(root=ISAIDPatch.ROOT_SEV_NEW, **kwargs)

    @staticmethod
    def DES(**kwargs):
        return ISAIDPatch(root=ISAIDPatch.ROOT_DES, **kwargs)

    def __init__(self, root, json_prefix=JSON_PREFIX, img_prefix=IMG_PREFIX, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES, cind2name_remapper=CIND2NAME_REMAPPER,
                 set_names=None, **kwargs):
        super().__init__(root, json_prefix=json_prefix, img_prefix=img_prefix, json_folder=json_folder,
                         cind2name_remapper=cind2name_remapper, task_type=task_type, cls_names=cls_names,
                         set_names=set_names, **kwargs)


class ISAIDObj(Voc):
    ROOT_SEV_NEW = '//ses-data//JD//ISAIDObj'
    ROOT_SEV_NEW1 = '//ses-data//JD//ISAIDObj1'
    ROOT_SEV_NEW2 = '//ses-data//JD//ISAIDObj2'
    ROOT_SEV_NEW3 = '//ses-data//JD//ISAIDObj3'
    ROOT_DES = 'D://Datasets//ISAIDObj//'

    CLS_NAMES = ISAID.CLS_NAMES
    CLS_NAMES1 = ('Small_Vehicle',)
    CLS_NAMES2 = ('Large_Vehicle', 'ship')
    CLS_NAMES3 = ('storage_tank', 'plane', 'Swimming_pool', 'Harbor', 'tennis_court',
                  'Ground_Track_Field', 'Soccer_ball_field', 'baseball_diamond',
                  'Bridge', 'basketball_court', 'Roundabout', 'Helicopter')
    DYAM_MASK = [1, 0, 1, 0, 1,
                 1, 1, 1,
                 0, 1, 1, 0]

    IMG_FOLDER = 'Patches'
    SET_FOLDER = 'ImageSets/Patch'
    COLORS = Voc.COLORS

    @staticmethod
    def SEV_NEW(**kwargs):
        return ISAIDObj(root=ISAIDObj.ROOT_SEV_NEW, **kwargs)

    @staticmethod
    def SEV_NEW1(**kwargs):
        return ISAIDObj(root=ISAIDObj.ROOT_SEV_NEW1, cls_names=ISAIDObj.CLS_NAMES1, **kwargs)

    @staticmethod
    def SEV_NEW2(**kwargs):
        return ISAIDObj(root=ISAIDObj.ROOT_SEV_NEW2, cls_names=ISAIDObj.CLS_NAMES2, **kwargs)

    @staticmethod
    def SEV_NEW3(**kwargs):
        return ISAIDObj(root=ISAIDObj.ROOT_SEV_NEW3, cls_names=ISAIDObj.CLS_NAMES3, **kwargs)

    @staticmethod
    def DES(**kwargs):
        return ISAIDObj(root=ISAIDObj.ROOT_DES, **kwargs)

    def __init__(self, root, cls_names=CLS_NAMES, colors=COLORS,
                 task_type=TASK_TYPE.INSTANCE, img_folder=IMG_FOLDER, anno_folder=Voc.ANNO_FOLDER,
                 mask_folder=Voc.MASK_FOLDER, inst_folder=Voc.INST_FOLDER, set_folder=SET_FOLDER, **kwargs):
        super().__init__(root, cls_names, colors, task_type, mask_folder, inst_folder, set_folder,
                         img_folder, anno_folder, **kwargs)

    def dataset(self, set_name='train', task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = ISAIDObj.BUILDER_MAPPER[task_type]
        kwargs_update = dict(img_folder=self.img_folder, cls_names=self.cls_names,
                             anno_folder=self.anno_folder, set_folder=self.set_folder, set_name=set_name,
                             mask_folder=self.mask_folder, inst_folder=self.inst_folder, colors=self.colors)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = builder(root=self.root, fmt=set_name + '_%d' if set_name is not None else 'all_%d', **kwargs_update)
        return dataset


class ISAIDPart(CoCo):
    ROOT_SEV_NEW = '//home//data-storage//ISAIDPart'
    ROOT_DES = 'D://Datasets//ISAIDPart//'

    CLS_NAMES = ISAID.CLS_NAMES
    CIND2NAME_REMAPPER = ISAID.CIND2NAME_REMAPPER

    IMG_PREFIX = ISAID.IMG_PREFIX
    JSON_PREFIX = ISAID.JSON_PREFIX
    JSON_FOLDER = ISAID.JSON_FOLDER

    @staticmethod
    def SEV_NEW(**kwargs):
        return ISAIDPart(root=ISAIDPart.ROOT_SEV_NEW, **kwargs)

    @staticmethod
    def DES(**kwargs):
        return ISAIDPart(root=ISAIDPart.ROOT_DES, **kwargs)

    def __init__(self, root, json_prefix=JSON_PREFIX, img_prefix=IMG_PREFIX, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES, cind2name_remapper=CIND2NAME_REMAPPER,
                 set_names=None, **kwargs):
        super().__init__(root, json_prefix=json_prefix, img_prefix=img_prefix, json_folder=json_folder,
                         cind2name_remapper=None, task_type=task_type, cls_names=cls_names,
                         set_names=set_names, **kwargs)


if __name__ == '__main__':
    ds_voc = ISAIDObj.SEV_NEW()
