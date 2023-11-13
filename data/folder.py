import os
import shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from utils.frame import NameMapper, PreLoader, DataSet
from PIL import Image
from collections import Counter
from utils.label import IndexCategory, CategoryLabel, img2imgP
from utils.file import *

# <editor-fold desc='folder编辑'>
IMAGE_APPENDEIX = ['jpg', 'JPEG', 'png']


def resample_by_names(cls_names, resample):
    presv_inds = []
    for i, cls_name in enumerate(cls_names):
        if not cls_name in resample.keys():
            presv_inds.append(i)
            continue
        resamp_num = resample[cls_name]
        low = np.floor(resamp_num)
        high = np.ceil(resamp_num)
        resamp_num_rand = np.random.uniform(low=low, high=high)
        resamp_num = int(low if resamp_num_rand > resamp_num else high)
        for j in range(resamp_num):
            presv_inds.append(i)
    return presv_inds


def get_pths(set_dir):
    img_pths = []
    names = []
    for cls_name in os.listdir(set_dir):
        cls_dir = os.path.join(set_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for img_dir, _, img_names in os.walk(cls_dir):
            for img_name in img_names:
                if img_name.split('.')[1] not in IMAGE_APPENDEIX:
                    continue
                img_pths.append(os.path.join(cls_dir, img_dir, img_name))
                names.append(cls_name)
    return img_pths, names


def get_cls_names(set_dir):
    cls_names = [cls_name for cls_name in os.listdir(set_dir)
                 if os.path.isdir(os.path.join(set_dir, cls_name))]
    return cls_names


# </editor-fold>

class PKLDataset(NameMapper, PreLoader, DataSet):
    EXTEND = 'pkl'

    def __len__(self):
        return len(self.img_pths)

    def _getitem_protype(self, index):
        img_pth = self.img_pths[index]
        img = self.load_img(img_pth)
        label = self.labels[index] if self.labels is not None else None
        return img, label

    def __init__(self, root, cls_names, img_folder='images', pkl_name='label', **kwargs):
        self.root = root
        self.img_folder = img_folder
        self.pkl_name = pkl_name
        self.labels = load_pkl(self.pkl_pth) if os.path.exists(self.pkl_pth) else None
        super(PKLDataset, self).__init__(cls_names)
        super(NameMapper, self).__init__(**kwargs)

    def delete(self):
        shutil.rmtree(self.img_dir)
        os.remove(self.pkl_pth)
        print('Delete complete at ' + self.root + ' < ' + self.img_folder + ' , ' + self.pkl_name + ' > ')
        return self

    @property
    def img_folder(self):
        return self._img_folder

    @img_folder.setter
    def img_folder(self, img_folder):
        self._img_folder = img_folder
        self.img_dir = os.path.join(self.root, img_folder)
        img_names = sorted(os.listdir(self.img_dir))
        self.img_pths = [os.path.join(self.img_dir, img_name) for img_name in img_names]

    @property
    def pkl_name(self):
        return self._pkl_name

    @pkl_name.setter
    def pkl_name(self, pkl_name):
        self._pkl_name = pkl_name
        self.pkl_pth = os.path.join(self.root, ensure_extend(pkl_name, PKLDataset.EXTEND))

    @staticmethod
    def create(imgs, labels, root, img_folder='images', pkl_name='label', img_extend='jpg'):
        print('Create dataset at ' + root + ' < ' + img_folder + ' , ' + pkl_name + ' > ')
        img_dir = os.path.join(root, img_folder)
        ensure_folder_pth(img_dir)
        pkl_pth = os.path.join(root, ensure_extend(pkl_name, PKLDataset.EXTEND))
        ensure_file_dir(pkl_pth)
        for i, img in MEnumerate(imgs, prefix='Writing ', with_eta=True):
            meta = labels[i].meta
            img_pth = os.path.join(img_dir, ensure_extend(meta, img_extend))
            imgP = img2imgP(img)
            imgP.save(img_pth)
        labels.sort(key=lambda x: x.meta)
        save_pkl(pkl_pth, labels)
        print('Create complete')
        return True


class FolderDataset(NameMapper, PreLoader, DataSet):
    def __init__(self, root, cls_names=None, resample=None, pre_aug_seq=None, **kwargs):
        cls_names = get_cls_names(root) if cls_names is None else cls_names
        super(FolderDataset, self).__init__(cls_names)
        self.root = root
        self.img_pths, self.names = get_pths(root)
        if resample is not None:
            presv_inds = resample_by_names(self.names, resample=resample)
            self.img_pths = [self.img_pths[ind] for ind in presv_inds]
            self.names = [self.names[ind] for ind in presv_inds]

        super(NameMapper, self).__init__(pre_aug_seq=pre_aug_seq)

    def __len__(self):
        return len(self.img_pths)

    def _getitem_protype(self, index):
        img_pth, name = self.img_pths[index], self.names[index]
        meta = os.path.split(os.path.basename(img_pth))[0]
        img = Image.open(img_pth)
        label = CategoryLabel(
            category=IndexCategory(cindN=int(self.name2cind(name)), conf=1, num_cls=self.num_cls),
            img_size=img.size, meta=meta, name=name)
        return img, label

    def append(self, dir_dict):
        for name, dir in dir_dict.items():
            img_names = os.listdir(dir)
            for img_name in img_names:
                img_pth = os.path.join(dir, img_name)
                self.img_pths.append(img_pth)
                self.names.append(name)
        return True

    def __repr__(self):
        num_dict = Counter(self.names)
        msg = '\n'.join(['%10s ' % name + ' %5d' % num for name, num in num_dict.items()])
        return msg

    def delete(self):
        print('Delete dataset ' + self.root)
        for pth in self.img_pths:
            if os.path.isfile(pth):
                os.remove(pth)
        print('Deleting complete')
        return None


class FolderMapDataset(FolderDataset):
    def __init__(self, root, name_mapper, resample=None, pre_aug_seq=None):
        super(FolderMapDataset, self).__init__(root=root, cls_names=None, resample=None, pre_aug_seq=None)
        self.names = [name_mapper[name] for name in self.names]
        self.cls_names = [name_mapper[name] for name in self.cls_names]
        if resample is not None:
            presv_inds = resample_by_names(self.names, resample=resample)
            self.img_pths = [self.img_pths[ind] for ind in presv_inds]
            self.names = [self.names[ind] for ind in presv_inds]
        super(NameMapper, self).__init__(pre_aug_seq=pre_aug_seq)
