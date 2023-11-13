import hashlib
import random
import threading
from collections import OrderedDict

from torch.utils.data import DataLoader, ConcatDataset

from utils.aug import *
from utils.counter import *
from utils.deploy import *
from utils.deploy import ONNXExportable
from utils.loadsd import load_fmt, refine_chans, MATCH_TYPE
from utils.pack import pack_module, PACK
from utils.ropr import *


# <editor-fold desc='数据'>
class TASK_TYPE:
    CLASSIFICATION = 'classification'
    DETECTION = 'detection'
    SEGMENTATION = 'segmentation'
    INSTANCE = 'instance'
    NONE = None


class DataSource(metaclass=ABCMeta):
    def __init__(self, root, set_names=('train', 'test'), task_type=TASK_TYPE.NONE):
        self.root = root
        self.set_names = set_names
        self.task_type = task_type

    def loader(self, set_name, batch_size=8, pin_memory=False, num_workers=0,
               aug_seq=None, shuffle=True, distribute=False, drop_last=True, **kwargs):
        if isinstance(set_name, DataSet):
            dataset = set_name
        elif isinstance(set_name, str):
            dataset = self.dataset(set_name=set_name, **kwargs)
        elif isinstance(set_name, Iterable):
            dataset = ConcatDataset([self.dataset(set_name=sub_name, **kwargs) for sub_name in set_name])
        else:
            raise Exception('err setname ' + str(set_name))
        loader = Loader(
            dataset,
            shuffle=shuffle,
            aug_seq=aug_seq,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            distribute=distribute,
            drop_last=drop_last,
        )
        return loader

    @abstractmethod
    def dataset(self, set_name, **kwargs):
        pass

    def __repr__(self):
        return '[' + self.root + ']' + str(self.set_names)


class DataSet(torch.utils.data.Dataset):

    @abstractmethod
    def name2cind(self, name):
        pass

    @abstractmethod
    def cind2name(self, cls):
        pass

    @property
    @abstractmethod
    def num_cls(self):
        pass


class NameMapper():
    def __init__(self, cls_names):
        self.cls_names = list(cls_names)

    def name2cind(self, name):
        if name not in self.cls_names:
            print('Auto add class [ ' + name + ' ]')
            self.cls_names.append(name)
        return self.cls_names.index(name)

    def cind2name(self, cind):
        num_cls = len(self.cls_names)
        if cind >= num_cls:
            expect_len = cind + 1
            print('Index out bound %d' % num_cls + ' -> ' + '%d' % expect_len)
            self.cls_names += ['C' + str(i) for i in range(num_cls, expect_len)]
        return self.cls_names[cind]

    @property
    def num_cls(self):
        return len(self.cls_names)


class ColorMapper(NameMapper):
    def __init__(self, cls_names, colors):
        super(ColorMapper, self).__init__(cls_names)
        self.colors = colors

    def col2name(self, color):
        return self.cls_names[self.colors.index(color)]

    def name2col(self, name):
        return self.colors[self.cls_names.index(name)]


# if __name__ == '__main__':
#     a = Image.new(mode='1', size=(1000, 1000))
#     save(a, file_pth='./test', extend='pkl')
#     b = get_size(a)

class PRE_TYPE:
    NONE = None
    SYS_ONLY = 'sys'
    MEM_ONLY = 'mem'
    SYS_MEM = 'sysmem'


# 数据管理与预加载
class PreLoader(metaclass=ABCMeta):
    IMG_BUFFER = {}
    EXTEND_CACHE = 'pkl'

    @staticmethod
    def get_cfile_name(fmt, index):
        if index < 0:
            cfile_name = fmt.replace('%d', 'all')
        else:
            cfile_name = fmt % index
        cfile_name = ensure_extend(cfile_name, PreLoader.EXTEND_CACHE)
        return cfile_name

    def __init__(self, pre_type=None, pre_aug_seq=None, fmt='%d', cache_dir='', clear_cache=False,
                 update_prop=0, share_img=False, **kwargs):

        '''
        初始化函数
        :param pre_type: 预加载类型选择
        :param pre_aug_seq: 预计加载增广方法
        :param fmt: 缓存文件命名形式
        :param cache_dir: 缓存文件夹
        :param clear_cache: 是否清理缓存文件夹
        :param update_prop: 更新增广概率
        :param share_img: 是否启用加载图像共享
        :param kwargs: 其它参数
        '''
        self.pre_type = pre_type
        self.pre_aug_seq = pre_aug_seq
        self.cache_dir = cache_dir
        self.update_prop = update_prop
        self.fmt = fmt
        self.share_img = share_img
        self.clear_cache = clear_cache
        if self.pre_type == PRE_TYPE.NONE:
            pass
        elif self.pre_type == PRE_TYPE.MEM_ONLY:
            self.pre_data = self._init_mem()
        elif self.pre_type == PRE_TYPE.SYS_ONLY:
            assert isinstance(self.cache_dir, str) and len(self.cache_dir) > 0, 'cache_dir err ' + self.cache_dir
            ensure_folder_pth(self.cache_dir)
            self.buffer = {}
            if clear_cache:
                print('Cover cached data at ' + self.cache_dir + ' if exist')
            else:
                print('Try pre-load data to ' + self.cache_dir + ' if not exist')
            self._init_sys(cover_cache=clear_cache)
        elif self.pre_type == PRE_TYPE.SYS_MEM:
            assert isinstance(self.cache_dir, str) and len(self.cache_dir) > 0, 'cache_dir err ' + self.cache_dir
            ensure_folder_pth(self.cache_dir)
            data_pth = os.path.join(self.cache_dir, PreLoader.get_cfile_name(self.fmt, index=-1))
            if not clear_cache and os.path.exists(data_pth):
                print('Use cached data ' + data_pth)
                self.pre_data = load_pkl(data_pth)
            else:
                self.pre_data = self._init_mem()
                print('Save cached data ' + data_pth)
                save_pkl(obj=self.pre_data, file_pth=data_pth)
        else:
            raise Exception('fmt err ' + pre_type.__class__.__name__)

    # 清理缓存
    def clear_cache(self):
        if isinstance(self.cache_dir, str) and len(self.cache_dir) > 0 and os.path.exists(self.cache_dir):
            print('Clear cache ' + self.cache_dir)

            os.removedirs(self.cache_dir)
        return True

    # 获取带增广的数据
    def _getitem_with_aug(self, index):
        img, label = self._getitem_protype(index)
        if self.pre_aug_seq is not None:
            imgs, labels = self.pre_aug_seq([img], [label])
            img = img2imgP(imgs[0])  # PIL占用空间最小
            label = labels[0]
        return img, label

    # <editor-fold desc='文件缓存分支'>
    # 将所有数据缓存到文件系统
    def _init_sys(self, cover_cache=False):
        total_num = self.__len__()
        time_start = time.time()
        sizes = []
        interval = MEnumerate.calc_interval(total_num, 20)
        for index in range(total_num):
            data_pth = os.path.join(self.cache_dir, PreLoader.get_cfile_name(self.fmt, index=index))
            if not cover_cache and os.path.exists(data_pth):
                continue
            cur_data = self._getitem_with_aug(index)
            save_pkl(obj=cur_data, file_pth=data_pth)
            if index % interval == 0 or index + 1 == total_num:
                cur_size = get_size(cur_data)
                sizes.append(cur_size)
                est_size = np.mean(sizes) * total_num
                print('Pre-loading %5d /' % (index + 1) + '%-5d' % total_num +
                      ' | Memory cur %9.3f MB' % (cur_size / NUMMB) + ' total %9.3f MB' % (est_size / NUMMB) +
                      ' | ETA ' + sec2hour_min_sec(calc_eta(index + 1, total=total_num, time_start=time_start)))
        return self

    # 更新文件缓存位置
    def _update_sys(self, index):
        data_pth = os.path.join(self.cache_dir, PreLoader.get_cfile_name(self.fmt, index=index))
        img, label = self._getitem_with_aug(index)
        save_pkl(obj=(img, label), file_pth=data_pth)
        if index in self.buffer.keys():
            del self.buffer[index]

    def _fetch_sys(self, index):
        data_pth = os.path.join(self.cache_dir, PreLoader.get_cfile_name(self.fmt, index=index))
        assert os.path.exists(data_pth), 'file lost'
        if index in self.buffer.keys():
            return self.buffer[index]
        data = load_pkl(file_pth=data_pth)
        if self.update_prop > 0 and random.random() < self.update_prop:
            self.buffer[index] = data
            threading.Thread(target=self._update_sys, args=(index,), daemon=True).start()
        return data

    def depack(self):
        datas_pth = os.path.join(self.cache_dir, PreLoader.get_cfile_name(self.fmt, index=-1))
        pre_datas = load_pkl(datas_pth)
        for index, data in enumerate(pre_datas):
            data_pth = os.path.join(self.cache_dir, PreLoader.get_cfile_name(self.fmt, index=index))
            save_pkl(obj=data, file_pth=data_pth)
        return self

    # </editor-fold>

    # <editor-fold desc='内存缓存分支'>

    # 将所有数据加载到内存
    def _init_mem(self):
        print('Pre-load data to memory')
        pre_data = []
        total_num = self.__len__()
        time_start = time.time()
        sizes = []
        interval = MEnumerate.calc_interval(total_num, 20)
        for index in range(total_num):
            cur_data = self._getitem_with_aug(index)
            pre_data.append(cur_data)
            if index % interval == 0 or index + 1 == total_num:
                cur_size = get_size(cur_data)
                sizes.append(cur_size)
                est_size = np.mean(sizes) * total_num
                print('Pre-loading %5d /' % (index + 1) + '%-5d' % total_num +
                      ' | Memory cur %9.3f MB' % (cur_size / NUMMB) + ' total %9.3f MB' % (est_size / NUMMB) +
                      ' | ETA ' + sec2hour_min_sec(calc_eta(index + 1, total=total_num, time_start=time_start)))

        return pre_data

    # 更新内存缓存位置
    def _update_mem(self, index):
        self.pre_data[index] = self._getitem_with_aug(index)

    def _fetch_mem(self, index):
        data = self.pre_data[index]
        if self.update_prop > 0 and random.random() < self.update_prop:
            threading.Thread(target=self._update_mem, args=(index,), daemon=True).start()
        return data

    # </editor-fold>

    # 具备缓存共享功能的图像加载
    def load_img(self, img_pth):
        if not self.share_img:
            return Image.open(img_pth).convert("RGB")
        if img_pth in PreLoader.IMG_BUFFER.keys():
            return PreLoader.IMG_BUFFER[img_pth]
        else:
            img = Image.open(img_pth).convert("RGB")
            PreLoader.IMG_BUFFER[img_pth] = img
        return img

    # 数据出口
    def __getitem__(self, index):
        if isinstance(index, Iterable):
            items = [self.__getitem__(item_sub) for item_sub in index]
            return items
        if self.pre_type == PRE_TYPE.SYS_ONLY:
            return self._fetch_sys(index)
        elif self.pre_type == PRE_TYPE.MEM_ONLY or self.pre_type == PRE_TYPE.SYS_MEM:
            return self._fetch_mem(index)
        else:
            return self._getitem_protype(index)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def _getitem_protype(self, index):
        pass


class Loader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.Dataset, shuffle: bool = False, num_workers: int = 0,
                 batch_size: int = 1, pin_memory: bool = False, distribute: bool = False,
                 aug_seq: AugSeq = None, drop_last=True, **kwargs):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distribute else None
        shuffle = shuffle and not distribute
        self.dataset = dataset
        super(Loader, self).__init__(dataset=dataset, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                                     batch_size=batch_size, pin_memory=pin_memory, drop_last=drop_last, **kwargs)
        self.aug_seq = aug_seq
        dataset_prop = dataset.datasets[0] if isinstance(dataset, ConcatDataset) else dataset

        self.num_cls = dataset_prop.num_cls if hasattr(dataset_prop, 'num_cls') else 1
        self.name2cind = dataset_prop.name2cind if hasattr(dataset_prop, 'name2cind') else lambda x: 0
        self.cind2name = dataset_prop.cind2name if hasattr(dataset_prop, 'cind2name') else lambda x: 'unknown'
        self.collate_fn = partial(Loader.collate_fn, aug_seq=self.aug_seq)

    @property
    def img_size(self):
        return self.aug_seq.img_size if self.aug_seq is not None else None

    @img_size.setter
    def img_size(self, img_size):
        if self.aug_seq is not None:
            self.aug_seq.img_size = img_size
            self.collate_fn = partial(Loader.collate_fn, aug_seq=self.aug_seq)

    @staticmethod
    def collate_fn(batch, aug_seq):
        imgs = []
        labels = []
        for img, ann in batch:
            labels.append(ann)
            imgs.append(img)
        if aug_seq is not None:
            imgs, labels = aug_seq(imgs, labels)
        return imgs, labels

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.__getitem__([item])
        elif isinstance(item, list) or isinstance(item, tuple):
            batch = [self.dataset.__getitem__(index) for index in item]
            imgs, labels = self.collate_fn(batch)
            return imgs, labels
        else:
            raise Exception('index err')


# </editor-fold>

# <editor-fold desc='模型'>
# </editor-fold>
class ImageRecognizable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def img_size(self):
        pass

    @property
    @abstractmethod
    def num_cls(self):
        pass


class IndependentInferable(ImageRecognizable):

    @abstractmethod
    def imgs2labels(self, imgs, cind2name=None, **kwargs):
        pass

    def loader2labels(self, loader, **kwargs):
        print('Start Annotation')
        labels_anno = []
        for i, (imgs, labels) in MEnumerate(loader, prefix='Annotating ', interval=20, broadcast=print):
            labels_md = self.imgs2labels(imgs=imgs, cind2name=loader.cind2name, **kwargs)
            for j, (label_md, label) in enumerate(zip(labels_md, labels)):
                assert isinstance(label_md, ImageLabel), 'fmt err ' + label_md.__class__.__name__
                assert isinstance(label, ImageLabel), 'fmt err ' + label.__class__.__name__
                label_md.meta = label.meta
                label_md.init_size = label.init_size
                label_md.ctx_border = label.ctx_border
                labels_anno.append(label_md)
        return labels_anno


class SurpervisedInferable(ImageRecognizable):

    @abstractmethod
    def imgs_labels2labels(self, imgs, labels, cind2name=None, **kwargs):
        pass

    def loader2labels_with_surp(self, loader, **kwargs):
        print('Start Annotation')
        labels_anno = []
        for i, (imgs, labels) in MEnumerate(loader, prefix='Annotating ', interval=20, broadcast=print):
            labels_md = self.imgs_labels2labels(imgs=imgs, labels=labels, cind2name=loader.cind2name, **kwargs)
            for j, (label_md, label) in enumerate(zip(labels_md, labels)):
                assert isinstance(label_md, ImageLabel), 'fmt err ' + label_md.__class__.__name__
                assert isinstance(label, ImageLabel), 'fmt err ' + label.__class__.__name__
                label_md.meta = label.meta
                label_md.init_size = label.init_size
                label_md.ctx_border = label.ctx_border
                labels_anno.append(label_md)
        return labels_anno


class OneStageTrainable(metaclass=ABCMeta):

    @abstractmethod
    def labels2tars(self, labels, **kwargs):
        pass

    @abstractmethod
    def imgs_tars2loss(self, imgs, targets, **kwargs):
        pass

    @property
    @abstractmethod
    def img_size(self):
        pass

    @img_size.setter
    @abstractmethod
    def img_size(self, img_size):
        pass


class TorchModel(nn.Module, OneStageTrainable):
    def __init__(self, device=None, pack=PACK.AUTO, **modules):
        super(TorchModel, self).__init__()
        super(nn.Module, self).__init__()
        for name, module in modules.items():
            self.__setattr__(name, module)
        self.repack(device=device, pack=pack)

    def repack(self, device=None, pack=PACK.AUTO):
        device_ids = select_device(device, min_thres=0.01, one_thres=0.5)
        self.device_ids = device_ids
        device = torch.device('cpu' if device_ids[0] is None else 'cuda:' + str(device_ids[0]))
        self.device = device
        self.pkd_modules = {}
        for name, module in self.named_children():
            module.to(device)
            self.pkd_modules[name] = pack_module(module, device_ids, pack=pack, module_name=name)
        self._pack = pack
        return self

    def add_module(self, **modules):
        for name, module in modules.items():
            self.__setattr__(name, module)
            module.to(self.device)
            self.pkd_modules[name] = pack_module(module, self.device_ids, pack=self.pack, module_name=name)
        return self

    @property
    def pack(self):
        return self._pack

    # 保存权重
    def save(self, file_name):
        file_name = file_name if str.endswith(file_name, '.pth') else file_name + '.pth'
        torch.save(self.state_dict(), file_name)
        return None

    # 读取权重
    def load(self, file_name, transfer=False, show_detial=False, power=1.0):
        if isinstance(file_name, str):
            file_name = file_name if str.endswith(file_name, '.pth') else file_name + '.pth'
            print('Load weight ' + file_name)
            sd = torch.load(file_name, map_location=self.device)
        elif isinstance(file_name, OrderedDict):
            sd = file_name
        else:
            raise Exception('dict err')
        if load_fmt(self, sd_ori=sd, match_type=MATCH_TYPE.SIZE | MATCH_TYPE.FULL_NAME, only_fullmatch=True,
                    show_detial=show_detial, power=power):
            refine_chans(self)
            self.to(self.device)
            return None
        print('Struct changed, Try to match by others')
        if load_fmt(self, sd_ori=sd, match_type=MATCH_TYPE.SIZE | MATCH_TYPE.LAST_NAME, only_fullmatch=not transfer,
                    show_detial=show_detial, power=power):
            refine_chans(self)
            self.to(self.device)
            return None
        print('Tolerates imperfect matches')
        load_fmt(self, sd_ori=sd, match_type=MATCH_TYPE.SIZE | MATCH_TYPE.FULL_NAME, only_fullmatch=False,
                 show_detial=show_detial, power=power)
        self.to(self.device)
        return None

    def train(self, mode: bool = True):
        super().train(mode)
        for name, module in self.pkd_modules.items():
            module.train(mode)
        return self

    # @abstractmethod
    def export_onnx(self, onnx_pth, **kwargs):
        for name, module in self.named_children():
            if isinstance(module, ONNXExportable):
                module.export(onnx_pth.split('.')[0] + '_' + name)

        return self

    @abstractmethod
    def export_onnx_trt(self, **kwargs):
        raise NotImplementedError()

    # 快速转化loss
    # def imgs_labels2loss(self, imgs, labels, show=False):
    #     target = self.labels2tars(labels)
    #     loss = self.imgs_tars2loss(imgs, target)
    #     loss, losses, names = TorchModel.process_loss(loss)
    #     if show:
    #         print(''.join([n + ' %-10.5f  ' % l for l, n in zip(losses, names)]))
    #     return loss

    def __bytes__(self):
        hasher = hashlib.sha1()
        for para in self.parameters():
            byte = para.byte().view(-1).detach().cpu().numpy()
            hasher.update(byte)
        return hasher.digest()


class OneStageTorchModel(TorchModel):
    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super(OneStageTorchModel, self).__init__(backbone=backbone, device=device, pack=pack)

    def export_onnx(self, onnx_pth, batch_size=1):
        W, H = self.img_size
        model2onnx(self.backbone, onnx_pth, input_size=(batch_size, 3, H, W))
        return True

    def export_onnx_trt(self, onnx_pth, trt_pth, batch_size=1):
        # W, H = self.norm
        # from deploy.onnx import model2onnx
        # from deploy.trt import onnx2trt
        # model2onnx(self.backbone, onnx_pth, input_size=(batch_size, 3, H, W))
        # onnx2trt(onnx_pth=onnx_pth, trt_pth=trt_pth, max_batch=4, min_batch=1, std_batch=2)
        return True
