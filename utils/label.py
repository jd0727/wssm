import copy
from abc import ABCMeta, abstractmethod
from typing import Iterable, Sized

from .cvting import *


# <editor-fold desc='原型接口'>
# 可移动标签 自定义增广方法接口
class Movable(metaclass=ABCMeta):
    @abstractmethod
    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        pass

    @abstractmethod
    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        pass

    @abstractmethod
    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    @size.setter
    @abstractmethod
    def size(self, size):
        pass


class PntExtractable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def num_pnt(self) -> int:
        pass

    @abstractmethod
    def extract_xlylN(self):
        pass

    @abstractmethod
    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, **kwargs):
        pass


class BoolMaskExtractable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def num_bool_chan(self) -> int:
        pass

    @abstractmethod
    def extract_maskNb(self):
        pass

    @abstractmethod
    def refrom_maskNb(self, maskNb, **kwargs):
        pass

    def extract_maskNb_enc(self, index):
        maskNb = self.extract_maskNb()
        inds = np.arange(index, index + self.num_bool_chan)
        return np.max(maskNb * inds, axis=2, keepdims=True)

    def refrom_maskNb_enc(self, maskNb_enc, index, **kwargs):
        inds = np.arange(index, index + self.num_bool_chan)
        maskNb = maskNb_enc == inds
        self.refrom_maskNb(maskNb, **kwargs)
        return self


class ValMaskExtractable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def num_chan(self) -> int:
        pass

    @abstractmethod
    def extract_maskN(self):
        pass

    @abstractmethod
    def refrom_maskN(self, maskN, **kwargs):
        pass


# 作为图像标签可被其余增广包扩展
class Augurable(BoolMaskExtractable, PntExtractable, ValMaskExtractable):

    @property
    @abstractmethod
    def img_size(self):
        pass

    @img_size.setter
    @abstractmethod
    def img_size(self, img_size):
        pass


class Border(Movable):

    @abstractmethod
    def align_with(self, ixysN: np.ndarray):
        pass

    @abstractmethod
    def expend(self, ratio: np.ndarray = np.ones(shape=2)):
        pass

    @staticmethod
    def convert(border):
        if isinstance(border, Border):
            return border
        elif isinstance(border, Iterable) and isinstance(border, Sized):
            if len(border) == 4:
                return XYXYBorder.convert(border)
            elif len(border) == 5:
                return XYWHABorder.convert(border)
            else:
                return XLYLBorder.convert(border)
        else:
            raise Exception('err fmt' + str(border.__class__.__name__))

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def measure(self):
        pass

    @abstractmethod
    def area(self):
        pass


class BoolRegion(Movable):

    @property
    @abstractmethod
    def maskNb(self) -> np.ndarray:
        pass

    @property
    def ixysN(self) -> np.ndarray:
        iys, ixs = np.nonzero(self.maskNb)
        ixys = np.stack([ixs, iys], axis=1)
        return ixys

    @staticmethod
    def convert(mask):
        if isinstance(mask, BoolRegion):
            return mask
        elif isinstance(mask, np.ndarray) or isinstance(mask, PIL.Image.Image) or isinstance(mask, torch.Tensor):
            return AbsBoolRegion(mask)
        else:
            raise Exception('err fmt ' + str(mask.__class__.__name__))

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def measure(self):
        pass

    @abstractmethod
    def area(self):
        pass


# </editor-fold>


# <editor-fold desc='几何元素'>
class XYXYBorder(Border, BoolRegion, PntExtractable):
    def align_with(self, ixysN: np.ndarray):
        self.xyxyN = ixysN2xyxyN(ixysN)
        return self

    @property
    def ixysN(self) -> np.ndarray:
        return xyxyN2ixysN(self.xyxyN, self._size)

    def measure(self):
        return np.sqrt(np.prod(self.xyxyN[2:] - self.xyxyN[:2]))

    def area(self):
        return np.prod(self.xyxyN[2:] - self.xyxyN[:2])

    WIDTH = 4
    __slots__ = ('xyxyN', '_size')

    def __init__(self, xyxyN, size):
        self.xyxyN = np.array(xyxyN).astype(np.float32)
        self.size = size

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.xyxyN = xyxyN_clip(self.xyxyN, xyxyN_rgn=xyxyN_rgn)
        return self

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        xlyl = xyxyN2xlylN(self.xyxyN)
        xlyl = xlyl * scale + bias
        self.xyxyN = xlylN2xyxyN(xlyl)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        self.xyxyN = xyxyN_perspective(self.xyxyN, H=H)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def expend(self, ratio: np.ndarray = np.ones(shape=2)):
        xywh = xyxyN2xywhN(self.xyxyN)
        xywh[2:4] *= ratio
        self.xyxyN = xywhN2xyxyN(xywh)
        return self

    @staticmethod
    def convert(border):
        if isinstance(border, XYXYBorder):
            return XYXYBorder(border.xyxyN, border.size)
        elif isinstance(border, Iterable):
            border = np.array(border)
            size = tuple(border[2:4].astype(np.int32))
            return XYXYBorder(border, size=size)
        elif isinstance(border, XYWHBorder):
            return XYXYBorder(xywhN2xyxyN(border.xywhN), border.size)
        elif isinstance(border, XYWHABorder):
            xyxy = xywhaN2xyxyN(border.xywhaN)
            xyxy = xyxyN_clip(xyxy, xyxyN_rgn=np.array([0, 0, border.size[0], border.size[1]]))
            return XYXYBorder(xyxy, border.size)
        elif isinstance(border, XLYLBorder):
            return XYXYBorder(xlylN2xyxyN(border.xlylN), border.size)
        else:
            raise Exception('err fmt' + str(border.__class__.__name__))

    def __repr__(self):
        return 'xyxyN' + str(self.xyxyN)

    @property
    def num_pnt(self):
        return 4

    def extract_xlylN(self):
        return xyxyN2xlylN(self.xyxyN)

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, **kwargs):
        self.xyxyN = xlylN2xyxyN(xlylN)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @property
    def maskNb(self):
        maskN = xyxyN2maskNb(self.xyxyN, self._size)
        return maskN


class XYWHBorder(Border, BoolRegion, PntExtractable):
    WIDTH = 4
    __slots__ = ('xywhN', '_size')

    def __init__(self, xywhN, size):
        self.xywhN = np.array(xywhN).astype(np.float32)
        self.size = size

    def align_with(self, ixysN: np.ndarray):
        self.xywhN = ixysN2xywhN(ixysN)
        return self

    @property
    def ixysN(self) -> np.ndarray:
        return xywhN2ixysN(self.xywhN, self._size)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    def measure(self):
        return np.sqrt(np.prod(self.xywhN[2:]))

    def area(self):
        return np.prod(self.xywhN[2:])

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.xywhN = xywhN_clip(self.xywhN, xyxyN_rgn=xyxyN_rgn)
        return self

    def __repr__(self):
        return 'xywhN' + str(self.xywhN)

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        xlyl = xywhN2xlylN(self.xywhN)
        xlyl = xlyl * scale + bias
        self.xywhN = xlylN2xywhN(xlyl)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        self.xywhN = xywhN_perspective(self.xywhN, H=H)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def expend(self, ratio: np.ndarray = np.ones(shape=2)):
        self.xywhN[2:4] *= ratio
        return self

    @staticmethod
    def convert(border):
        if isinstance(border, XYWHBorder):
            return border
        elif isinstance(border, Iterable):
            border = np.array(border)
            size = tuple((border[:2] + border[2:4] / 2).astype(np.int32))
            return XYWHBorder(border, size=size)
        elif isinstance(border, XYXYBorder):
            return XYWHBorder(xyxyN2xywhN(border.xyxyN), size=border.size)
        elif isinstance(border, XYWHABorder):
            xyxy = xywhaN2xyxyN(border.xywhaN)
            xyxy = xyxyN_clip(xyxy, xyxyN_rgn=np.array([0, 0, border.size[0], border.size[1]]))
            return XYWHBorder(xyxyN2xywhN(xyxy), size=border.size)
        elif isinstance(border, XLYLBorder):
            return XYWHBorder(xlylN2xywhN(border.xlylN), size=border.size)
        else:
            raise Exception('err fmt' + str(border.__class__.__name__))

    @property
    def num_pnt(self):
        return 4

    def extract_xlylN(self):
        return xywhN2xlylN(self.xywhN)

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, **kwargs):
        self.xywhN = xlylN2xywhN(xlylN)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @property
    def maskNb(self):
        maskN = xywhN2maskNb(self.xywhN, self._size)
        return maskN


class XYWHABorder(Border, BoolRegion, PntExtractable):
    WIDTH = 5
    __slots__ = ('xywhaN', '_size')

    def __init__(self, xywhaN, size):
        self.xywhaN = np.array(xywhaN).astype(np.float32)
        self.size = size

    def align_with(self, ixysN: np.ndarray):
        self.xywhaN = ixysN2xywhaN(ixysN, alphaN=self.xywhaN[4])
        return self

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    def measure(self):
        return np.sqrt(np.prod(self.xywhaN[2:4]))

    def area(self):
        return np.prod(self.xywhaN[2:4])

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.xywhaN = xywhaN_clip(self.xywhaN, xyxyN_rgn=xyxyN_rgn)
        return self

    def __repr__(self):
        return 'xywhaN' + str(self.xywhaN)

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        xlyl = xywhaN2xlylN(self.xywhaN)
        self.xywhaN = xlylN2xywhaN(xlyl * scale + bias)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        self.xywhaN = xywhaN_perspective(self.xywhaN, H=H)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def expend(self, ratio: np.ndarray = np.ones(shape=2)):
        self.xywhaN[2:4] *= ratio
        return self

    @staticmethod
    def convert(border):
        if isinstance(border, XYWHABorder):
            return border
        elif isinstance(border, Iterable):
            xywhaN = np.array(border)
            xyxyN = xywhaN2xyxyN(xywhaN)
            return XYWHABorder(xywhaN, size=tuple(xyxyN[2:4].astype(np.int32)))
        elif isinstance(border, XYXYBorder):
            return XYWHABorder(xyxyN2xywhaN(border.xyxyN), size=border.size)
        elif isinstance(border, XYWHBorder):
            return XYWHABorder(xywhN2xywhaN(border.xywhN), size=border.size)
        elif isinstance(border, XLYLBorder):
            return XYWHABorder(xlylN2xywhaN(border.xlylN), size=border.size)
        else:
            raise Exception('err fmt' + str(border.__class__.__name__))

    @property
    def num_pnt(self):
        return 4

    def extract_xlylN(self):
        return xywhaN2xlylN(self.xywhaN)

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, **kwargs):
        self.xywhaN = xlylN2xywhaN(xlylN)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @property
    def maskNb(self):
        return xywhaN2maskNb(self.xywhaN, self._size)


class XLYLBorder(Border, BoolRegion, PntExtractable):
    __slots__ = ('xlylN', '_size')

    def __init__(self, xlylN, size):
        self.xlylN = np.array(xlylN).astype(np.float32)
        self.size = size

    def align_with(self, ixysN: np.ndarray):
        # TODO non impld
        return self

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    def measure(self):
        return np.sqrt(xlylN2areaN(self.xlylN))

    def area(self):
        return xlylN2areaN(self.xlylN)

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.xlylN = xlylN_clip(self.xlylN, xyxyN_rgn=xyxyN_rgn)
        return self

    def __repr__(self):
        return 'xlylN' + str(self.xlylN)

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        self.xlylN = self.xlylN * scale + bias
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        self.xlylN = xlylN_perspective(self.xlylN, H=H)
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def expend(self, ratio: np.ndarray = np.ones(shape=2)):
        xy = np.mean(self.xlylN, axis=0)
        self.xlylN = (self.xlylN - xy) * ratio + xy
        return self

    @staticmethod
    def convert(border):
        if isinstance(border, XLYLBorder):
            return border
        elif isinstance(border, Iterable):
            xlylN = np.array(border)
            size = tuple(np.max(xlylN, axis=0).astype(np.int32))
            return XLYLBorder(xlylN, size=size)
        elif isinstance(border, XYXYBorder):
            return XLYLBorder(xyxyN2xlylN(border.xyxyN), size=border.size)
        elif isinstance(border, XYWHBorder):
            return XLYLBorder(xywhN2xlylN(border.xywhN), size=border.size)
        elif isinstance(border, XYWHABorder):
            return XLYLBorder(xywhaN2xlylN(border.xywhaN), size=border.size)
        else:
            raise Exception('err fmt' + str(border.__class__.__name__))

    @property
    def num_pnt(self):
        return self.xlylN.shape[0]

    def extract_xlylN(self):
        return self.xlylN

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, **kwargs):
        self.xlylN = xlylN
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @property
    def maskNb(self):
        if self.xlylN.shape[0] <= 4:
            return xlylN2maskNb_convex(self.xlylN, self._size)
        else:
            return xlylN2maskNb(self.xlylN, self._size)


class AbsBoolRegion(BoolRegion, BoolMaskExtractable):
    __slots__ = ('maskNb_abs',)
    CONF_THRES = 0.5

    def __init__(self, maskNb_abs):
        self.maskNb_abs = np.array(maskNb_abs).astype(bool)

    @property
    def size(self):
        shape = self.maskNb_abs.shape
        return (shape[1], shape[0])

    @size.setter
    def size(self, size):
        if not tuple(size) == self.size:
            A = np.array([[1, 0, 0], [0, 1, 0]]).astype(np.float32)
            self.maskNb_abs = cv2.warpAffine(self.maskNb_abs.astype(np.float32), A, size) > 0.5

    @property
    def maskNb(self):
        return self.maskNb_abs

    @property
    def num_bool_chan(self) -> int:
        return 1

    def extract_maskNb(self):
        return self.maskNb_abs[..., None]

    def refrom_maskNb(self, maskNb, **kwargs):
        self.maskNb_abs = maskNb[..., 0]
        return self

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        if np.all(xyxyN_rgn == np.array((0, 0, self.maskNb_abs.shape[1], self.maskNb_abs.shape[0]))):
            return self
        maskNb = xyxyN2maskNb(xyxyN_rgn, size=self.size)
        self.maskNb_abs = self.maskNb_abs * maskNb
        return self

    def measure(self):
        return np.sqrt(np.sum(self.maskNb_abs))

    def area(self):
        return np.sum(self.maskNb_abs)

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2),
               resample=cv2.INTER_LANCZOS4, **kwargs):
        A = np.array([[scale[0], 0, bias[0]], [0, scale[1], bias[1]]])
        self.maskNb_abs = cv2.warpAffine(self.maskNb_abs.astype(np.float32), A, size,
                                         flags=resample) > AbsBoolRegion.CONF_THRES
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), resample=cv2.INTER_LANCZOS4, **kwargs):
        self.maskNb_abs = cv2.warpPerspective(self.maskNb_abs.astype(np.float32), H, size,
                                              flags=resample) > AbsBoolRegion.CONF_THRES
        return self

    @staticmethod
    def convert(mask):
        if isinstance(mask, AbsBoolRegion):
            return mask
        elif isinstance(mask, BoolRegion):
            return AbsBoolRegion(maskNb_abs=mask.maskNb)
        else:
            raise Exception('err fmt ' + str(mask.__class__.__name__))

    def __repr__(self):
        return 'bmsk' + str(self.size)


class AbsValRegion(BoolRegion, PntExtractable):
    __slots__ = ('maskN_abs', 'conf_thres')

    def __init__(self, maskN_abs, conf_thres=0.5):
        self.maskN_abs = np.array(maskN_abs).astype(np.float32)
        self.conf_thres = conf_thres

    @property
    def size(self):
        shape = self.maskN_abs.shape
        return (shape[1], shape[0])

    @size.setter
    def size(self, size):
        if not size == self.size:
            A = np.array([[1, 0, 0], [0, 1, 0]]).astype(np.float32)
            self.maskN_abs = cv2.warpAffine(self.maskN_abs, A, size)

    @property
    def maskN(self):
        return self.maskN_abs

    @property
    def maskNb(self):
        return self.maskN_abs > self.conf_thres

    @property
    def num_pnt(self) -> int:
        return 4

    def extract_xlylN(self):
        return xyxyN2xlylN(np.array([0, 0, self.maskN_abs.shape[1], self.maskN_abs.shape[0]])).astype(np.float32)

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, resample=cv2.INTER_LANCZOS4, **kwargs):
        xlyl_ori = self.extract_xlylN()
        H = cv2.getPerspectiveTransform(xlyl_ori, xlylN.astype(np.float32))
        self.maskN_abs = cv2.warpPerspective(self.maskN_abs, H, size, flags=resample)
        return self

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        if np.all(xyxyN_rgn == np.array([0, 0, self.maskN_abs.shape[1], self.maskN_abs.shape[0]])):
            return self
        maskNb = xyxyN2maskNb(xyxyN_rgn, size=self.size)
        self.maskN_abs = self.maskN_abs * maskNb
        return self

    def measure(self):
        return np.sqrt(np.sum(self.maskN_abs > self.conf_thres))

    def area(self):
        return np.sum(self.maskN_abs > self.conf_thres)

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2),
               resample=cv2.INTER_LANCZOS4, **kwargs):
        A = np.array([[scale[0], 0, bias[0]], [0, scale[1], bias[1]]])
        self.maskN_abs = cv2.warpAffine(self.maskN_abs, A, size, flags=resample)
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), resample=cv2.INTER_LANCZOS4, **kwargs):
        self.maskN_abs = cv2.warpPerspective(self.maskN_abs, H, size, flags=resample)
        return self

    @staticmethod
    def convert(mask):
        if isinstance(mask, AbsValRegion):
            return mask
        elif isinstance(mask, BoolRegion):
            return AbsValRegion(mask.maskNb, conf_thres=0.5)
        else:
            raise Exception('err fmt ' + str(mask.__class__.__name__))

    def __repr__(self):
        return 'amsk' + str(self.size)


class RefValRegion(BoolRegion, PntExtractable):
    __slots__ = ('maskN_ref', 'xyN', '_size', 'conf_thres')

    def __init__(self, maskN_ref, xyN, size, conf_thres=0.5):
        self.maskN_ref = np.array(maskN_ref).astype(np.float32)
        self.conf_thres = conf_thres
        self.xyN = np.array(xyN).astype(np.int32)
        self._size = size

    @property
    def ixysN(self) -> np.ndarray:
        iys, ixs = np.nonzero(self.maskN_ref > self.conf_thres)
        ixys_ref = np.stack([ixs, iys], axis=1)
        return self.xyN + ixys_ref

    @property
    def maskNb_ref(self):
        return self.maskN_ref > self.conf_thres

    @property
    def whN(self):
        return np.array((self.maskN_ref.shape[1], self.maskN_ref.shape[0]))

    def measure(self):
        return np.sqrt(np.sum(self.maskN_ref > self.conf_thres))

    def area(self):
        return np.sum(self.maskN_ref > self.conf_thres)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    @property
    def xyxyN(self):
        return np.concatenate([self.xyN, self.xyN + self.whN], axis=0).astype(np.float32)

    @property
    def maskNb(self):
        maskNb = np.zeros(shape=(self.size[1], self.size[0]), dtype=bool)
        shape_ref = self.maskN_ref.shape
        maskNb[self.xyN[1]:self.xyN[1] + shape_ref[0], self.xyN[0]:self.xyN[0] + shape_ref[1]] = \
            self.maskNb_ref[0:self.size[1] - self.xyN[1], 0:self.size[0] - self.xyN[0]]
        return maskNb

    @property
    def maskN(self):
        maskN = np.zeros(shape=(self.size[1], self.size[0]), dtype=np.float32)
        shape_ref = self.maskN_ref.shape
        maskN[self.xyN[1]:self.xyN[1] + shape_ref[0], self.xyN[0]:self.xyN[0] + shape_ref[1]] = \
            self.maskN_ref[0:self.size[1] - self.xyN[1], 0:self.size[0] - self.xyN[0]]
        return maskN

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        whN = self.whN
        xyN_max = self.xyN + whN
        if np.any(self.xyN < xyxyN_rgn[:2]) or np.any(xyN_max > xyxyN_rgn[2:4]):
            if np.any(xyN_max <= xyxyN_rgn[:2]) or np.any(self.xyN >= xyxyN_rgn[2:4]):
                self.maskN_ref = np.zeros(shape=(0, 0))
                return self
            xyxyN_rgn_ref = xyxyN_rgn - self.xyN[[0, 1, 0, 1]]
            xyxyN_rgn_ref = xyxyN_clip(xyxyN_rgn_ref, np.array([0, 0, whN[0], whN[1]]))
            self.maskN_ref = self.maskN_ref[xyxyN_rgn_ref[1]:xyxyN_rgn_ref[3], xyxyN_rgn_ref[0]:xyxyN_rgn_ref[2]]
            self.xyN = np.maximum(self.xyN, xyxyN_rgn[:2])
        return self

    @property
    def num_pnt(self) -> int:
        return 4

    def extract_xlylN(self):
        return xyxyN2xlylN(self.xyxyN)

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, resample=cv2.INTER_LANCZOS4, **kwargs):
        if np.prod(self.whN) == 0:  # 避免出现无法求解的矩阵
            return self
        xlylN_ori = xyxyN2xlylN(self.xyxyN)
        xyxyN = np.round(xlylN2xyxyN(xlylN)).astype(np.int32)
        H = cv2.getPerspectiveTransform((xlylN_ori - self.xyN).astype(np.float32),
                                        (xlylN - xyxyN[:2]).astype(np.float32))
        size_ref = tuple(xyxyN[2:] - xyxyN[:2])
        self.maskN_ref = cv2.warpPerspective(self.maskN_ref.astype(np.float32), H, size_ref, flags=resample)
        self.xyN = xyxyN[:2]
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2),
               resample=cv2.INTER_LANCZOS4, **kwargs):
        xyxyN_lin = xlylN2xyxyN(xyxyN2xlylN(self.xyxyN) * np.array(scale) + np.array(bias))
        xyxyN_lin = np.round(xyxyN_lin).astype(np.int32)
        Ap = np.array([[scale[0], 0, 0], [0, scale[1], 0]])
        size_ref = tuple(xyxyN_lin[2:] - xyxyN_lin[:2])
        self.maskN_ref = cv2.warpAffine(self.maskN_ref.astype(np.float32), Ap, size_ref, flags=resample)
        self.xyN = xyxyN_lin[:2]
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), resample=cv2.INTER_LANCZOS4, **kwargs):
        xlylN_ori = xyxyN2xlylN(self.xyxyN)
        xlylN_persp = xlylN_perspective(xlylN_ori, H)
        xyxyN_persp = np.round(xlylN2xyxyN(xlylN_persp)).astype(np.int32)
        Hp = cv2.getPerspectiveTransform((xlylN_ori - self.xyN).astype(np.float32),
                                         (xlylN_persp - xyxyN_persp[:2]).astype(np.float32))
        size_ref = tuple(xyxyN_persp[2:] - xyxyN_persp[:2])
        self.maskN_ref = cv2.warpPerspective(self.maskN_ref.astype(np.float32), Hp, size_ref, flags=resample)
        self.xyN = xyxyN_persp[:2]
        self.clip(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @staticmethod
    def _maskNb2xyxyN(maskNb):
        ys, xs = np.where(maskNb)
        xyxy = np.array([np.min(xs), np.min(ys), np.max(xs) + 1, np.max(ys) + 1]).astype(np.int32) \
            if len(ys) > 0 else np.array([0, 0, 1, 1]).astype(np.int32)
        return xyxy

    @staticmethod
    def convert(rgn):
        if isinstance(rgn, RefValRegion):
            return rgn
        elif isinstance(rgn, AbsBoolRegion):
            xyxy = RefValRegion._maskNb2xyxyN(rgn.maskNb)
            maskNb_ref = rgn.maskNb_abs[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            return RefValRegion(maskN_ref=maskNb_ref.astype(np.float32), xyN=xyxy[:2], size=rgn.size, conf_thres=0.5)
        elif isinstance(rgn, AbsValRegion):
            xyxy = RefValRegion._maskNb2xyxyN(rgn.maskNb)
            maskNb_ref = rgn.maskN_abs[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            return RefValRegion(maskN_ref=maskNb_ref.astype(np.float32), xyN=xyxy[:2], size=rgn.size,
                                conf_thres=rgn.conf_thres)
        elif isinstance(rgn, XYXYBorder):
            xyxyN = rgn.xyxyN.astype(np.int32)
            size_ref = xyxyN[2:4] - xyxyN[:2]
            maskNb_ref = np.full(shape=(size_ref[1], size_ref[0]), fill_value=1)
            return RefValRegion(maskN_ref=maskNb_ref, xyN=xyxyN[:2], size=rgn.size, conf_thres=0.5)
        elif isinstance(rgn, Border):
            xlylN = XLYLBorder.convert(rgn).xlylN
            xyxyN = xlylN2xyxyN(xlylN)
            xyN_min = np.maximum(np.floor(xyxyN[:2]), np.zeros(shape=2)).astype(np.int32)
            xyN_max = np.minimum(np.ceil(xyxyN[2:4]), np.array(rgn.size)).astype(np.int32)
            # size = np.maximum(xyN_max - xyN_min, np.zeros(shape=2))
            maskNb_ref = xlylN2maskNb(xlylN - xyN_min, size=tuple(xyN_max - xyN_min))
            return RefValRegion(maskN_ref=maskNb_ref.astype(np.float32), xyN=xyN_min, size=rgn.size, conf_thres=0.5)
        else:
            raise Exception('err fmt ' + str(rgn.__class__.__name__))

    @staticmethod
    def from_maskNb_xyxyN(maskNb, xyxyN):
        size = (maskNb.shape[1], maskNb.shape[0])
        xyxyN = xyxyN_clip(xyxyN, np.array([0, 0, maskNb.shape[1], maskNb.shape[0]])).astype(np.int32)
        maskNb_ref = maskNb[xyxyN[1]:xyxyN[3], xyxyN[0]:xyxyN[2]]
        return RefValRegion(maskN_ref=maskNb_ref.astype(np.float32), xyN=xyxyN[:2], size=size, conf_thres=0.5)

    def __repr__(self):
        return 'rmsk' + str(self.whN)


# </editor-fold>

# <editor-fold desc='分类标签'>
def cates2chotsN(cates: list) -> np.ndarray:
    chotsN = []
    for cate in cates:
        assert isinstance(cate, CategoryLabel), 'class err ' + cate.__class__.__name__
        chotsN.append(OneHotCategory.convert(cate.category).chotN[None, :])
    chotsN = np.concatenate(chotsN, axis=0)
    return chotsN


def cates2cindsN(cates: list) -> np.ndarray:
    cindsN = []
    for cate in cates:
        assert isinstance(cate, CategoryLabel), 'class err ' + cate.__class__.__name__
        cindsN.append(IndexCategory.convert(cate.category).cindN)
    cindsN = np.array(cindsN)
    return cindsN


def chotsT2cates(chotsT: torch.Tensor, img_size: tuple, cind2name=None) -> list:
    cates = []
    chotsN = chotsT.detach().cpu().numpy()
    for chotN in chotsN:
        category = OneHotCategory(chotN=chotN)
        cate = CategoryLabel(category=category, img_size=img_size)
        if cind2name is not None:
            cate['name'] = cind2name(np.argmax(chotN))
        cates.append(cate)
    return cates


def cindN2chotN(cindN: np.ndarray, num_cls: int, conf: float = 1) -> np.ndarray:
    chotN = np.zeros(shape=num_cls)
    chotN[cindN] = conf
    return chotN


def chotN2cindN(chotN: np.ndarray) -> np.ndarray:
    return np.argmax(chotN)


def chotsN2cindsN(chotN: np.ndarray) -> np.ndarray:
    return np.argmax(chotN, axis=1)


class Category():
    def __init__(self, num_cls):
        self.num_cls = num_cls

    @abstractmethod
    def conf_scale(self, scale):
        pass

    @property
    @abstractmethod
    def conf(self):
        pass

    @staticmethod
    def convert(category):
        if isinstance(category, Category):
            return category
        elif isinstance(category, int):
            return IndexCategory.convert(category)
        elif isinstance(category, Iterable):
            return OneHotCategory.convert(category)
        else:
            raise Exception('err fmt ' + category.__class__.__name__)


class IndexCategory(Category):
    __slots__ = ('cindN', '_conf')

    def __init__(self, cindN, num_cls, conf=1.0):
        super().__init__(num_cls)
        self.cindN = np.array(cindN).astype(np.int32)
        self._conf = np.array(conf)

    def __repr__(self):
        return '<' + str(self.cindN) + '>'

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, conf):
        self._conf = conf

    def conf_scale(self, scale):
        self._conf *= scale

    @staticmethod
    def convert(category):
        if isinstance(category, int):
            return IndexCategory(cindN=category, num_cls=category + 1, conf=1.0)
        elif isinstance(category, OneHotCategory):
            return IndexCategory(cindN=chotN2cindN(category.chotN), num_cls=category.num_cls,
                                 conf=np.max(category.chotN))
        elif isinstance(category, IndexCategory):
            return category
        else:
            raise Exception('err fmt ' + category.__class__.__name__)


class OneHotCategory(Category):
    __slots__ = ('chotN',)

    def __init__(self, chotN):
        self.chotN = np.array(chotN)
        super().__init__(num_cls=self.chotN.shape)

    def conf_scale(self, scale):
        self.chotN *= scale

    @property
    def conf(self):
        return np.max(self.chotN)

    @staticmethod
    def convert(category):
        if isinstance(category, IndexCategory):
            return OneHotCategory(chotN=cindN2chotN(category.cindN, num_cls=category.num_cls))
        elif isinstance(category, OneHotCategory):
            return category
        elif isinstance(category, Iterable):
            return OneHotCategory(chotN=category)
        else:
            raise Exception('err fmt ' + category.__class__.__name__)

    def __repr__(self):
        return '<' + str(np.argmax(self.chotN)) + '>'


# </editor-fold>

# <editor-fold desc='标签单体'>

# img_size 对应图像大小
# ctx_border 对应图像有内容区域
# img_size_init 初始图像大小
# meta 图像唯一标识
class ImageLabel(Augurable, Movable):
    def __init__(self, img_size, meta=None, **kwargs):
        self.init_size = img_size
        self.meta = meta
        self.kwargs = kwargs
        for name, val in kwargs.items():
            self.__setattr__(name, val)

    @property
    def img_size(self):
        return self.ctx_border.size

    @img_size.setter
    def img_size(self, img_size):
        self.ctx_border.size = img_size

    @property
    def size(self):
        return self.ctx_border.size

    @size.setter
    def size(self, size):
        self.ctx_border.size = size

    @property
    def ctx_size(self):
        return self.ctx_border.size

    # 重置图像标记区域大小
    @ctx_size.setter
    def ctx_size(self, ctx_size):
        self.ctx_border = XLYLBorder(xyxyN2xlylN(np.array([0, 0, ctx_size[0], ctx_size[1]])), size=ctx_size)

    @property
    def init_size(self):
        return self._init_size

    # 重置初始图像大小和图像标记区域大小
    @init_size.setter
    def init_size(self, init_size):
        self._init_size = init_size
        self.ctx_size = init_size

    # 通过投影变换恢复标注
    def recover(self):
        self.ctx_size = self.init_size
        return self

    # 迁移原图坐标位置
    def ctx_from(self, label):
        self._init_size = label._init_size
        self.ctx_border = label.ctx_border

    def extract_xlylN(self):
        return self.ctx_border.xlylN

    def refrom_xlylN(self, xlylN, size: tuple, **kwargs):
        self.ctx_border.xlylN = xlylN
        self.img_size = size
        return self

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        self.ctx_border.xlylN = self.ctx_border.xlylN * scale + bias
        self.img_size = size
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        self.img_size = size
        self.ctx_border.xlylN = xlylN_perspective(self.ctx_border.xlylN, H=H)
        return self


class ImageItem(dict, Augurable, Movable):
    def __init__(self, *seq, **kwargs):
        super(ImageItem, self).__init__(*seq, **kwargs)

    @abstractmethod
    def measure(self):
        pass

    @abstractmethod
    def area(self):
        pass

    @property
    def num_pnt(self):
        num_pnt = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, PntExtractable):
                num_pnt += attr.num_pnt
        return num_pnt

    def extract_xlylN(self):
        xlylN = []
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, PntExtractable):
                xlylN.append(attr.extract_xlylN())
        xlylN = np.concatenate(xlylN, axis=0) if len(xlylN) > 0 else None
        return xlylN

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, **kwargs):
        ptr = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, PntExtractable):
                dt = attr.num_pnt
                attr.refrom_xlylN(xlylN[ptr:ptr + dt], size, **kwargs)
                ptr = ptr + dt
        return self

    @property
    def num_bool_chan(self):
        num_bool_chan = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, BoolMaskExtractable):
                num_bool_chan += attr.num_bool_chan
        return num_bool_chan

    def extract_maskNb(self):
        maskNb = []
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, BoolMaskExtractable):
                maskNb.append(attr.extract_maskNb())
        maskNb = np.concatenate(maskNb, axis=-1) if len(maskNb) > 0 else None
        return maskNb

    def refrom_maskNb(self, maskNb, **kwargs):
        ptr = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, BoolMaskExtractable):
                dt = attr.num_bool_chan
                attr.refrom_maskNb(maskNb[..., ptr:ptr + dt], **kwargs)
                ptr = ptr + dt
        return self

    @property
    def num_chan(self):
        num_chan = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, ValMaskExtractable):
                num_chan += attr.num_chan
        return num_chan

    def extract_maskN(self):
        maskN = []
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, ValMaskExtractable):
                maskN.append(attr.extract_maskN())
        maskN = np.concatenate(maskN, axis=-1) if len(maskN) > 0 else None
        return maskN

    def refrom_maskN(self, maskN, **kwargs):
        ptr = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, ValMaskExtractable):
                dt = attr.num_chan
                attr.refrom_maskN(maskN[..., ptr:ptr + dt], **kwargs)
                ptr = ptr + dt
        return self

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, Movable):
                attr.linear(size=size, bias=bias, scale=scale, **kwargs)
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, Movable):
                attr.perspective(size=size, H=H, **kwargs)
        return self

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, Movable):
                attr.clip(xyxyN_rgn=xyxyN_rgn, **kwargs)
        return self


class CategoryLabel(ImageLabel, dict):

    @property
    def num_chan(self) -> int:
        return 0

    def extract_maskN(self):
        return None

    def refrom_maskN(self, maskN, **kwargs):
        return self

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        return self

    __slots__ = ('category',)

    def __init__(self, category, img_size=(0, 0), meta=None, *seq, **kwargs):
        super(CategoryLabel, self).__init__(img_size=img_size, meta=meta)
        super(ImageLabel, self).__init__(*seq, **kwargs)
        self.category = Category.convert(category)

    @property
    def num_pnt(self):
        return 4

    @property
    def num_bool_chan(self):
        return 0

    def extract_maskNb(self):
        return None

    def refrom_maskNb(self, maskNb, **kwargs):
        return self

    def __repr__(self):
        return self.category.__repr__() + super(CategoryLabel, self).__repr__()


class PointItem(ImageItem):
    def measure(self):
        return 0 if np.any(np.isnan(self.xyN)) else np.inf

    def area(self):
        return 0

    __slots__ = ('xyN', 'category', '_size',)

    def __init__(self, xyN, category, size, *seq, **kwargs):
        super().__init__(*seq, **kwargs)
        self.xyN = np.array(xyN)
        self.category = Category.convert(category)
        self._size = size

    @property
    def img_size(self):
        return self._size

    @property
    def size(self):
        return self._size

    @img_size.setter
    def img_size(self, img_size):
        self._size = tuple(img_size)

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    @property
    def num_pnt(self):
        return 1

    def extract_xlylN(self):
        return self.xyN[None, :]

    def refrom_xlylN(self, xlylN, size, **kwargs):
        self.xyN = xlylN[0]
        self.clip(xyxyN_rgn=np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @property
    def num_bool_chan(self):
        return 0

    def extract_maskNb(self):
        return None

    def refrom_maskNb(self, maskNb, **kwargs):
        return self

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        self.xyN = self.xyN * scale + bias
        self.clip(xyxyN_rgn=np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective(self, H: np.ndarray = np.zeros(shape=(3, 3)), size: tuple = None, **kwargs):
        self.xyN = xyN_perspective(self.xyN, H=H)
        self.clip(xyxyN_rgn=np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        if self.xyN[0] < xyxyN_rgn[0] or self.xyN[0] >= xyxyN_rgn[2] \
                or self.xyN[1] < xyxyN_rgn[1] or self.xyN[1] >= xyxyN_rgn[3]:
            self.xyN[:] = np.nan
        return self

    @staticmethod
    def convert(pnt):
        if isinstance(pnt, PointItem):
            return pnt
        if isinstance(pnt, Iterable):
            return PointItem(xyN=pnt, category=0, size=tuple(pnt))
        else:
            raise Exception('err fmt' + str(pnt.__class__.__name__))

    def __repr__(self):
        return self.category.__repr__() + 'pnt' + str(self.xyN) + super(PointItem, self).__repr__()


class BoxItem(ImageItem):
    def measure(self):
        return self.border.measure()

    def area(self):
        return self.border.area()

    @property
    def img_size(self):
        return self.border.size

    @img_size.setter
    def img_size(self, img_size):
        self.border.size = img_size

    @property
    def size(self):
        return self.border.size

    @size.setter
    def size(self, size):
        self.border.size = size

    __slots__ = ('border', 'category',)

    def __init__(self, border, category, *seq, **kwargs):
        super().__init__(*seq, **kwargs)
        # print(border)
        self.border = Border.convert(border)
        self.category = Category.convert(category)

    @property
    def num_pnt(self):
        return self.border.num_pnt

    @property
    def num_bool_chan(self):
        return 0

    def extract_maskNb(self):
        return None

    def refrom_maskNb(self, maskNb, **kwargs):
        return self

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.border.clip(xyxyN_rgn, **kwargs)
        return self

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        self.border.linear(bias=bias, scale=scale, size=size, **kwargs)
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        self.border.perspective(H=H, size=size, **kwargs)
        return self

    @staticmethod
    def convert(box):
        if isinstance(box, BoxItem):
            return box
        elif isinstance(box, InstItem):
            return BoxItem(border=box.border, category=box.category, **box)
        elif isinstance(box, Iterable):
            return BoxItem(border=box, category=0)
        else:
            raise Exception('err fmt ' + box.__class__.__name__)

    def __repr__(self):
        return self.category.__repr__() + self.border.__repr__() + super(BoxItem, self).__repr__()

    def extract_xlylN(self):
        return self.border.extract_xlylN()

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, **kwargs):
        self.border.refrom_xlylN(xlylN, size, **kwargs)
        return self


class BoxRefItem(BoxItem):
    __slots__ = ('border', 'border_ref', 'category',)

    def __init__(self, border, border_ref, category, *seq, **kwargs):
        super().__init__(border, category, *seq, **kwargs)
        self.border_ref = XYWHABorder.convert(border_ref)

    @staticmethod
    def convert(box):
        if isinstance(box, BoxRefItem):
            return box
        elif isinstance(box, BoxItem):
            border_ref = copy.deepcopy(box.border)
            return BoxRefItem(border=box.border, category=box.category, border_ref=border_ref, **box)
        elif isinstance(box, Iterable):
            return BoxRefItem(border=box, category=0, border_ref=copy.deepcopy(box))
        else:
            raise Exception('err fmt' + str(box.__class__.__name__))

    def __repr__(self):
        return self.category.__repr__() + self.border.__repr__() + \
               '(' + self.border_ref.__repr__() + ')' + super(BoxItem, self).__repr__()

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        super(BoxRefItem, self).linear(bias=bias, scale=scale, size=size, **kwargs)
        self.border_ref.linear(bias=bias, scale=scale, size=size, **kwargs)
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        super(BoxRefItem, self).perspective(H=H, size=size, **kwargs)
        self.border_ref.perspective(H=H, size=size, **kwargs)
        return self

    def extract_xlylN(self):
        xlyl = self.border.extract_xlylN()
        xlyl_ref = self.border_ref.extract_xlylN()
        return np.concatenate([xlyl, xlyl_ref], axis=0)

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, **kwargs):
        self.border.refrom_xlylN(xlylN[:self.border.num_pnt], size, **kwargs)
        self.border_ref.refrom_xlylN(xlylN[self.border.num_pnt:], size, **kwargs)
        return self

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.border.clip(xyxyN_rgn, **kwargs)
        self.border_ref.clip(xyxyN_rgn, **kwargs)
        return self

    @property
    def num_pnt(self):
        return self.border.num_pnt + self.border_ref.num_pnt


class SegItem(ImageItem):

    @property
    def img_size(self):
        return self.rgn.size

    @img_size.setter
    def img_size(self, img_size):
        self.rgn.size = img_size

    @property
    def size(self):
        return self.rgn.size

    @size.setter
    def size(self, size):
        self.rgn.size = size

    __slots__ = ('rgn', 'category',)

    def __init__(self, category, rgn, *seq, **kwargs):
        super(SegItem, self).__init__(*seq, **kwargs)
        self.category = Category.convert(category)
        self.rgn = BoolRegion.convert(rgn)

    @staticmethod
    def convert(seg):
        if isinstance(seg, SegItem):
            return seg
        elif isinstance(seg, BoxItem):
            rgn = XLYLBorder.convert(seg.border)
            return SegItem(rgn=rgn, category=seg.category, **seg)
        elif isinstance(seg, PIL.Image.Image) or isinstance(seg, np.ndarray):
            rgn = AbsBoolRegion(seg)
            return SegItem(rgn=rgn, category=0)
        else:
            raise Exception('err fmt ' + seg.__class__.__name__)

    def __repr__(self):
        return self.category.__repr__() + self.rgn.__repr__()

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        self.rgn.linear(bias=bias, scale=scale, size=size, **kwargs)
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        self.rgn.perspective(H=H, size=size, **kwargs)
        return self

    def measure(self):
        return self.rgn.measure()

    def area(self):
        return self.rgn.area()

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.rgn.clip(xyxyN_rgn=xyxyN_rgn, **kwargs)
        return self


class InstItem(ImageItem):
    __slots__ = ('border', 'rgn', 'category',)

    def __init__(self, border, rgn, category, *seq, **kwargs):
        super(InstItem, self).__init__(*seq, **kwargs)
        self.category = Category.convert(category)
        self.rgn = BoolRegion.convert(rgn)
        self.border = Border.convert(border)

    @property
    def img_size(self):
        return self.rgn.size

    @img_size.setter
    def img_size(self, img_size):
        self.rgn.size = img_size
        self.border.size = img_size

    @property
    def size(self):
        return self.rgn.size

    @size.setter
    def size(self, size):
        self.rgn.size = size
        self.border.size = size

    def measure(self):
        return min(self.border.measure(), self.rgn.measure())

    def area(self):
        return self.rgn.area()

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.border.clip(xyxyN_rgn=xyxyN_rgn, **kwargs)
        self.rgn.clip(xyxyN_rgn=xyxyN_rgn, **kwargs)
        return self

    def align(self):
        self.border.align_with(self.rgn.ixysN)
        return self

    @staticmethod
    def convert(inst):
        if isinstance(inst, InstItem):
            return inst
        elif isinstance(inst, BoxItem):
            rgn = RefValRegion.convert(inst.border)
            return InstItem(border=inst.border, category=inst.category, rgn=rgn, **inst)
        else:
            raise Exception('err fmt' + str(inst.__class__.__name__))

    def __repr__(self):
        return self.category.__repr__() + self.border.__repr__() + self.rgn.__repr__() + super(InstItem,
                                                                                               self).__repr__()

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        self.border.linear(bias=bias, scale=scale, size=size, **kwargs)
        self.rgn.linear(bias=bias, scale=scale, size=size, **kwargs)
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        self.border.perspective(H=H, size=size, **kwargs)
        self.rgn.perspective(H=H, size=size, **kwargs)
        return self


class InstRefItem(InstItem):
    __slots__ = ('border', 'border_ref', 'rgn', 'category',)

    def __init__(self, border, border_ref, rgn, category, *seq, **kwargs):
        super(InstRefItem, self).__init__(border, rgn, category, *seq, **kwargs)
        self.border_ref = border_ref

    @property
    def img_size(self):
        return self.rgn.size

    @img_size.setter
    def img_size(self, img_size):
        self.rgn.size = img_size
        self.border.size = img_size
        self.border_ref.size = img_size

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.border.clip(xyxyN_rgn=xyxyN_rgn, **kwargs)
        self.rgn.clip(xyxyN_rgn=xyxyN_rgn, **kwargs)
        self.border_ref.clip(xyxyN_rgn=xyxyN_rgn, **kwargs)
        return self

    @staticmethod
    def convert(inst):
        if isinstance(inst, InstRefItem):
            return inst
        elif isinstance(inst, InstItem):
            return InstRefItem(border=inst.border, border_ref=inst.border,
                               category=inst.category, rgn=inst.rgn, **inst)
        elif isinstance(inst, BoxRefItem):
            rgn = RefValRegion.convert(inst.border)
            return InstRefItem(border=inst.border, border_ref=inst.border_ref,
                               category=inst.category, rgn=rgn, **inst)
        else:
            raise Exception('err fmt' + str(inst.__class__.__name__))

    def __repr__(self):
        return self.category.__repr__() + self.border.__repr__() + \
               '(' + self.border_ref.__repr__() + ')' + self.rgn.__repr__() + super(InstItem, self).__repr__()

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        self.border.linear(bias=bias, scale=scale, size=size, **kwargs)
        self.border_ref.linear(bias=bias, scale=scale, size=size, **kwargs)
        self.rgn.linear(bias=bias, scale=scale, size=size, **kwargs)
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        self.border.perspective(H=H, size=size, **kwargs)
        self.border_ref.perspective(H=H, size=size, **kwargs)
        self.rgn.perspective(H=H, size=size, **kwargs)
        return self


class StereoObjLabel(ImageItem):
    def __init__(self, mesh, category, mesh_posi=(0, 0, 0), mesh_axang=(0, 0, 0), scale=1, *seq, **kwargs):
        super(StereoObjLabel, self).__init__(*seq, **kwargs)
        self.mesh = mesh
        self.mesh_posi = torch.Tensor(mesh_posi)
        self.mesh_axang = torch.Tensor(mesh_axang)
        self.scale = scale
        if not isinstance(category, Category):
            category = IndexCategory.convert(category)
        self.category = category

    @property
    def transed_mesh(self):
        pass
        # R = axis_angle_to_matrix(self.mesh_axang)
        # mesh = self.mesh
        # verts = mesh.verts_list()[0] * self.scale
        # verts = self.mesh_posi + verts @ R
        # mesh = Meshes(verts=[verts], faces=self.mesh.faces_list(), textures=self.mesh.textures)
        # return mesh

    @staticmethod
    def anchor_mesh(b=1):
        pass
        # verts = torch.Tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]]).repeat(b, 1, 1)
        # faces = torch.Tensor([[[0, 3, 2], [0, 2, 1], [0, 1, 3], [3, 1, 2]]]).repeat(b, 1, 1)
        # texes = TexturesVertex(verts)
        # meshes = Meshes(verts=verts, faces=faces, textures=texes)
        # return meshes

    def measure(self):
        pass

    def area(self):
        pass

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        pass

    def linear(self, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        pass

    def perspective(self, H: np.ndarray = np.zeros(shape=(3, 3)), **kwargs):
        pass


class StereoObjItem(ImageItem):
    def __init__(self, *objs, focus=100, light_posi=(0, 0, 0), img_size=(0, 0), meta=None,
                 diffuse_color=(0.3, 0.3, 0.3), ambient_color=(0.5, 0.5, 0.5), specular_color=(0.2, 0.2, 0.2),
                 **kwargs):
        super(StereoObjItem, self).__init__(*objs, img_size=img_size, meta=meta, **kwargs)
        self.focus = focus
        self.ambient_color = torch.Tensor(ambient_color)
        self.diffuse_color = torch.Tensor(diffuse_color)
        self.specular_color = torch.Tensor(specular_color)
        self.light_posi = torch.Tensor(light_posi)

    @property
    def camera(self):
        pass
        # w, h = self.img_size
        # R = axis_angle_to_matrix(torch.Tensor([[0, 0, math.pi]]))
        # T = torch.Tensor([[w / 2, h / 2, self.focus]])
        # aspect_ratio = 1
        # camera = FoVPerspectiveCameras(
        #     R=R, T=T, aspect_ratio=aspect_ratio, znear=0.01, zfar=100,
        #     fov=np.arctan(w / 2 / self.focus) * 2, degrees=False)
        # return camera

    @property
    def light(self):
        pass
        # light = PointLights(location=self.light_posi[None, :], ambient_color=self.ambient_color[None, :],
        #                     diffuse_color=self.diffuse_color[None, :], specular_color=self.specular_color[None, :])
        # return light


# </editor-fold>

# <editor-fold desc='快速导出工具'>
# 基本导出工具
class BaseExportable(Iterable):
    def export_attrsN(self, aname, default):
        return np.array([getattr(item, aname, default) for item in self])

    def export_valsN(self, key, default):
        return np.array([item.get(key, default) for item in self])

    def export_namesN(self):
        return self.export_valsN('name', 'unknown')

    def export_ignoresN(self):
        return self.export_valsN('ignore', False).astype(bool) * self.export_valsN('difficult', False).astype(bool)

    def export_crowdsN(self):
        return self.export_valsN('crowd', False).astype(bool)


class CategoryExportable(BaseExportable):
    ANMAE_CATEGORY = 'category'

    def export_cindsN(self, aname_cate=ANMAE_CATEGORY):
        cindsN = []
        for item in self:
            cindsN.append(IndexCategory.convert(getattr(item, aname_cate)).cindN)
        cindsN = np.array(cindsN)
        return cindsN

    def export_confsN(self, aname_cate=ANMAE_CATEGORY):
        confsN = []
        for item in self:
            confsN.append(getattr(item, aname_cate).conf)
        confsN = np.array(confsN)
        return confsN

    def export_cindsN_confsN(self, aname_cate=ANMAE_CATEGORY):
        confsN = []
        cindsN = []
        for item in self:
            cate = IndexCategory.convert(getattr(item, aname_cate))
            cindsN.append(cate.cindN)
            confsN.append(cate.conf)
        confsN = np.array(confsN)
        cindsN = np.array(cindsN)
        return cindsN, confsN

    def export_chotsN(self, num_cls, aname_cate=ANMAE_CATEGORY):
        chotsN = [np.zeros(shape=(0, num_cls))]
        for item in self:
            chotsN.append(OneHotCategory.convert(getattr(item, aname_cate)).chotN[None])
        chotsN = np.concatenate(chotsN, axis=0)
        return chotsN


class BorderExportable(CategoryExportable):
    ANMAE_BORDER = 'border'

    def _export_bordersN(self, border_type, aname_bdr=ANMAE_BORDER):
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        for item in self:
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
        bordersN = np.concatenate(bordersN, axis=0)
        return bordersN

    def _export_bordersN_cindsN(self, border_type, aname_bdr=ANMAE_BORDER,
                                aname_cate=CategoryExportable.ANMAE_CATEGORY):
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        cindsN = []
        for item in self:
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
            cindsN.append(IndexCategory.convert(getattr(item, aname_cate)).cindN)
        bordersN = np.concatenate(bordersN, axis=0)
        cindsN = np.array(cindsN)
        return bordersN, cindsN

    def _export_bordersN_chotsN(self, border_type, num_cls, aname_bdr=ANMAE_BORDER,
                                aname_cate=CategoryExportable.ANMAE_CATEGORY):
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        chotsN = [np.zeros(shape=(0, num_cls))]
        for item in self:
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
            chotsN.append(OneHotCategory.convert(getattr(item, aname_cate)).chotN[None])
        bordersN = np.concatenate(bordersN, axis=0)
        chotsN = np.concatenate(chotsN, axis=0)
        return bordersN, chotsN

    def _export_bordersN_cindsN_confsN(self, border_type, aname_bdr=ANMAE_BORDER,
                                       aname_cate=CategoryExportable.ANMAE_CATEGORY):
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        cindsN = []
        confsN = []
        for item in self:
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
            cate = IndexCategory.convert(getattr(item, aname_cate))
            cindsN.append(cate.cindN)
            confsN.append(cate.conf)
        bordersN = np.concatenate(bordersN, axis=0)
        cindsN = np.array(cindsN)
        confsN = np.array(confsN)
        return bordersN, cindsN, confsN

    def export_xyxysN(self, aname_bdr=ANMAE_BORDER):
        return self._export_bordersN(XYXYBorder, aname_bdr)

    def export_xywhsN(self, aname_bdr=ANMAE_BORDER):
        return self._export_bordersN(XYWHBorder, aname_bdr)

    def export_xywhasN(self, aname_bdr=ANMAE_BORDER):
        return self._export_bordersN(XYWHABorder, aname_bdr)

    def export_xlyls(self, aname_bdr=ANMAE_BORDER):
        xlyls = []
        for item in self:
            border = XLYLBorder.convert(getattr(item, aname_bdr))
            xlyls.append(border.xlylN)
        return xlyls

    def export_xyxysN_cindsN(self, aname_bdr=ANMAE_BORDER, aname_cate=CategoryExportable.ANMAE_CATEGORY):
        return self._export_bordersN_cindsN(XYXYBorder, aname_bdr, aname_cate)

    def export_xywhsN_cindsN(self, aname_bdr=ANMAE_BORDER, aname_cate=CategoryExportable.ANMAE_CATEGORY):
        return self._export_bordersN_cindsN(XYWHBorder, aname_bdr, aname_cate)

    def export_xyxysN_chotsN(self, num_cls, aname_bdr=ANMAE_BORDER, aname_cate=CategoryExportable.ANMAE_CATEGORY):
        return self._export_bordersN_chotsN(XYXYBorder, num_cls, aname_bdr, aname_cate)

    def export_xywhsN_chotsN(self, num_cls, aname_bdr=ANMAE_BORDER, aname_cate=CategoryExportable.ANMAE_CATEGORY):
        return self._export_bordersN_chotsN(XYWHBorder, num_cls, aname_bdr, aname_cate)

    def export_xywhasN_cindsN(self, aname_bdr=ANMAE_BORDER, aname_cate=CategoryExportable.ANMAE_CATEGORY):
        return self._export_bordersN_cindsN(XYWHABorder, aname_bdr, aname_cate)

    def export_border_masksN_enc(self, img_size: tuple, num_cls: int, aname_bdr=ANMAE_BORDER,
                                 aname_cate=CategoryExportable.ANMAE_CATEGORY) -> np.ndarray:
        maskN = np.full(shape=(img_size[1], img_size[0]), fill_value=num_cls, dtype=np.int32)
        for item in self:
            cind = IndexCategory.convert(getattr(item, aname_cate)).cindN
            border = getattr(item, aname_bdr)
            if isinstance(border, XYXYBorder) or isinstance(border, XYWHBorder):
                xyxy = XYXYBorder.convert(border).xyxyN.astype(np.int32)
                maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = cind
            else:
                rgn = RefValRegion.convert(border)
                xyxy = rgn.xyxyN.astype(np.int32)
                patch = maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                filler = np.full_like(patch, fill_value=cind, dtype=np.int32)
                merge = np.where(np.array(rgn.maskNb_ref), filler, patch)
                maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = merge
        return maskN


class RegionExportable(CategoryExportable):
    ANMAE_REGION = 'rgn'

    def export_masksN_stk(self, img_size: tuple, aname_rgn=ANMAE_REGION, ) -> np.ndarray:
        masksN = [np.zeros(shape=(0, img_size[1], img_size[0]), dtype=bool)]
        for item in self:
            rgn = getattr(item, aname_rgn)
            masksN.append(rgn.maskNb)
        masksN = np.concatenate(masksN, axis=0)
        return masksN

    def export_masksN_abs(self, img_size: tuple, num_cls: int, aname_cate=CategoryExportable.ANMAE_CATEGORY,
                          aname_rgn=ANMAE_REGION, append_bkgd: bool = True) -> np.ndarray:
        masksN = np.zeros(shape=(img_size[1], img_size[0], num_cls), dtype=bool)
        for item in self:
            cind = IndexCategory.convert(getattr(item, aname_cate)).cindN
            rgn = getattr(item, aname_rgn)
            if isinstance(rgn, RefValRegion):
                xyxy = rgn.xyxyN.astype(np.int32)
                masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], cind] += np.array(rgn.maskNb_ref)
            else:
                masksN[..., cind] += rgn.maskNb

        if append_bkgd:
            maskN_bkgd = np.all(~masksN, keepdims=True, axis=2)
            masksN = np.concatenate([masksN, maskN_bkgd], axis=2)
        return masksN

    def export_masksN_enc(self, img_size: tuple, num_cls: int, aname_cate=CategoryExportable.ANMAE_CATEGORY,
                          aname_rgn=ANMAE_REGION) -> np.ndarray:
        masksN = np.full(shape=(img_size[1], img_size[0]), fill_value=num_cls, dtype=np.int32)
        for item in self:
            cind = IndexCategory.convert(getattr(item, aname_cate)).cindN
            rgn = getattr(item, aname_rgn)
            if isinstance(rgn, RefValRegion):
                xyxy = rgn.xyxyN.astype(np.int32)
                if np.prod(rgn.maskNb_ref.size) > 0:
                    maskN_ref = np.array(rgn.maskNb_ref)
                    patch = masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    merge = np.where(maskN_ref, np.full_like(patch, fill_value=cind, dtype=np.int32), patch)
                    masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = merge
            else:
                masksN[rgn.maskNb] = cind
        return masksN


class InstExportable(BorderExportable, RegionExportable):

    def _export_bordersN_cindsN_masksN_enc(self, border_type, img_size: tuple, num_cls: int,
                                           aname_cate=CategoryExportable.ANMAE_CATEGORY,
                                           aname_bdr=BorderExportable.ANMAE_BORDER,
                                           aname_rgn=RegionExportable.ANMAE_REGION) -> np.ndarray:
        cindsN = []
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        masksN = np.full(shape=(img_size[1], img_size[0]), fill_value=num_cls, dtype=np.int32)
        for item in self:
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
            cind = IndexCategory.convert(getattr(item, aname_cate)).cindN
            cindsN.append(cind)
            rgn = getattr(item, aname_rgn)
            if isinstance(rgn, RefValRegion):
                xyxy = rgn.xyxyN.astype(np.int32)

                maskN_ref = np.array(rgn.maskNb_ref)
                patch = masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                merge = np.where(maskN_ref, np.full_like(patch, fill_value=cind, dtype=np.int32), patch)
                masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = merge
            else:
                masksN[rgn.maskNb] = cind
        return masksN

    def export_xyxysN_cindsN_masksN_enc(self, img_size: tuple, num_cls: int,
                                        aname_cate=CategoryExportable.ANMAE_CATEGORY,
                                        aname_bdr=BorderExportable.ANMAE_BORDER,
                                        aname_rgn=RegionExportable.ANMAE_REGION):
        return self._export_bordersN_cindsN_masksN_enc(XYXYBorder, img_size, num_cls, aname_cate, aname_bdr, aname_rgn)

    def export_xywhsN_cindsN_masksN_enc(self, img_size: tuple, num_cls: int,
                                        aname_cate=CategoryExportable.ANMAE_CATEGORY,
                                        aname_bdr=BorderExportable.ANMAE_BORDER,
                                        aname_rgn=RegionExportable.ANMAE_REGION):
        return self._export_bordersN_cindsN_masksN_enc(XYWHBorder, img_size, num_cls, aname_cate, aname_bdr, aname_rgn)

    def export_xywhasN_cindsN_masksN_enc(self, img_size: tuple, num_cls: int,
                                         aname_cate=CategoryExportable.ANMAE_CATEGORY,
                                         aname_bdr=BorderExportable.ANMAE_BORDER,
                                         aname_rgn=RegionExportable.ANMAE_REGION):
        return self._export_bordersN_cindsN_masksN_enc(XYWHABorder, img_size, num_cls, aname_cate, aname_bdr, aname_rgn)

    def export_rgn_xyxysN_cindsN_maskNs_ref(self, aname_cate=CategoryExportable.ANMAE_CATEGORY,
                                            aname_rgn=RegionExportable.ANMAE_REGION):
        xyxysN = [np.zeros(shape=(0, 4))]
        cindsN = []
        maskNs = []
        for item in self:
            cindsN.append(IndexCategory.convert(getattr(item, aname_cate)).cindN)
            rgn = RefValRegion.convert(getattr(item, aname_rgn))
            maskNs.append(rgn.maskNb_ref)
            xyxysN.append(rgn.xyxyN[None, :])
        xyxysN = np.concatenate(xyxysN, axis=0)
        cindsN = np.array(cindsN)
        return xyxysN, cindsN, maskNs


# </editor-fold>

# <editor-fold desc='标签列表'>
class ImageItemsLabel(list, ImageLabel):

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        for item in self:
            item.clip(xyxyN_rgn, **kwargs)
        return self

    def __init__(self, *items, img_size, meta=None, **kwargs):
        super(ImageItemsLabel, self).__init__(*items)
        super(list, self).__init__(img_size=img_size, meta=meta, **kwargs)

    @property
    def img_size(self):
        return self.ctx_border.size

    @img_size.setter
    def img_size(self, img_size):
        self.ctx_border.size = img_size
        for item in self:
            item.img_size = img_size

    @property
    def num_pnt(self):
        return 4 + sum([item.num_pnt for item in self])

    def extract_xlylN(self):
        xlylN = [self.ctx_border.xlylN]
        for item in self:
            xlylN_item = item.extract_xlylN()
            if xlylN_item is not None:
                xlylN.append(xlylN_item)
        xlylN = np.concatenate(xlylN, axis=0)
        return xlylN

    def refrom_xlylN(self, xlylN: np.ndarray, size: tuple, **kwargs):
        xlylN_ctx, xlylN = xlylN[:4], xlylN[4:]
        ptr = 0
        for item in self:
            dt = item.num_pnt
            if dt > 0:
                item.refrom_xlylN(xlylN[ptr:ptr + dt], size, **kwargs)
                ptr = ptr + dt
        super(ImageItemsLabel, self).refrom_xlylN(xlylN_ctx, size, **kwargs)
        return self

    @property
    def num_bool_chan(self):
        return sum([item.num_bool_chan for item in self])

    def extract_maskNb(self):
        maskNb = [item.extract_maskNb() for item in self if item.num_bool_chan > 0]
        maskNb = np.concatenate(maskNb, axis=-1) if len(maskNb) > 0 else None
        return maskNb

    def refrom_maskNb(self, maskNb, **kwargs):
        ptr = 0
        for item in self:
            if item.num_bool_chan > 0:
                dt = item.num_bool_chan
                item.refrom_maskNb(maskNb[..., ptr:ptr + dt], **kwargs)
                ptr = ptr + dt
        return self

    @property
    def num_chan(self) -> int:
        return sum([item.num_chan for item in self])

    def extract_maskN(self):
        maskN = [item.extract_maskN() for item in self if item.num_chan > 0]
        maskN = np.concatenate(maskN, axis=-1) if len(maskN) > 0 else None
        return maskN

    def refrom_maskN(self, maskN, **kwargs):
        ptr = 0
        for item in self:
            if item.num_chan > 0:
                dt = item.num_bool_chan
                item.refrom_maskN(maskN[..., ptr:ptr + dt], **kwargs)
                ptr = ptr + dt
        return self

    def linear(self, size: tuple, bias: np.ndarray = np.zeros(shape=2), scale: np.ndarray = np.ones(shape=2), **kwargs):
        for item in self:
            item.linear(bias=bias, scale=scale, size=size, **kwargs)
        super(ImageItemsLabel, self).linear(bias=bias, scale=scale, size=size, **kwargs)
        return self

    def perspective(self, size: tuple, H: np.ndarray = np.eye(3), **kwargs):
        for item in self:
            item.perspective(H=H, size=size, **kwargs)
        super(ImageItemsLabel, self).perspective(H=H, size=size, **kwargs)
        return self

    def flit(self, xyxyN_rgn: np.ndarray = None, thres: float = -1):
        for i in range(len(self) - 1, -1, -1):
            if xyxyN_rgn is not None:
                self[i].clip(xyxyN_rgn=xyxyN_rgn)
            if thres > 0 and self[i].measure() < thres:
                del self[i]
        return self

    def empty(self):
        items = self.__new__(self.__class__)
        items.__init__(img_size=self.img_size, meta=self.meta, **self.kwargs)
        return items

    def recover(self):
        xyxyN_init = np.array([0, 0, self.init_size[0], self.init_size[1]])
        xlylN_init = xyxyN2xlylN(xyxyN_init)
        if self.img_size == self.init_size and np.all(xlylN_init == self.ctx_border.xlylN):
            return self
        H = xlylN2homography(xlylN_src=self.ctx_border.xlylN, xlylN_tgd=xlylN_init)
        for item in self:
            item.perspective(H=H, size=self.init_size)
        self.ctx_size = self.init_size
        return self

    def __getitem__(self, item):
        if isinstance(item, Iterable):
            inst = self.empty()
            for ind in item:
                inst.append(self[ind])
            return inst
        else:
            return super(ImageItemsLabel, self).__getitem__(item)

    # def permutations(self,order):


class PointsLabel(ImageItemsLabel, CategoryExportable):
    def __init__(self, *pnts, img_size, meta=None, **kwargs):
        super(PointsLabel, self).__init__(*pnts, img_size=img_size, meta=meta, **kwargs)
        for i in range(len(self)):
            self[i] = PointItem.convert(self[i])
            self[i].size = img_size

    @staticmethod
    def convert(pnts):
        if isinstance(pnts, PointsLabel):
            return pnts
        else:
            raise Exception('err fmt ' + pnts.__class__.__name__)


class BoxesLabel(ImageItemsLabel, BorderExportable):
    def __init__(self, *boxes, img_size, meta=None, **kwargs):
        super(BoxesLabel, self).__init__(*boxes, img_size=img_size, meta=meta, **kwargs)
        for i in range(len(self)):
            self[i] = BoxItem.convert(self[i])
            self[i].size = img_size

    @staticmethod
    def convert(boxes):
        if isinstance(boxes, InstsLabel):
            boxes_list = [BoxItem(border=inst.border, category=inst.category, **inst) for inst in boxes]
            return BoxesLabel(boxes_list, img_size=boxes.img_size, meta=boxes.meta)
        elif isinstance(boxes, BoxesLabel):
            return boxes
        else:
            raise Exception('err fmt ' + boxes.__class__.__name__)

    @staticmethod
    def _from_bordersN_confsN_cindsN(border_type, bordersN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray,
                                     img_size: tuple, num_cls: int, cind2name=None):
        boxes = BoxesLabel(img_size=img_size)
        for borderN, conf, cind in zip(bordersN, confsN, cindsN):
            category = IndexCategory(cindN=cind, conf=conf, num_cls=num_cls)
            box = BoxItem(category=category, border=border_type(borderN, size=img_size))
            if cind2name is not None:
                box['name'] = cind2name(category.cindN)
            boxes.append(box)
        return boxes

    @staticmethod
    def _from_bordersT_confsT_cindsT(border_type, bordersT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor,
                                     img_size: tuple,
                                     num_cls: int, cind2name=None):
        bordersN = bordersT.detach().cpu().numpy()
        confsN = confsT.detach().cpu().numpy()
        cindsN = cindsT.detach().cpu().numpy()
        return BoxesLabel._from_bordersN_confsN_cindsN(border_type, bordersN, confsN, cindsN, img_size, num_cls,
                                                       cind2name)

    @staticmethod
    def from_xyxysN_confsN(xyxysN: np.ndarray, confsN: np.ndarray, img_size: tuple, num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersN_confsN_cindsN(XYXYBorder, xyxysN, confsN, np.zeros_like(confsN), img_size,
                                                       num_cls, cind2name)

    @staticmethod
    def from_xyxysT_confsT(xyxysT: torch.Tensor, confsT: torch.Tensor, img_size: tuple, num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XYXYBorder, xyxysT, confsT, torch.zeros_like(confsT), img_size,
                                                       num_cls, cind2name)

    @staticmethod
    def from_xyxysN_confsN_cindsN(xyxysN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray, img_size: tuple,
                                  num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersN_confsN_cindsN(XYXYBorder, xyxysN, confsN, cindsN, img_size, num_cls, cind2name)

    @staticmethod
    def from_xyxysT_confsT_cindsT(xyxysT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor, img_size: tuple,
                                  num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XYXYBorder, xyxysT, confsT, cindsT, img_size, num_cls, cind2name)

    @staticmethod
    def from_xywhsN_confsN_cindsN(xywhsN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray, img_size: tuple,
                                  num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersN_confsN_cindsN(XYWHBorder, xywhsN, confsN, cindsN, img_size, num_cls, cind2name)

    @staticmethod
    def from_xywhsT_confsT_cindsT(xywhsT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor, img_size: tuple,
                                  num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XYWHBorder, xywhsT, confsT, cindsT, img_size, num_cls, cind2name)

    @staticmethod
    def from_xywhasN_confsN_cindsN(xywhasN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray, img_size: tuple,
                                   num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersN_confsN_cindsN(XYWHABorder, xywhasN, confsN, cindsN, img_size, num_cls,
                                                       cind2name)

    @staticmethod
    def from_xywhasT_confsT_cindsT(xywhasT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor, img_size: tuple,
                                   num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XYWHABorder, xywhasT, confsT, cindsT, img_size, num_cls,
                                                       cind2name)

    @staticmethod
    def from_xlylsT_confsT_cindsT(xlylsT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor, img_size: tuple,
                                  num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XLYLBorder, xlylsT, confsT, cindsT, img_size, num_cls,
                                                       cind2name)


class SegsLabel(ImageItemsLabel, RegionExportable):
    def __init__(self, *segs, img_size, meta=None, **kwargs):
        super(SegsLabel, self).__init__(*segs, img_size=img_size, meta=meta, **kwargs)
        for i in range(len(self)):
            self[i] = SegItem.convert(self[i])
            self[i].size = img_size

    @staticmethod
    def convert(segs):
        if isinstance(segs, SegsLabel):
            return segs
        elif isinstance(segs, BoxesLabel):
            segs_new = SegsLabel(img_size=segs.img_size, meta=segs.meta, **segs.kwargs)
            for box in segs:
                rgn = XLYLBorder.convert(box.border)
                seg = SegItem(rgn=rgn, category=box.category, **box)
                segs_new.append(seg)
            return segs_new
        else:
            raise Exception('err fmt ' + segs.__class__.__name__)

    @staticmethod
    def from_masksN(masksN: np.ndarray, num_cls: int, conf_thres: float = None, cind2name=None):
        _, H, W = masksN.shape
        segs = SegsLabel(img_size=(W, H))
        for cind in range(num_cls):
            masksN_cind = masksN[cind:cind + 1]
            conf_thres_i = conf_thres if conf_thres is not None else np.max(masksN_cind) / 2
            category = IndexCategory(cindN=cind, conf=np.max(masksN_cind), num_cls=num_cls)
            rgn = AbsValRegion(maskN_abs=masksN_cind, conf_thres=conf_thres_i)
            seg = SegItem(rgn=rgn, category=category)
            if cind2name is not None:
                seg['name'] = cind2name(cind)
            segs.append(seg)
        return segs

    @staticmethod
    def from_masksT(masksT: torch.Tensor, num_cls: int, conf_thres: float = None, cind2name=None):
        masksN = masksT.detach().cpu().numpy().astype(np.float32)
        return SegsLabel.from_masksN(masksN, num_cls, conf_thres, cind2name)


class InstsLabel(ImageItemsLabel, InstExportable):
    def __init__(self, *insts, img_size, meta=None, **kwargs):
        super(InstsLabel, self).__init__(*insts, img_size=img_size, meta=meta, **kwargs)
        for i in range(len(self)):
            self[i] = InstItem.convert(self[i])
            self[i].size = img_size

    def align(self):
        for inst in self:
            inst.align()
        return self

    def avoid_overlap(self):
        # 从大到小排序
        measures = np.array([inst.measure() for inst in self])
        order = np.argsort(-measures)
        lst_sortd = [self[ind] for ind in order]
        for i in range(len(self)):
            self[i] = lst_sortd[i]

        for inst in self:
            inst.rgn = AbsBoolRegion.convert(inst.rgn)
        if self.num_bool_chan > 0:
            maskN = self.extract_maskNb_enc(index=1)
            self.refrom_maskNb_enc(maskN, index=1)
        return self

    @staticmethod
    def convert(insts):
        if isinstance(insts, InstsLabel):
            return insts
        elif isinstance(insts, BoxesLabel):
            insts_new = InstsLabel(img_size=insts.img_size, meta=insts.meta, **insts.kwargs)
            for box in insts:
                rgn = XLYLBorder.convert(box.border)
                inst = InstItem(border=box.border, rgn=rgn, category=box.category, **box)
                insts_new.append(inst)
            return insts_new
        else:
            raise Exception('err fmt ' + insts.__class__.__name__)

    @staticmethod
    def from_boxes_masksN_ref(boxes: BoxesLabel, masksN: np.ndarray, conf_thres: float = 0.2,
                              resample=cv2.INTER_LANCZOS4, cind: int = None):
        img_size = boxes.img_size
        insts = InstsLabel(img_size=img_size)
        for box, maskcN in zip(boxes, masksN):
            cind_i = IndexCategory.convert(box.category).cindN if cind is None else cind
            maskN = maskcN[cind_i]
            conf_thres_i = conf_thres if conf_thres is not None else np.max(maskN) / 2
            xyxyN = XYXYBorder.convert(box.border).xyxyN
            xyxyN = np.round(xyxyN).astype(np.int32)
            size = list(xyxyN[2:4] - xyxyN[:2])
            maskN = cv2.resize(maskN, size, interpolation=resample)
            mask = RefValRegion(xyN=xyxyN[:2], maskN_ref=maskN, size=img_size, conf_thres=conf_thres_i)
            inst = InstItem(border=box.border, category=box.category, rgn=mask, **box)
            insts.append(inst)
        return insts

    @staticmethod
    def from_boxes_masksT_ref(boxes: BoxesLabel, masksT: torch.Tensor, conf_thres: float = 0.2,
                              resample=cv2.INTER_LANCZOS4, cind: int = None):
        masksN = masksT.detach().cpu().numpy()
        return InstsLabel.from_boxes_masksN_ref(boxes, masksN, conf_thres, resample, cind)

    @staticmethod
    def from_boxes_masksN_abs(boxes: BoxesLabel, masksN: np.ndarray, conf_thres: float = None, cind: int = None,
                              only_inner: bool = True):
        img_size = boxes.img_size
        insts = InstsLabel(img_size=img_size)
        for box in copy.deepcopy(boxes):
            cind_i = IndexCategory.convert(box.category).cindN if cind is None else cind
            maskN_abs = masksN[cind_i]
            ####################测试
            if only_inner:
                border = box.border_ref if hasattr(box, 'border_ref') else box.border
                border = XYWHABorder.convert(border)
                border.xywhaN[2:4] = np.ceil(border.xywhaN[2:4])
                maskN_abs = maskN_abs * border.maskNb.astype(np.float32)
            ####################测试
            conf_thres_i = conf_thres if conf_thres is not None else np.max(maskN_abs) / 2
            rgn = AbsValRegion(maskN_abs, conf_thres=conf_thres_i)
            inst = InstItem(border=box.border, rgn=rgn, category=box.category, **box)
            insts.append(inst)
        return insts

    @staticmethod
    def from_boxes_masksT_abs(boxes: BoxesLabel, masksT: torch.Tensor, conf_thres: float = 0.2,
                              cind: int = None, only_inner: bool = True):
        masksN = masksT.detach().cpu().numpy()
        return InstsLabel.from_boxes_masksN_abs(boxes, masksN, conf_thres, cind, only_inner)


class StereoBoxesLabel(ImageItemsLabel):
    pass


# </editor-fold>

# <editor-fold desc='便捷工具'>
# 恢复到原图标签大小
def labels_rescale(labels, imgs, ratios):
    assert len(labels) == len(ratios) and len(labels) == len(imgs), 'len err'
    ratios = np.array(ratios)
    if np.all(ratios == 1):
        return labels
    labels_scaled = []
    for label, img, ratio in zip(labels, imgs, ratios):
        label_scaled = copy.deepcopy(label)
        label_scaled.linear(scale=np.array([ratio, ratio]), size=img2size(img))
        labels_scaled.append(label_scaled)
    return labels_scaled


# </editor-fold>

# <editor-fold desc='标签和数据转化'>


import pydensecrf.densecrf as dcrf


def masksT_crf(imgT: torch.Tensor, masksT: torch.Tensor, sxy: int = 40, srgb: int = 10,
               num_infer: int = 2) -> torch.Tensor:
    C, H, W = masksT.size()

    d = dcrf.DenseCRF2D(W, H, C)
    masksT = -torch.log(masksT.clamp_(min=1e-8))
    maskN = masksT.detach().cpu().numpy()
    maskN = maskN.reshape(maskN.shape[0], -1)

    d.setUnaryEnergy(maskN)
    imgN = imgT2imgN(imgT)
    imgN = np.ascontiguousarray(imgN.astype(np.uint8))
    d.addPairwiseBilateral(sxy=(sxy, sxy), srgb=(srgb, srgb, srgb), rgbim=imgN, compat=10, kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(num_infer)
    qn = np.array(Q).reshape((C, H, W))
    masksT_crf = torch.from_numpy(qn).to(imgT.device)
    return masksT_crf


def masksN_crf(imgN: np.ndarray, masksN: np.ndarray, sxy: int = 40, srgb: int = 10,
               num_infer: int = 2) -> np.ndarray:
    H, W, C = masksN.shape
    d = dcrf.DenseCRF2D(W, H, C)
    masksN = -np.log(np.clip(masksN, a_min=1e-5, a_max=None))
    maskN = masksN.transpose((2, 0, 1))  # (C, H, W)
    maskN = np.ascontiguousarray(maskN.reshape(maskN.shape[0], -1))

    d.setUnaryEnergy(maskN.astype(np.float32))
    imgN = np.ascontiguousarray(imgN.astype(np.uint8))
    d.addPairwiseBilateral(sxy=(sxy, sxy), srgb=(srgb, srgb, srgb), rgbim=imgN, compat=10, kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(num_infer)
    qn = np.array(Q).reshape((C, H, W))
    qn = qn.transpose((1, 2, 0))  # (H,W,C)
    return qn


def masksT2masksNb_with_conf(masksT: torch.Tensor, conf_thres: float = None) -> np.ndarray:
    conf_thres_i = conf_thres if conf_thres is not None else torch.max(masksT) / 2
    masksTb = (masksT > conf_thres_i).bool()
    masksNb = masksTb.detach().cpu().numpy().astype(bool)
    if len(masksNb.shape) == 4 and masksNb.shape[0] == 1:
        masksNb = masksNb.squeeze(axis=0)
    masksNb = np.transpose(masksNb, (1, 2, 0))  # CHW转为HWC
    return masksNb

# </editor-fold>
