import collections
import inspect
import sys
import time
from functools import partial

import PIL.Image as Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from utils.logger import print


# <editor-fold desc='数据集分析'>
# </editor-fold>
# 检查数据分布
def count_distribute(arr, quant_step=None, a_min=None, a_max=None, with_unit=True, with_terminal=True):
    arr = np.array(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return 0, 0
    quant_step = quant_step if quant_step is not None else np.std(arr) / 5
    assert quant_step > 0
    a_min = a_min if a_min is not None else min_val
    a_max = a_max if a_max is not None else max_val

    num_quant = int(np.ceil((a_max - a_min) / quant_step))

    quants = (np.arange(num_quant) + 0.5) * quant_step + a_min
    indexs = np.floor((arr - a_min) / quant_step)
    fliter = (indexs >= 0) * (indexs < num_quant)
    indexs, nums_sub = np.unique(indexs[fliter], return_counts=True)
    nums = np.zeros(shape=num_quant)
    nums[indexs.astype(np.int32)] = nums_sub
    # 添加端部节点
    if with_terminal:
        quants = np.concatenate([[a_min], quants, [a_max]])
        nums = np.concatenate([[0], nums, [0]])
    # 归一化
    if with_unit:
        nums = nums / np.sum(nums) / quant_step
    return quants, nums


# <editor-fold desc='常用层FLOPs Para统计函数'>
class HOOK():
    # 统计不为0的通道
    @staticmethod
    def count_nonzero(data, dim=0):
        if not dim == 0:
            data = data.transpose(dim, 0).contiguous()
        sum_val = torch.sum(torch.abs(data.reshape(data.shape[0], -1)), dim=1)
        num = torch.count_nonzero(sum_val).item()
        return num

    @staticmethod
    def Conv2d(module, data_input, data_output, ignore_zero, handler):
        _, _, fw, fh = list(data_output.size())
        co, ci, kw, kh = list(module.weight.size())
        if not ignore_zero:
            ci = HOOK.count_nonzero(data_input[0], dim=1)
            co = HOOK.count_nonzero(module.weight, dim=0)
        flop = (kw * kh * ci) * co * fw * fh
        if module.bias is not None:
            flop += co * fw * fh
        size = (co, ci, kw, kh)
        handler(size=size, flop=flop)

    @staticmethod
    def AvgPool(module, data_input, data_output, ignore_zero, handler):
        _, c, fwi, fhi = data_input[0].shape
        _, _, fwo, fho = data_output.shape
        if not ignore_zero:
            c = HOOK.count_nonzero(data_output, dim=1)
        flop = (1 + (fwi // fwo) * (fhi // fho)) * fwo * fho * c
        handler(size=[c], flop=flop)

    @staticmethod
    def Linear(module, data_input, data_output, ignore_zero, handler):
        co, ci = module.weight.shape
        if not ignore_zero:
            ci = HOOK.count_nonzero(data_input[0], dim=1)
            co = HOOK.count_nonzero(data_output, dim=1)
        flop = (ci) * co
        if not module.bias is None:
            flop += co
        handler(size=[co, ci], flop=flop)

    @staticmethod
    def BatchNorm2d(module, data_input, data_output, ignore_zero, handler):
        _, c, fw, fh = data_input[0].shape
        if not ignore_zero:
            c = HOOK.count_nonzero(module.weight.data)
        flop = c * fw * fh * 2
        handler(size=[c], flop=flop)

    MAPPER = {
        nn.SiLU: None,
        nn.Conv2d: Conv2d,
        # nn.AvgPool2d: hook_AvgPool,
        nn.AvgPool2d: None,
        # nn.AdaptiveAvgPool2d: hook_AvgPool,
        nn.AdaptiveAvgPool2d: None,
        nn.Linear: Linear,
        nn.BatchNorm2d: BatchNorm2d,
        nn.ReLU: None,
        nn.MaxPool2d: None,
        nn.Dropout: None,
        nn.Dropout2d: None,
        nn.CrossEntropyLoss: None,
        nn.UpsamplingNearest2d: None,
        nn.LeakyReLU: None,
        # nn.Mish: None,
        nn.UpsamplingBilinear2d: None
    }

    IGNORE = [
        'SiLU',
        'Focus'
    ]

    @staticmethod
    def get_hook(model, handler, ignore_zero):
        type_m = type(model)
        if type_m in HOOK.MAPPER.keys():
            hook = HOOK.MAPPER[type_m]
            return partial(hook.__func__, ignore_zero=ignore_zero, handler=handler) if hook is not None else None
        elif model.__class__.__name__ in HOOK.IGNORE:
            return None
        else:
            print('Not support type ' + model.__class__.__name__)
            return None


# </editor-fold>


# <editor-fold desc='统计模型整体FLOPs与Para'>
# 规范4维输入
def input_size_fmt(input_size=(32, 32), default_channel=3, default_batch=1):
    if isinstance(input_size, int):
        input_size = (default_batch, default_channel, input_size, input_size)
    elif len(input_size) == 1:
        input_size = (default_batch, default_channel, input_size[0], input_size[0])
    elif len(input_size) == 2:
        input_size = (default_batch, default_channel, input_size[0], input_size[1])
    elif len(input_size) == 3:
        input_size = (default_batch, input_size[0], input_size[1], input_size[2])
    elif len(input_size) == 4:
        pass
    else:
        raise Exception('err size ' + str(input_size))
    return input_size


def count_fn_delay(fn, args, iter_num=100):
    time_start = time.time()
    for i in range(iter_num):
        _ = fn(args)
    time_end = time.time()
    total_time = time_end - time_start
    delay = total_time / iter_num
    return delay


def count_model_delay(model, input_size=(32, 32), iter_num=1000):
    input_size = input_size_fmt(input_size, default_batch=1, default_channel=3)
    test_input = torch.rand(size=input_size)
    if isinstance(model, nn.Module):
        test_input = test_input.to(next(iter(model.parameters())).device)
        with torch.no_grad():
            delay = count_fn_delay(fn=model, iter_num=iter_num, args=test_input)
    else:
        test_input = np.zeros(shape=input_size)
        delay = count_fn_delay(fn=model, iter_num=iter_num, args=test_input)
    return delay


def count_flop(model, input_size=(32, 32), ignore_zero=False):
    assert isinstance(model, nn.Module), 'mdoel err'
    msgs = collections.OrderedDict()
    hooks = []

    def handler(module_name, cls_name, size, flop):
        if module_name not in msgs.keys():
            msgs[module_name] = {
                'Name': module_name,
                'Class': cls_name,
                'Size': str(size),
                'FLOP': flop
            }
        else:
            msgs[module_name]['FLOP'] += flop
        return None

    # 添加hook
    def add_hook(module, module_name=None):
        if len(list(module.children())) == 0:
            handler_m = partial(handler, module_name=module_name, cls_name=module.__class__.__name__)
            hook = HOOK.get_hook(module, handler_m, ignore_zero)
            if hook is not None:
                handle = module.register_forward_hook(hook)
                hooks.append(handle)
        else:
            for name, sub_model in module.named_children():
                sub_model_name = name if module_name is None else module_name + '.' + name
                add_hook(sub_model, sub_model_name)

    # 规范输入
    input_size = input_size_fmt(input_size, default_channel=3, default_batch=1)
    test_input = (torch.rand(size=input_size) - 0.5) * 4
    test_input = test_input.to(next(iter(model.parameters())).device)
    model = model.eval()
    with torch.no_grad():
        add_hook(model, None)
        _ = model(test_input)
    # 移除hook
    for hook in hooks:
        hook.remove()
    order = ['Name', 'Class', 'Size', 'FLOP']
    data = pd.DataFrame(columns=order)
    for msg in msgs.values():
        data = pd.concat([data, pd.DataFrame(msg, index=[0])])
    return data


def count_para(model):
    assert isinstance(model, nn.Module), 'mdoel err ' + model.__class__.__name__
    count_para.data = pd.DataFrame(columns=['Name', 'Class', 'Para'])

    def stat_para(module, module_name=None):
        if len(list(module.children())) == 0:
            para_sum = 0
            for para in module.parameters():
                para_sum += para.numel()
            row = pd.DataFrame({
                'Name': module_name,
                'Class': module.__class__.__name__,
                'Para': para_sum
            }, index=[0])
            count_para.data = pd.concat([count_para.data, row])
        else:
            for name, sub_model in module.named_children():
                sub_model_name = name if module_name is None else module_name + '.' + name
                stat_para(sub_model, module_name=sub_model_name)

    stat_para(model, None)
    data = count_para.data
    return data


def analyse_cens(whs, centers, whr_thres=4):
    n_clusters = centers.shape[0]
    ratios = whs[:, None, :] / centers[None, :, :]
    ratios = np.max(np.maximum(ratios, 1 / ratios), axis=2)
    markers = ratios < whr_thres
    # 输出
    print('* Centers --------------')
    for i in range(n_clusters):
        width, height = centers[i, :]
        print('[ %5d' % int(width) + ' , %5d' % int(height) + ' ] --- ' + str(np.sum(markers[:, i])))
    print('* Boxes --------------')
    matched = np.sum(np.sum(markers, axis=1) > 0)
    print('Matched ' + '%5d' % int(matched) + ' / %5d' % int(whs.shape[0]))
    print('* -----------------------')
    return True


def cluster_wh(whs, n_clusters=9, log_metric=True):
    if log_metric:
        whs_log = np.log(whs)
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(whs_log)
        centers_log = kmeans_model.cluster_centers_
        centers = np.exp(centers_log)
    else:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(whs)
        centers = kmeans_model.cluster_centers_
    areas = centers[:, 0] * centers[:, 1]
    order = np.argsort(areas)
    centers_sorted = centers[order]
    return centers_sorted


def count_loader_delay(loader, iter_num=10):
    time_start = time.time()
    i = 0
    for _ in iter(loader):
        i = i + 1
        if i == iter_num:
            break
    time_end = time.time()
    delay = (time_end - time_start) / i
    return delay


# </editor-fold>

# <editor-fold desc='计算对象占用空间'>
NUMMB = 1048576
NUMKB = 1024


# # 计算内存占用
def get_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, Image.Image):
        size += sys.getsizeof(obj.tobytes())
        return size
    elif isinstance(obj, np.ndarray):
        size += obj.dtype.itemsize * obj.size
        return size
    elif isinstance(obj, torch.Tensor):
        size += obj.element_size() * torch.numel(obj)
        return size

    if isinstance(obj, dict):
        for v in obj.values():
            if not isinstance(v, (str, int, float, bytes, bytearray)):
                size += get_size(v, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        for v in obj:
            if not isinstance(v, (str, int, float, bytes, bytearray)):
                size += get_size(v, seen)

    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                dct = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(dct) or inspect.ismemberdescriptor(dct):
                    size += get_size(obj.__dict__, seen)
                break

    if hasattr(obj, '__slots__'):
        for s in obj.__slots__:
            size += get_size(getattr(obj, s), seen)

    return size


# </editor-fold>

if __name__ == '__main__':
    save_pth = '/ses-data/JD/cache_dota/train0.pkl'
