import copy
import json
import os
import pickle
import time
import types
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


# <editor-fold desc='迭代器'>
# 显示进度的迭代器
class MEnumerate(Iterable):
    @staticmethod
    def calc_interval(total, num_rec=10):
        lev = (np.log(total) - np.log(num_rec)) / np.log(10)
        lev = int(np.round(lev * 2))
        interval = 10 ** (lev // 2) * 5 ** (lev % 2)
        return interval

    def __init__(self, seq, prefix='Iterating', suffix='', interval=None, broadcast=print, with_eta=True, total=None):
        self.seq = seq
        self.total = len(seq) if total is None else total
        self.ptr = 0
        self.prefix = prefix
        self.suffix = suffix
        self.broadcast = broadcast
        self.interval = interval if interval is not None else MEnumerate.calc_interval(self.total, num_rec=10)
        self.with_eta = with_eta

    def __iter__(self):
        self.ptr = 0
        self._core = iter(self.seq)
        self.time_start = time.time()
        return self

    def __next__(self):
        ptr = copy.deepcopy(self.ptr)
        val = next(self._core)
        if ptr % self.interval == 0 or (ptr + 1) == self.total:
            msg = '%8d' % (ptr + 1) + ' / %-8d' % self.total
            if self.with_eta:
                eta = calc_eta(index_cur=ptr, total=self.total, time_start=self.time_start)
                msg += ' ETA ' + sec2hour_min_sec(eta)
            self.broadcast(self.prefix + msg + self.suffix)
        self.ptr = self.ptr + 1
        return ptr, val


# eta计算
def calc_eta(index_cur, total, time_start):
    time_cur = time.time()
    sec = 0 if index_cur == 0 else \
        (time_cur - time_start) / (index_cur) * (total - index_cur)
    return sec


def sec2hour_min_sec(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


# </editor-fold>

# if __name__ == '__main__':
#     a = [1, 2, 3, 34, 234, ]*100
#     for i, r in MEnumerate(a):
#         pass

# <editor-fold desc='文件路径编辑'>

# 规范文件类型
def ensure_folder_pth(folder_pth):
    if not os.path.exists(folder_pth):
        os.makedirs(folder_pth)
    return folder_pth


def ensure_file_dir(file_pth):
    if not os.path.exists(os.path.dirname(file_pth)):
        os.makedirs(os.path.dirname(file_pth))
    return file_pth


def ensure_extend(file_pth, extend=''):
    extend = '.' + extend.replace('.', '')
    if file_pth.endswith(extend):
        return file_pth
    if len(file_pth) > 0:
        file_pth = os.path.splitext(file_pth)[0] + extend
    return file_pth


def listdir_extend(file_dir, extends=None):
    file_names = os.listdir(file_dir)
    if extends is None:
        return file_names
    elif isinstance(extends, str):
        return filter(lambda x: str.endswith(x, extends), file_names)
    elif isinstance(extends, list):
        return filter(lambda x: os.path.splitext(x)[1] in extends, file_names)
    raise Exception('err fmt ' + extends.__class__.__name__)


# </editor-fold>


def spec_cluster(A, n_cluster=3):
    from sklearn.cluster import KMeans
    # 计算L
    D = np.sum(A, axis=1)
    L = np.diag(D) - A
    sqD = np.diag(1.0 / (D ** (0.5)))
    L = np.dot(np.dot(sqD, L), sqD)
    # 特征值
    lam, H = np.linalg.eig(L)
    order = np.argsort(lam)
    order = order[:n_cluster]
    V = H[:, order]
    # 聚类
    res = KMeans(n_clusters=n_cluster).fit(V)
    lbs = res.labels_
    return lbs


# <editor-fold desc='多种文件读写'>


def _prepare_save(file_pth, extend=''):
    if not os.path.exists(os.path.dirname(file_pth)):
        os.makedirs(os.path.dirname(file_pth))
    file_pth = ensure_extend(file_pth, extend=extend)
    return file_pth


# 写xls
def save_np2xlsx(data, file_pth, extend='xlsx'):
    file_pth = _prepare_save(file_pth, extend=extend)
    df = pd.DataFrame(data, columns=None, index=None)
    df.to_excel(file_pth, index=False, header=False)
    return


# 读xls
def load_xlsx2np(file_pth, extend='xlsx'):
    file_pth = ensure_extend(file_pth, extend=extend)
    data = pd.read_excel(file_pth, header=None, sheet_name=None)
    data = np.array(data)
    return data


# 保存对象
def save_pkl(file_pth, obj, extend='pkl'):
    file_pth = _prepare_save(file_pth, extend=extend)
    with open(file_pth, 'wb+') as f:
        pickle.dump(obj, f)
    return None


# 读取对象
def load_pkl(file_pth, extend='pkl'):
    file_pth = ensure_extend(file_pth, extend=extend)
    with open(file_pth, 'rb+') as f:
        obj = pickle.load(f)
    return obj


# 保存当前工作区
IGNORE_NAMES = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__', '__builtins__']


def save_space(file_pth, locals, extend='pkl', show_detial=True):
    save_dict = {}
    for name, val in locals.items():
        if callable(val):
            continue
        if name in IGNORE_NAMES:
            continue
        if type(val) == types.ModuleType:
            continue
        if show_detial:
            print('saving', name, type(val))
        save_dict[name] = copy.copy(val)
    # 保存
    save_pkl(save_dict, file_pth, extend)
    return None


# 恢复工作区
def load_space(file_pth, locals, extend='pkl'):
    save_dict = load_pkl(file_pth, extend)
    for name, val in save_dict.items():
        locals[name] = val


def load_txt(file_pth, extend='txt', encoding='utf-8'):
    file_pth = ensure_extend(file_pth, extend=extend)
    with open(file_pth, 'r', encoding=encoding) as file:
        lines = file.readlines()
    lines = [line.replace('\n', '') for line in lines]
    return lines


def save_txt(file_pth, lines, extend='txt', encoding='utf-8'):
    file_pth = _prepare_save(file_pth, extend=extend)
    lines_enter = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1 and not line.endswith('\r\n'):
            line = line + '\r\n'
        lines_enter.append(line)
    with open(file_pth, 'w', encoding=encoding) as file:
        file.writelines(lines_enter)
    return None


def save_json(file_pth, dct, extend='json', indent=None, encoding='utf-8'):
    file_pth = _prepare_save(file_pth, extend=extend)
    with open(file_pth, 'w', encoding=encoding) as file:
        json.dump(dct, fp=file, indent=indent)
    return None


def load_json(file_pth, extend='json', encoding='utf-8'):
    file_pth = ensure_extend(file_pth, extend=extend)
    with open(file_pth, 'r', encoding=encoding) as file:
        dct = json.load(file)
    return dct


def save_yaml(file_pth, dct, extend='yaml', indent=None, encoding='utf-8'):
    file_pth = _prepare_save(file_pth, extend=extend)
    with open(file_pth, 'w', encoding=encoding) as file:
        yaml.dump(dct, fp=file, indent=indent)
    return None


def load_yaml(file_pth, extend='yaml', encoding='utf-8'):
    file_pth = ensure_extend(file_pth, extend=extend)
    with open(file_pth, 'r', encoding=encoding) as file:
        dct = yaml.safe_load(file)
    return dct


# </editor-fold>


if __name__ == '__main__':
    save_pth = '/ses-data/JD/cache_dota/train0.pkl'
    b = load_pkl(save_pth)
