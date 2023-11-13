import enum

import numpy as np
import torch
import torch.nn as nn

from utils.logger import print


class MATCH_TYPE(enum.Enum):
    FULL_NAME = 1
    LAST_NAME = 2
    SIZE = 4

    def __or__(self, other):
        if isinstance(other, MATCH_TYPE):
            other = other.value
        return self.value | other


# 简单按序匹配算法
def match(arr1, arr2, cert=None):
    if cert is None:
        cert = lambda x, y: x == y
    num1, num2 = len(arr1), len(arr2)
    match_mat = np.full(shape=(num1, num2), fill_value=False)
    for i in range(num1):
        for j in range(num2):
            match_mat[i, j] = cert(arr1[i], arr2[j])
    match_pairs = []
    for s in range(num1 + num2):
        for i in range(s, -1, -1):
            j = s - i
            if i >= num1 or j >= num2 or j < 0:
                continue
                # 查找匹配
            if match_mat[i, j]:
                match_pairs.append((i, j))
                match_mat[i, :] = False
                match_mat[:, j] = False
                # print('Fit ',i,' --- ', j)
    return match_pairs


# 根据character匹配state_dict
def match_state_dict(sd_tar, sd_ori, match_type=MATCH_TYPE.FULL_NAME, show_detial=False):
    names_tar = list(sd_tar.keys())
    names_ori = list(sd_ori.keys())
    tensors_tar = list(sd_tar.values())
    tensors_ori = list(sd_ori.values())
    arr_tar = [[] for _ in range(len(names_tar))]
    arr_ori = [[] for _ in range(len(names_ori))]
    cert = lambda x, y: x == y
    characters = ''
    if MATCH_TYPE.SIZE | match_type == match_type:
        characters += 'size '
        for i in range(len(arr_tar)):
            arr_tar[i].append(tensors_tar[i].size())
        for i in range(len(arr_ori)):
            arr_ori[i].append(tensors_ori[i].size())
    if MATCH_TYPE.FULL_NAME | match_type == match_type:
        characters += 'full_name '
        for i in range(len(arr_tar)):
            arr_tar[i].append(names_tar[i])
        for i in range(len(arr_ori)):
            arr_ori[i].append(names_ori[i])
    if MATCH_TYPE.LAST_NAME | match_type == match_type:
        characters += 'last_name '
        for i in range(len(arr_tar)):
            arr_tar[i].append(names_tar[i].split('.')[-1])
        for i in range(len(arr_ori)):
            arr_ori[i].append(names_ori[i].split('.')[-1])
    if len(characters) == 0:
        raise Exception('Unknown match type ' + str(match_type))
    print('Try to match by  [ ' + characters + ']')
    match_pairs = match(arr_tar, arr_ori, cert)
    if show_detial:
        print('[ Matched pairs ] ' + str(len(match_pairs)))
        unmatched_tar = np.full(fill_value=True, shape=len(sd_tar))
        unmatched_ori = np.full(fill_value=True, shape=len(sd_ori))
        for ind_tar, ind_ori in match_pairs:
            msg_ori = '%-50s' % names_ori[ind_ori] + ' %-20s' % str(tuple(tensors_ori[ind_ori].size()))
            msg_tar = '%-50s' % names_tar[ind_tar] + ' %-20s' % str(tuple(tensors_tar[ind_tar].size()))
            print(msg_ori + ' -> ' + msg_tar)
            unmatched_tar[ind_tar] = False
            unmatched_ori[ind_ori] = False
        if np.any(unmatched_tar):
            inds_tar = np.nonzero(unmatched_tar)[0]
            print('[ Unmatched target ] ' + str(len(inds_tar)))
            for ind_tar in inds_tar:
                print('%-50s' % names_tar[ind_tar] + ' %-20s' % str(tuple(tensors_tar[ind_tar].size())))
        if np.any(unmatched_ori):
            inds_ori = np.nonzero(unmatched_ori)[0]
            print('[ Unmatched source ] ' + str(len(inds_ori)))
            for ind_ori in inds_ori:
                print('%-50s' % names_ori[ind_ori] + ' %-20s' % str(tuple(tensors_ori[ind_ori].size())))
    return match_pairs


# 自定义state_dict加载
def load_fmt(model, sd_ori, match_type=MATCH_TYPE.FULL_NAME, only_fullmatch=False, show_detial=False, power=1.0):
    if isinstance(sd_ori, str):
        device = next(iter(model.parameters())).device
        sd_ori = torch.load(sd_ori, map_location=device)
    sd_tar = model.state_dict()
    names_tar = list(sd_tar.keys())
    tensors_ori = list(sd_ori.values())
    # #匹配
    fit_pairs = match_state_dict(sd_tar, sd_ori, match_type=match_type, show_detial=show_detial)
    # 检查匹配结果
    print(' Source %d' % len(sd_ori) + ' Target %d' % len(names_tar) +
          ' Match %d' % len(fit_pairs) + ' Power %.2f' % power)
    if only_fullmatch and len(fit_pairs) < len(names_tar):
        print('Not enough match')
        return False
    fit_sd = {}
    for i in range(len(fit_pairs)):
        i_tar, i_ori = fit_pairs[i]
        fit_sd[names_tar[i_tar]] = tensors_ori[i_ori]
    for name, tensor in sd_tar.items():
        if name not in fit_sd.keys():
            fit_sd[name] = tensor
    # 匹配添加
    power = max(min(power, 1), 0)
    for name, tensor in fit_sd.items():
        names = str.split(name, '.')
        tar = model
        for n in names:
            tar = getattr(tar, n)
        tar.data = power * tensor + (1 - power) * tar.data
    return True


# 调整模型通道显示
def refine_chans(model):
    def refine(model):
        if len(list(model.children())) == 0:
            if isinstance(model, nn.Conv2d):
                wei = model.weight
                model.in_channels = wei.size()[1]
                model.out_channels = wei.size()[0]
            elif isinstance(model, nn.BatchNorm2d):
                wei = model.weight
                model.num_features = wei.size()[0]
                model.running_mean = model.running_mean[:wei.size()[0]]
                model.running_var = model.running_var[:wei.size()[0]]
            elif isinstance(model, nn.Linear):
                wei = model.weight
                model.in_features = wei.size()[1]
                model.out_features = wei.size()[0]
        else:
            for name, sub_model in model.named_children():
                refine(sub_model)

    refine(model)
    return None
