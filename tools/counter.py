import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import time


#######################################################计算用时
def calc_delay(model, input_size=(32, 32), iter=1000):
    test_x = torch.rand(1, 3, input_size[0], input_size[1])
    test_x.to(model.device)
    torch.cuda.synchronize()
    time_start = time.time()
    for i in range(iter):
        _ = model(test_x)
    torch.cuda.synchronize()
    time_end = time.time()
    total_time = time_end - time_start
    delay = total_time / iter
    return delay


#######################################################计算FLOP
# 统计不为0的通道
def num_not_zero(data, dim=0):
    data = data.detach().cpu().numpy()
    if not dim == 0:
        data = data.swapaxes(dim, 0)

    sum_val = np.sum(np.abs(data.reshape(data.shape[0], -1)), axis=1)
    num_out = np.sum(sum_val > 1e-7)
    return num_out


# 计算有关层的hook
def hook_Conv2d(module, data_input, data_output, flop_list, ignore_zero=False, **msg):
    _, _, fw, fh = data_output.shape
    co, ci, kw, kh = module.weight.shape
    if not ignore_zero:
        ci = num_not_zero(data_input[0], dim=1)
        co = num_not_zero(module.weight, dim=0)

    flop = (kw * kh * ci) * co * fw * fh
    para = co * ci * kw * kh
    if module.bias is not None:
        flop += co * fw * fh
        para += co
    msg['size'] = [co, ci, kw, kh]
    msg['flop'] = flop
    msg['para'] = para
    flop_list.append(msg)
    return


def hook_AvgPool(module, data_input, data_output, flop_list, ignore_zero=False, **msg):
    _, c, fwi, fhi = data_input[0].shape
    _, _, fwo, fho = data_output.shape
    if not ignore_zero:
        c = num_not_zero(data_output, dim=1)
    flop = (1 + (fwi // fwo) * (fhi // fho)) * fwo * fho * c
    msg['flop'] = flop
    msg['para'] = 0
    msg['size'] = [c]
    flop_list.append(msg)
    return


def hook_Linear(module, data_input, data_output, flop_list, ignore_zero=False, **msg):
    co, ci = module.weight.shape
    if not ignore_zero:
        ci = num_not_zero(data_input[0], dim=1)
        co = num_not_zero(data_output, dim=1)

    flop = (ci) * co
    para = (ci) * co
    if not module.bias is None:
        flop += co
        para += co
    msg['flop'] = flop
    msg['para'] = para
    msg['size'] = [co, ci]
    flop_list.append(msg)
    return


def hook_BatchNorm2d(module, data_input, data_output, flop_list, ignore_zero=False, **msg):
    _, c, fw, fh = data_input[0].shape
    if not ignore_zero:
        c = num_not_zero(module.weight.data)
    flop = c * fw * fh * 2
    msg['flop'] = flop
    msg['para'] = c * 2
    msg['size'] = [c]
    flop_list.append(msg)
    return


# hook映射表
hook_dict = {
    nn.Conv2d: hook_Conv2d,
    # nn.AvgPool2d: hook_AvgPool,
    nn.AvgPool2d: None,
    # nn.AdaptiveAvgPool2d: hook_AvgPool,
    nn.AdaptiveAvgPool2d: None,
    nn.Linear: hook_Linear,
    nn.BatchNorm2d: hook_BatchNorm2d,
    nn.ReLU: None,
    nn.MaxPool2d: None,
    nn.Dropout: None,
    nn.Dropout2d: None,
    nn.CrossEntropyLoss: None
}


# 统计模型信息与计算量
def stat_model(model, input_size=(32, 32), ignore_zero=False):
    msg_list = []
    handles = []

    # hook映射
    def get_hook(model, model_name=None):
        type_m = type(model)
        if type_m in hook_dict.keys():
            func = hook_dict[type_m]
            if func is None:
                return None
            else:
                msg = {
                    'model_name': model_name,
                    'cls_name': model.__class__.__name__
                }
                # func=functools.partial(func,msg_list=msg_list,ignore_zero=ignore_zero,msg=*msg)
                # return func
                return lambda m, x, y: func(m, x, y, msg_list, ignore_zero, **msg)
        else:
            print('not support type', model.__class__.__name__)
            return None

    # 添加hook
    def add_hook(model, model_name=None):
        if len(list(model.children())) == 0:
            hook = get_hook(model, model_name)
            if hook is None:
                return
            else:
                handle = model.register_forward_hook(hook)
                handles.append(handle)
        else:
            for name, sub_model in model.named_children():
                sub_model_name = name if model_name is None else model_name + '.' + name
                add_hook(sub_model, sub_model_name)

    # 规范输入
    if isinstance(input_size, int):
        test_x = torch.rand(size=(5, 3, input_size, input_size)) * 3
    elif len(input_size) == 2:
        test_x = torch.rand(size=(5, 3, input_size[0], input_size[1])) * 3
    elif len(input_size) == 3:
        test_x = torch.rand(size=(5, input_size[0], input_size[1], input_size[2])) * 3
    elif len(input_size) == 4:
        test_x = torch.rand(size=input_size) * 3
    else:
        print('err size')
        return 0
    # 传播
    test_x = (test_x - 0.5) * 4
    # for para in model.parameters():
    #     print(para.device)
    # para=next(iter(model.parameters()))
    # device=para.device
    device = model.device
    test_x = test_x.to(device)
    #
    model = model.eval()
    add_hook(model, None)
    _ = model(test_x)
    # 移除hook
    for handle in handles:
        handle.remove()
    # 转化为pd
    order = ['model_name', 'cls_name', 'size', 'flop', 'para']
    data = pd.DataFrame(columns=order)
    for msg in msg_list:
        data = data.append(msg, ignore_index=True)
    return data


#######################################################自定义方法计算
def calc_flop_para(model, input_size=(32, 32), ignore_zero=False):
    data = stat_model(model, input_size=input_size, ignore_zero=ignore_zero)
    print('-' * 70)
    convs = data['cls_name'] == 'Conv2d'
    lins = data['cls_name'] == 'Linear'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    pd.set_option('display.max_colwidth', 500)
    pd.set_option('display.max_rows', None)
    print(data.loc[convs | lins])
    flop = np.sum(np.array(data['flop']))
    para = np.sum(np.array(data['para']))
    print('-' * 20 + ' ' * 5 + 'flop', flop, ' ' * 5 + 'para', para, ' ' * 5 + '-' * 20)
    return flop, para
