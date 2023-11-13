from abc import abstractmethod

import onnxruntime
import torch
import torch.nn as nn
from torch.onnx import OperatorExportTypes

from utils.file import *
from utils.pack import select_device


# 得到device
def get_device(model):
    if hasattr(model, 'device'):
        return model.device
    else:
        if len(model.state_dict()) > 0:
            return next(iter(model.parameters())).device
        else:
            return torch.device('cpu')


# 规范4维输入
def _get_input_size(input_size=(32, 32), default_channel=3, default_batch=1):
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


def model2onnx(model, onnx_pth, input_size, **kwargs):
    onnx_dir = os.path.dirname(onnx_pth)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_pth = onnx_pth + '.onnx' if not str.endswith(onnx_pth, '.onnx') else onnx_pth
    input_size = _get_input_size(input_size, default_batch=1, default_channel=3)
    # 仅支持单输入单输出
    input_names = ['input']
    output_names = ['output']
    dynamic_batch = input_size[0] is None or input_size[0] < 0
    if dynamic_batch:
        input_size = list(input_size)
        input_size[0] = 1
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        print('Using dynamic batch size')
    else:
        dynamic_axes = None
        print('Exporting static batch size')
    test_input = (torch.rand(size=input_size) - 0.5) * 4
    test_input = test_input.to(get_device(model))
    model.eval()
    print('Exporting onnx to ' + onnx_pth)
    torch.onnx.export(model, test_input, onnx_pth, verbose=True, opset_version=11,
                      operator_export_type=OperatorExportTypes.ONNX, do_constant_folding=True,
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    return True


class ONNXExportable(nn.Module):

    @property
    @abstractmethod
    def input_names(self):
        pass

    @property
    @abstractmethod
    def output_names(self):
        pass

    @property
    @abstractmethod
    def input_sizes(self):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def export_onnx(self, onnx_pth, dynamic_batch=False):
        onnx_pth = ensure_extend(onnx_pth, 'onnx')
        ensure_folder_pth(onnx_pth)
        input_names = self.input_names
        output_names = self.output_names
        input_sizes = self.input_sizes
        device = self.device
        input_sizes = [[1] + list(input_size) for input_size in input_sizes]
        input_tens = [torch.rand(input_size, device=device) for input_size in input_sizes]

        if dynamic_batch:
            dynamic_axes = {}
            for input_name in input_names:
                dynamic_axes[input_name] = {0: 'batch_size'}

            for output_name in output_names:
                dynamic_axes[output_name] = {0: 'batch_size'}
            print('Using dynamic batch size')
        else:
            dynamic_axes = None
            print('Exporting static batch size')

        print('Exporting onnx to ' + onnx_pth)
        torch.onnx.export(self, input_tens, onnx_pth, verbose=True, opset_version=11,
                          operator_export_type=OperatorExportTypes.ONNX, do_constant_folding=True,
                          input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
        return self


class SISOONNXExportable(ONNXExportable):

    @property
    def input_names(self):
        return ['input']

    @property
    def output_names(self):
        return ['output']


class ImageONNXExportable(ONNXExportable):

    @property
    def input_names(self):
        return ['image']

    @property
    @abstractmethod
    def img_size(self):
        pass

    @property
    @abstractmethod
    def in_channels(self):
        pass

    @property
    def input_sizes(self):
        return [(self.in_channels, self.img_size[1], self.img_size[0])]


class ONNXModule():
    def __init__(self, onnx_pth, device=None):
        onnx_pth = onnx_pth + '.onnx' if not str.endswith(onnx_pth, '.onnx') else onnx_pth
        device_ids = select_device(device)
        if device_ids[0] is None:
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_pth, providers=['CPUExecutionProvider'])
        else:
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_pth, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': str(device_ids[0])}])

    @property
    def output_names(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    @property
    def input_names(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name


class OneStageONNXModule(ONNXModule):
    def __init__(self, onnx_pth, device=None):
        super().__init__(onnx_pth=onnx_pth, device=device)
        inputs = self.onnx_session.get_inputs()
        outputs = self.onnx_session.get_outputs()
        assert len(inputs) == 1, 'fmt err'
        assert len(outputs) == 1, 'fmt err'
        self.input_size = inputs[0].shape
        self.output_size = outputs[0].shape
        print('ONNXModule from ' + onnx_pth + ' * input ' + str(inputs[0].shape) + ' * output ' + str(outputs[0].shape))

    def __call__(self, input, **kwargs):
        input_feed = {self.input_names[0]: input}
        outputs = self.onnx_session.run(self.output_names, input_feed=input_feed)
        output = outputs[0]
        return output
