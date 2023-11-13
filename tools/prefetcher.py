import copy
import threading
import time
from abc import abstractmethod, ABCMeta

import torch


class Prefetcher():
    def __init__(self, loader, device, cycle=False):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device) if device.index is not None else None
        self.loader_iter = None
        self.cycle = cycle

    @property
    def sampler(self):
        return self.loader.sampler

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.loader_iter is None:
            self.loader_iter = iter(self.loader)
            self.thread = threading.Thread(target=self.load, daemon=True)
            self.thread.start()
        return self

    def __next__(self):
        self.thread.join()
        if self.imgs is None:
            if self.cycle:
                self.loader_iter = iter(self.loader)
                self.thread = threading.Thread(target=self.load, daemon=True)
                self.thread.start()
            else:
                self.loader_iter = None
            raise StopIteration
        else:
            imgs, labels = self.imgs, self.labels
            if isinstance(imgs,torch.Tensor):
                imgs = imgs.to(device=self.device)
            self.thread = threading.Thread(target=self.load, daemon=True)
            self.thread.start()
            return imgs, labels

    def load(self):
        try:
            self.imgs, self.labels = next(self.loader_iter)
            if self.stream is not None and isinstance(self.imgs,torch.Tensor):
                with torch.cuda.stream(self.stream):
                    self.imgs = self.imgs.to(device=self.device,non_blocking=True)
        except StopIteration:
            self.imgs, self.labels = None, None
        return None


class Prefetcher2():
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.loader_iter = None
        self.loader_iter_next = None
        self.thread = None

    @property
    def sampler(self):
        return self.loader.sampler

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.loader_iter_next is not None:
            self.loader_iter = self.loader_iter_next
            self.loader_iter_next = None
        else:
            self.loader_iter = iter(self.loader)
        self.thread = threading.Thread(target=self.load, daemon=True)
        self.thread.start()
        return self

    def __next__(self):
        return next(self.loader_iter)

    def load(self):
        self.loader_iter_next = iter(copy.deepcopy(self.loader))
        return None


class VirtualLoader(metaclass=ABCMeta):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class SimuLoader(VirtualLoader):
    def __init__(self, size=10, delay_next=0.2, delay_iter=1.0):
        self.size = size
        self.delay_next = delay_next
        self.delay_iter = delay_iter
        self.ptr = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        print('Build iter')
        time.sleep(self.delay_iter)
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr == self.size:
            raise StopIteration
        else:
            print('Fetching ', self.ptr)
            time.sleep(self.delay_next)
            self.ptr = self.ptr + 1
            imgs = torch.zeros(size=(1,))
            labels = []
            return imgs, labels


class SingleSampleLoader(VirtualLoader):
    def __init__(self, imgs, targets, total_iter=10):
        super().__init__()
        self.imgs = imgs
        self.targets = targets
        self.total_iter = total_iter
        self.ptr = 0
        self.batch_size = len(imgs)

    def __len__(self):
        return self.total_iter

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.total_iter:
            raise StopIteration
        else:
            self.ptr = self.ptr + 1
            return self.imgs, self.targets


class TargetLoader(VirtualLoader):
    def __init__(self, processor, loader):
        super().__init__()
        self.processor = processor
        self.loader = loader

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def batch_size(self):
        return self.loader.batch_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        return self

    def __next__(self):
        # time1 = time.time()
        imgs, labels = next(self.loader_iter)
        # time2 = time.time()
        target = self.processor(labels)
        # time3 = time.time()
        # print(time2-time1,time3-time2)
        return imgs, target


if __name__ == '__main__':
    loader = SimuLoader(delay_next=0.2, size=10)
    loader = Prefetcher2(loader=loader, device=torch.device('cuda:0'))
    for n in range(2):
        for imgs, labels in loader:
            pass
