# from utils.logger import print
import gc
import logging
import sys
from logging.handlers import TimedRotatingFileHandler

import torch

from tools.prefetcher import Prefetcher, SingleSampleLoader, TargetLoader
from tools.scheduler import LRScheduler, IMScheduler
from utils.file import *
from utils.frame import TorchModel, Loader


# <editor-fold desc='检查'>
# 检查梯度
def check_grad(model, loader, accu_step=1, grad_norm=0, **kwargs):
    model.train()
    print('Checking Grad')
    loader_iter = iter(loader)
    for i in range(accu_step):
        (imgs, labels) = next(loader_iter)
        target = model.labels2tars(labels, **kwargs)
        loss = model.imgs_tars2loss(imgs, target, **kwargs)
        loss, losses, names = Trainner.process_loss(loss)
        print('Loss ', ''.join([n + ' %-10.5f  ' % l for l, n in zip(losses, names)]))
        (loss / accu_step).backward()
    if grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_norm)
    for name, para in model.named_parameters():
        print('%-50s' % name + '%10.5f' % para.grad.norm().item())
    return None


# 检查参数
def check_para(model):
    print('Checking Para')
    for name, para in model.named_parameters():
        if torch.any(torch.isnan(para)):
            print('nan occur in models')
            para.data = torch.where(torch.isnan(para), torch.full_like(para, 0.1), para)
        if torch.any(torch.isinf(para)):
            print('inf occur in models')
            para.data = torch.where(torch.isinf(para), torch.full_like(para, 0.1), para)
        max = torch.max(para).item()
        min = torch.min(para).item()
        print('Range [ %10.5f' % min + ' , ' + '%10.5f' % max + ']  --- ' + name)
    return None


# </editor-fold>


# <editor-fold desc='trainner'>
class Trainner():
    FORMATTER = logging.Formatter(fmt='%(asctime)s [%(funcName)s] : %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    def __init__(self, name='trainer', main_proc=True):
        self.main_proc = main_proc
        self.logger = logging.getLogger(name=name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.parent = None
        if main_proc:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            handler.setFormatter(Trainner.FORMATTER)
            self.logger.addHandler(handler)
        self.broadcast = self.logger.info

    def bind_pth(self, log_pth, new_log=True):
        if len(log_pth) == 0:
            return None
        if not os.path.exists(os.path.dirname(log_pth)):
            os.makedirs(os.path.dirname(log_pth))
        # if new_log and os.path.isfile(log_pth):
        #     os.remove(log_pth)
        handler = TimedRotatingFileHandler(log_pth, when='D', encoding='utf-8')
        handler.setLevel(logging.INFO)
        handler.setFormatter(Trainner.FORMATTER)
        if handler not in self.logger.handlers:
            self.logger.addHandler(handler)
        return handler

    @staticmethod
    def get_pths(load_pth):
        load_pth_pure, extend = os.path.splitext(load_pth)
        dir_name = os.path.dirname(load_pth)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        log_pth = load_pth_pure + '.log'
        record_pth = load_pth_pure + '.xlsx'
        wei_pth = load_pth_pure + '.pth'
        opt_pth = load_pth_pure + '.opt'
        msg_pth = load_pth_pure + '.json'
        return wei_pth, opt_pth, record_pth, log_pth, msg_pth

    @staticmethod
    def process_loss(loss):
        if isinstance(loss, dict):
            losses, names = list(loss.values()), list(loss.keys())
            for i, name in enumerate(names):
                loss_i = losses[i]
                if torch.isnan(loss_i):
                    print('nan in loss ' + str(name))
                    losses[i] = 0
                if torch.isinf(loss_i):
                    print('inf in loss ' + str(name))
                    losses[i] = 0
            loss = sum(losses)
            losses = [l.item() for l in losses]
            losses.insert(0, loss.item())
            names.insert(0, 'Loss')
        elif isinstance(loss, torch.Tensor):
            assert not torch.isnan(loss), 'nan in loss'
            assert not torch.isinf(loss), 'inf in loss'
            losses = [loss.item()]
            names = ['Loss']
        else:
            raise Exception('err loss')
        return loss, losses, names

    @staticmethod
    def sec2hour_min_sec(sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)


class OneStageTrainer(Trainner):
    def __init__(self, model, optimizer='sgd', metric=None, best_pfrmce=0, weight_decay=1e-4, interval=50,
                 name='trainer', main_proc=True):
        super().__init__(name=name, main_proc=main_proc)

        self.interval = interval
        # model
        self.model = model
        assert isinstance(model, TorchModel), 'model err' + model.__class__.__name__

        # optimizer
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), momentum=0.9,
                                        weight_decay=weight_decay, lr=0.001)
        self.optimizer = optimizer
        assert isinstance(optimizer, torch.optim.Optimizer), 'optimizer err ' + optimizer.__class__.__name__

        # metric
        if metric is not None:
            metric.broadcast = self.logger.info
        self.metric = metric

        # check_point
        self.record = pd.DataFrame()
        self.epoch_ind = 0
        if self.metric is not None and best_pfrmce < 0:
            self.logger.info('Perform initial test')
            best_pfrmce = self.metric(self.model)
        self.best_pfrmce = best_pfrmce

    # <editor-fold desc='训练相关'>
    # 训练一轮
    @staticmethod
    def _train_epoch(model, loader_tar, optimizer, lrscheduler=None, broadcast=lambda x: x, accu_step=1, grad_norm=0,
                     interval=50, **kwargs):
        record = pd.DataFrame()
        model.train()
        last_time = time.time()
        num_batch = len(loader_tar)
        for iter_ind, (imgs, targets) in enumerate(loader_tar):
            data_time = time.time()
            # loss
            loss = model.imgs_tars2loss(imgs, targets, **kwargs)
            loss, losses, lnames = Trainner.process_loss(loss)
            fwd_time = time.time()
            (loss / accu_step).backward()
            bkwd_time = time.time()
            # lr
            if lrscheduler is not None:
                lrscheduler.modify_optimizer(optimizer)
                lrscheduler.step()
            # accu
            if (iter_ind + 1) % accu_step == 0:
                if grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            opt_time = time.time()
            # record
            lr = optimizer.param_groups[0]['lr']
            times = np.array([last_time, data_time, fwd_time, bkwd_time, opt_time])
            times = times[1:] - times[:-1]
            times = [opt_time - last_time] + list(times)
            tnames = ['Time', 'data', 'fwd', 'bkwd', 'opt']
            columns = ['Iter', 'Lr'] + lnames + tnames
            vals = [iter_ind + 1, lr] + losses + times
            row = pd.DataFrame([dict(zip(columns, vals))], columns=columns)
            record = pd.concat([record, row])
            # show
            if iter_ind % interval == 0 or iter_ind + 1 == num_batch:
                vals_aver = np.average(record.iloc[-interval:, :], axis=0)
                div = 2 + len(lnames)
                broadcast(
                    columns[0] + ' %-5d' % (iter_ind + 1) +
                    ' | ' + columns[1] + ' %-8.7f ' % vals_aver[1] +
                    ' | ' + ''.join([columns[k] + ' %-8.5f ' % vals_aver[k] for k in range(2, div)]) +
                    ' | ' + ''.join(
                        [columns[k] + ' %-6.5f ' % vals_aver[k] for k in range(div, len(columns))]) + '|'
                )
            last_time = opt_time
        torch.cuda.empty_cache()
        gc.collect()
        return record

    # 训练单例
    def train_single(self, imgs, labels, lrscheduler, accu_step=1, grad_norm=0, **kwargs):
        # lrscheduler
        lrscheduler.iter_per_epoch = 1
        # loader
        targets = self.model.labels2tars(labels)
        loader_tar = SingleSampleLoader(targets=targets, imgs=imgs, total_iter=lrscheduler.total_epoch)
        # loader = Prefetcher(loader, self.model.device) if self.model.device.index is not None else loader
        record_epoch = OneStageTrainer._train_epoch(model=self.model, loader_tar=loader_tar, accu_step=accu_step,
                                                    optimizer=self.optimizer, broadcast=self.logger.info,
                                                    interval=self.interval,
                                                    lrscheduler=lrscheduler, grad_norm=grad_norm, **kwargs)
        return record_epoch

    @staticmethod
    def process_pth(ori_pth='', extends=('xlsx', 'pth', 'opt', 'json')):
        if len(ori_pth) == 0:
            return tuple([''] * len(extends))
        if not os.path.exists(os.path.dirname(ori_pth)):
            os.makedirs(os.path.dirname(ori_pth))
        ori_pth_pure = os.path.splitext(ori_pth)[0]
        prcd_pths = []
        for extend in extends:
            prcd_pths.append(ori_pth_pure + '.' + extend.replace('.', ''))
        return tuple(prcd_pths)

    # 训练多轮
    def train(self, loader, lrscheduler, imscheduler=None, accu_step=1, test_step=10, save_step=10, save_pth='',
              new_proc=True, grad_norm=0, **kwargs):
        save_pth = save_pth if self.main_proc else ''
        log_pth = OneStageTrainer.process_pth(save_pth, extends=('.log',))[0]
        self.bind_pth(log_pth=log_pth, new_log=new_proc)
        broadcast = self.logger.info if self.main_proc else lambda x: x
        record = self._train(model=self.model, loader=loader, optimizer=self.optimizer, lrscheduler=lrscheduler,
                             imscheduler=imscheduler, metric=self.metric, accu_step=accu_step, test_step=test_step,
                             save_step=save_step, save_pth=save_pth, new_proc=new_proc, grad_norm=grad_norm,
                             broadcast=broadcast, interval=self.interval, **kwargs)
        return record

    @staticmethod
    def _train(model, loader, optimizer, lrscheduler, imscheduler=None, metric=None, accu_step=1, test_step=10,
               save_step=10, save_pth='', new_proc=True, grad_norm=0, broadcast=lambda x: x, interval=50, **kwargs):
        # loader
        assert isinstance(loader, Loader), 'loader err' + loader.__class__.__name__
        loader_tar = TargetLoader(processor=model.labels2tars, loader=loader)
        loader_tar = Prefetcher(loader_tar, model.device, cycle=True) \
            if model.device.index is not None else loader_tar
        # lrscheduler
        lrscheduler = LRScheduler.convert(lrscheduler)
        lrscheduler.iter_per_epoch = len(loader)
        lrscheduler.modify_optimizer(optimizer)

        # imscheduler
        if imscheduler is not None:
            imscheduler = IMScheduler.convert(imscheduler)
            imscheduler.total_epoch = lrscheduler.total_epoch

        # test
        with_test = test_step > 0 and metric is not None
        with_save = save_step > 0 and len(save_pth) > 0

        # save_pth
        record_pth, wei_pth, opt_pth, msg_pth = OneStageTrainer.process_pth(
            ori_pth=save_pth, extends=('xlsx', 'pth', 'opt', 'json'))

        # load
        epoch_ind_last = 0
        best_pfrmce = 0
        record = pd.DataFrame()
        if not new_proc and len(save_pth) > 0:
            broadcast('Continue the previous training process')
            if os.path.exists(wei_pth):
                broadcast('Load weight from ' + wei_pth)
                model.load(wei_pth)
            if os.path.exists(opt_pth):
                broadcast('Load optimizer from ' + opt_pth)
                optimizer.load_state_dict(torch.load(opt_pth))
            if os.path.exists(record_pth):
                record = pd.read_excel(record_pth)
            if os.path.exists(msg_pth):
                msg = load_json(msg_pth)
                best_pfrmce = msg['best_pfrmce']
                epoch_ind_last = msg['epoch_ind']
        else:
            broadcast('Start a new training process')
        lrscheduler.set_epoch(epoch_ind_last)
        # train
        time_start = time.time()
        for epoch_ind in range(epoch_ind_last, lrscheduler.total_epoch):
            img_size = imscheduler[epoch_ind]
            loader.img_size = img_size
            model.img_size = img_size
            if metric is not None:
                metric.img_size = img_size
            # show
            eta = calc_eta(index_cur=epoch_ind - epoch_ind_last, total=lrscheduler.total_epoch - epoch_ind_last,
                           time_start=time_start)
            epoch_msg = '< Train > Epoch %d' % (epoch_ind + 1) + '  Length %d' % len(loader) + \
                        '  Batch %d' % loader.batch_size + '[%d]' % accu_step + \
                        '  ImgSize ' + str(loader.img_size) + \
                        '  ETA ' + sec2hour_min_sec(eta)
            broadcast(epoch_msg)
            if isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
                loader.sampler.set_epoch(epoch_ind)

            # train_epoch
            record_epoch = OneStageTrainer._train_epoch(
                model=model, loader_tar=loader_tar, optimizer=optimizer, accu_step=accu_step, lrscheduler=lrscheduler,
                grad_norm=grad_norm, broadcast=broadcast, interval=interval, **kwargs)
            record = pd.concat([record, record_epoch])

            # save
            if with_save and (epoch_ind + 1) % save_step == 0:
                broadcast('Save weight at ' + wei_pth)
                model.save(wei_pth)
                broadcast('Save optimizer at ' + opt_pth)
                torch.save(optimizer.state_dict(), opt_pth)
                record.to_excel(record_pth, index=False)
                msg = dict(epoch_ind=epoch_ind, best_pfrmce=best_pfrmce)
                save_json(msg_pth, msg)

            # test
            if with_test and (epoch_ind + 1) % test_step == 0:
                pfrmce = metric(model)
                if best_pfrmce < pfrmce:
                    best_pfrmce = max(best_pfrmce, pfrmce)
                    wei_pth_best = wei_pth.replace('.pth', '') + '_best.pth'
                    broadcast('Save best at ' + wei_pth_best)
                    model.save(wei_pth_best)

        return record

# </editor-fold>

# </editor-fold>
