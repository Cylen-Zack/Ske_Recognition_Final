#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import time
import glob
import pickle
import random
import traceback
# import resource

from collections import OrderedDict

import torch
import torch.optim as optim
import numpy as np
import yaml

from tqdm import tqdm
# from sklearn.metrics import confusion_matrix

from args import get_parser
from loss import LabelSmoothingCrossEntropy, get_mmd_loss, MultiClassFocalLossWithAlpha
from model.infogcn import InfoGCN
from utils import get_vector_property
from utils import BalancedSampler as BS

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.best_acc = 0
        self.best_acc_epoch = 0

        model = self.model.cuda()

        # self.model = torch.nn.DataParallel(model, device_ids=(0,1,2))

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        data_path = self.arg.data_path
        if self.arg.phase == 'train':
            dt = Feeder(data_path=data_path,
                split='train',
                window_size=self.arg.window_size,
                p_interval=[0.5, 1],
                vel=self.arg.use_vel,
                angle=self.arg.use_angle,
                random_rot=self.arg.random_rot,
                sort=True if self.arg.balanced_sampling else False,
            )
            if self.arg.balanced_sampling:
                sampler = BS(data_source=dt, args=self.arg)
                shuffle = False
            else:
                sampler = None
                shuffle = True
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dt,
                sampler=sampler,
                batch_size=self.arg.batch_size,
                shuffle=shuffle,
                num_workers=self.arg.num_worker,
                drop_last=True,
                pin_memory=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(
                data_path=data_path,
                split='test',
                window_size=self.arg.window_size,
                p_interval=[0.95],
                vel=self.arg.use_vel,
                angle=self.arg.use_angle
            ),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=init_seed)

    def load_model(self):
        self.model = InfoGCN(
            num_class=self.arg.num_class,
            num_point=self.arg.num_point,
            num_person=self.arg.num_person,
            num_frame=self.arg.window_size,
            graph=self.arg.graph,
            in_channels=self.arg.in_channels,
            drop_out=0,
            num_head=self.arg.n_heads,
            k=self.arg.k,
            noise_ratio=self.arg.noise_ratio,
            gain=self.arg.z_prior_gain,
            cl_mode=self.arg.cl_mode
        )
        if self.arg.focal_loss:
            self.loss = MultiClassFocalLossWithAlpha(num_class=self.arg.num_class).cuda()
        else:
            self.loss = LabelSmoothingCrossEntropy().cuda()

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda()] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch and self.arg.weights is None:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.adjust_learning_rate(epoch)

        loss_value = []
        mmd_loss_value = []
        l2_z_mean_value = []
        acc_value = []
        cos_z_value = []
        dis_z_value = []
        cos_z_prior_value = []
        dis_z_prior_value = []

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        for data, y, index in tqdm(self.data_loader['train'], dynamic_ncols=True):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda()
                y = y.long().cuda()
            timer['dataloader'] += self.split_time()

            # forward
            cl_loss = None
            if self.arg.cl_mode is not None:
                y_hat, z, cl_loss = self.model(data, y, get_cl_loss=True)
            else:
                y_hat, z = self.model(data)
            mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.z_prior, y, self.arg.num_class)
            cos_z, dis_z = get_vector_property(z_mean)
            cos_z_prior, dis_z_prior = get_vector_property(self.model.z_prior)
            cos_z_value.append(cos_z.data.item())
            dis_z_value.append(dis_z.data.item())
            cos_z_prior_value.append(cos_z_prior.data.item())
            dis_z_prior_value.append(dis_z_prior.data.item())

            cls_loss = self.loss(y_hat, y)
            loss = self.arg.lambda_2* mmd_loss + self.arg.lambda_1* l2_z_mean + cls_loss
            if self.arg.cl_mode is not None:
                full_loss = loss + self.arg.w_cl_weights * cl_loss.mean()
            else:
                full_loss = loss
            # backward
            self.optimizer.zero_grad()
            full_loss.backward()
            self.optimizer.step()

            loss_value.append(cls_loss.data.item())
            mmd_loss_value.append(mmd_loss.data.item())
            l2_z_mean_value.append(l2_z_mean.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(y_hat.data, 1)
            acc = torch.mean((predict_label == y.data).float())
            acc_value.append(acc.data.item())

            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(f'\tTraining loss: {np.mean(loss_value):.4f}.  Training acc: {np.mean(acc_value)*100:.2f}%.')
        self.print_log(f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')

        # if save_model:
        #     state_dict = self.model.state_dict()
        #     weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        #
        #     torch.save(weights, f'{self.arg.work_dir}/runs-{epoch+1}-{int(self.global_step)}.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], save_z=False):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            cls_loss_value = []
            mmd_loss_value = []
            l2_z_mean_value = []
            score_frag = []
            label_list = []
            pred_list = []
            cos_z_value = []
            dis_z_value = []
            cos_z_prior_value = []
            dis_z_prior_value = []
            step = 0
            z_list = []
            for data, y, index in tqdm(self.data_loader[ln], dynamic_ncols=True):
                label_list.append(y)
                with torch.no_grad():
                    data = data.float().cuda()
                    y = y.long().cuda()
                    y_hat, z = self.model(data)
                    if save_z:
                        z_list.append(z.data.cpu().numpy())
                    mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.z_prior, y, self.arg.num_class)
                    cos_z, dis_z = get_vector_property(z_mean)
                    cos_z_prior, dis_z_prior = get_vector_property(self.model.z_prior)
                    cos_z_value.append(cos_z.data.item())
                    dis_z_value.append(dis_z.data.item())
                    cos_z_prior_value.append(cos_z_prior.data.item())
                    dis_z_prior_value.append(dis_z_prior.data.item())
                    cls_loss = self.loss(y_hat, y)
                    loss = self.arg.lambda_2*mmd_loss + self.arg.lambda_1*l2_z_mean + cls_loss
                    score_frag.append(y_hat.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    cls_loss_value.append(cls_loss.data.item())
                    mmd_loss_value.append(mmd_loss.data.item())
                    l2_z_mean_value.append(l2_z_mean.data.item())

                    _, predict_label = torch.max(y_hat.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            cls_loss = np.mean(cls_loss_value)
            mmd_loss = np.mean(mmd_loss_value)
            l2_z_mean_loss = np.mean(l2_z_mean_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {:4f}.'.format(
                ln, self.arg.n_desired//self.arg.batch_size, np.mean(cls_loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)
                print('save_score')

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                with open(f'{self.arg.work_dir}/best_score.pkl', 'wb') as f:
                    pickle.dump(score_dict, f)

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)

            if save_z:
                z_list = np.concatenate(z_list)
                np.savez(f'{self.arg.work_dir}/z_values.npz', z=z_list, z_prior=self.model.z_prior.cpu().numpy(), y=label_list)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = 0
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # save_model = (epoch + 1 == self.arg.num_epoch) and (epoch + 1 > self.arg.save_epoch)

                self.train(epoch)

                # if epoch > 80:
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

                save_model = ((epoch + 1 == self.best_acc_epoch) or (epoch + 1 == self.arg.save_epoch)) and (epoch + 1 >= self.arg.save_epoch)

                if save_model:
                    state_dict = self.model.state_dict()
                    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                    torch.save(weights, f'{self.arg.work_dir}/runs-{epoch + 1}-{int(self.global_step)}.pt')

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            self.model.load_state_dict(weights)

            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'])
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], save_z=True)
            self.print_log('Done.\n')

def main():
    # parser arguments
    parser = get_parser()
    arg = parser.parse_args()
    if arg.config is not None:
        with open(arg.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(arg).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.device)
    init_seed(arg.seed)
    # execute process
    processor = Processor(arg)
    processor.start()

if __name__ == '__main__':
    main()