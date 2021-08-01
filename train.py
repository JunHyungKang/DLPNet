import torch
import torch.nn as nn
import os
import numpy as np
import loss
import cv2
import func_utils
import tqdm
import torchvision.models as torchmodels
import torch.nn.functional as F
from models import rl
from datasets.DOTA_devkit import dota_v2_evaluation_task1
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio):
        self.writer = None
        self.dataset = dataset
        self.dataset_phase = {'dota': ['train', 'test'],
                              'hrsc': ['train', 'test']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio

        self.rl = rl.ReinforceDisc(5)
        self.rl_agent = self.rl.agent

        self.optimizer = None
        self.optimizer_rl = None
        self.scheduler = None
        self.scheduler_rl = None

    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer, epoch

    def train_network(self, args):
        self.writer = SummaryWriter(args.save_dir + '/log')
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        self.optimizer_rl = torch.optim.Adam(self.rl_agent.parameters(), args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        self.scheduler_rl = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_rl, gamma=0.96, last_epoch=-1)
        save_path = args.save_dir
        start_epoch = 1
        
        # add resume part for continuing training when break previously, 10-16-2020
        if args.resume_train:
            self.model, self.optimizer, start_epoch = self.load_model(self.model, 
                                                                        self.optimizer, 
                                                                        args.resume_train, 
                                                                        strict=True)
        # end

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if args.ngpus>1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
                self.rl_agent = nn.DataParallel(self.rl_agent)
        self.model.to(self.device)
        self.rl_agent.to(self.device)

        criterion = loss.LossAll()
        print('Setting up data...')

        # self.dataset = {'dota': DOTA, 'hrsc': HRSC} --> DOTA is class object
        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio)
                 for x in self.dataset_phase[args.dataset]}

        dsets_loader = dict()
        dsets_loader['train'] = DataLoader(dsets['train'],
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=False,
                                           drop_last=True,
                                           collate_fn=collater)

        print('Starting training...')
        train_loss = []
        ap_list = []
        num_iter = 0
        for epoch in range(start_epoch, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss, num_iter, rl_loss, ep_ret = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion,
                                        num_iter=num_iter)

            train_loss.append(epoch_loss)
            self.scheduler.step()
            self.scheduler_rl.step()

            # np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_loss, fmt='%.6f')

            if epoch % 1 == 0 or epoch > 20:
                self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer)
                self.save_model(os.path.join(save_path, 'rl_model_{}.pth'.format(epoch)),
                                epoch,
                                self.rl_agent,
                                self.optimizer_rl)

            if 'test' in self.dataset_phase[args.dataset]:
                mAP = self.dec_eval(args, dsets['test'])
                ap_list.append(mAP)
                print('mAP: {}'.format(mAP))
                # np.savetxt(os.path.join(save_path, 'ap_list.txt'), np.array(ap_list))
                self.writer.add_scalar("Loss/train", epoch_loss, epoch)
                self.writer.add_scalar("Loss/rl_train", rl_loss, epoch)
                self.writer.add_scalar("mAP/train", mAP, epoch)
                self.writer.add_scalar("RL/return", ep_ret, epoch)

            self.save_model(os.path.join(save_path, 'model_last.pth'),
                            epoch,
                            self.model,
                            self.optimizer)

    def run_epoch(self, phase, data_loader, criterion, num_iter):
        rews = []
        ep_logp =[]
        acts = []

        if phase == 'train':
            self.model.train()
            self.rl_agent.train()
        else:
            self.model.eval()
            self.rl_agent.eval()

        running_loss = 0.
        num_iter = num_iter

        for data_dict in tqdm.tqdm(data_loader):
            torch.cuda.empty_cache()
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)

            if phase == 'train':
                self.optimizer.zero_grad()
                self.optimizer_rl.zero_grad()
                with torch.enable_grad():
                    data_tensor = torch.as_tensor(data_dict['input'])

                    act = self.rl.get_action(data_tensor)
                    acts.extend(act.detach().cpu().tolist())

                    para_list = self.rl.get_parameter(act)
                    pr_decs = self.model(data_dict['input'])

                    origin_hm_loss, origin_wh_loss, origin_off_loss, origin_cls_theta_loss, origin_loss = criterion(pr_decs, data_dict, None)
                    hm_loss, wh_loss, off_loss, cls_theta_loss, loss = criterion(pr_decs, data_dict, para_list)

                    self.writer.add_scalar("batch_loss/origin_hm_loss", origin_hm_loss, num_iter)
                    self.writer.add_scalar("batch_loss/origin_wh_loss", origin_wh_loss, num_iter)
                    self.writer.add_scalar("batch_loss/origin_off_loss", origin_off_loss, num_iter)
                    self.writer.add_scalar("batch_loss/origin_cls_theta_loss", origin_cls_theta_loss, num_iter)
                    self.writer.add_scalar("batch_loss/origin_sum_loss", origin_loss, num_iter)

                    self.writer.add_scalar("batch_loss/hm_loss", origin_hm_loss, num_iter)
                    self.writer.add_scalar("batch_loss/wh_loss", origin_wh_loss, num_iter)
                    self.writer.add_scalar("batch_loss/off_loss", origin_off_loss, num_iter)
                    self.writer.add_scalar("batch_loss/cls_theta_loss", origin_cls_theta_loss, num_iter)
                    self.writer.add_scalar("batch_loss/sum_loss", origin_loss, num_iter)

                    rew = origin_loss - loss
                    self.writer.add_scalar("RL/rl_reward", rew, num_iter)
                    rews.append(rew.detach().cpu().item())

                    loss.backward()
                    self.optimizer.step()

                    batch_logp = self.rl.compute_logp(data_tensor, act)
                    ep_logp.append(sum(batch_logp.cpu().tolist()))

            else:
                with torch.no_grad():
                    (alpha, beta) = self.rl.get_action(data_tensor)
                    pr_decs = self.model(data_dict['input'])

                    origin_loss = criterion(pr_decs, data_dict, 2, 4)
                    loss = criterion(pr_decs, data_dict, alpha, beta)
                    rew = origin_loss - loss

            running_loss += loss.item()
            num_iter += 1

            del data_dict

        ep_ret = sum(rews)
        ep_len = len(rews)
        # ep_weights = [ep_ret] * ep_len
        ep_weights = list(self.rl.reward_to_go(rews))
        mean = np.mean(ep_weights)
        std = np.std(ep_weights)
        ep_weights = (ep_weights-mean)/std

        rl_loss = self.rl.compute_loss(logp=torch.as_tensor(ep_logp),
                                       weights=torch.as_tensor(ep_weights))

        rl_loss.requires_grad = True
        rl_loss.backward()

        self.optimizer_rl.step()

        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        print('{} RL loss: {}'.format(phase, rl_loss))
        print('{} RL return: {}'.format(phase, ep_ret))
        for i in range(5):
            print('action count for {}: {}'.format(i, acts.count(i)))
        return epoch_loss, num_iter, rl_loss, ep_ret


    def dec_eval(self, args, dsets):
        result_path = 'result_' + args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args,
                                 self.model, dsets,
                                 self.down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path)
        if args.dataset == 'dota':
            merge_path = 'merge_' + args.dataset
            if not os.path.exists(merge_path):
                os.mkdir(merge_path)
            dsets.merge_crop_image_results(result_path, merge_path)
            detpath = os.path.join(merge_path, 'Task1_{:s}.txt')
            annopath = os.path.join(args.data_dir, 'Val', 'labelTxt/{:s}.txt')
            imagesetfile = os.path.join(args.data_dir, 'Val', 'test.txt')
            map = dota_v2_evaluation_task1.main(detpath, annopath, imagesetfile)
            return map
        else:
            ap = dsets.dec_evaluation(result_path)
            return ap