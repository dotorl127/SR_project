""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5

        (epoch_val): ; (epoch_save):
"""

import argparse
import os

import yaml
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchinfo import summary

import datasets
import models
import utils
from test import eval_psnr


def make_data_loader(spec, exist=True, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    if not exist:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'), num_workers=8, pin_memory=False)
    return loader


def make_data_loaders(exist=True):
    train_loader = make_data_loader(config.get('train_dataset'), exist, tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), exist, tag='val')
    return train_loader, val_loader


def prepare_training(save_path):
    exist = False

    if os.path.exists(os.path.join(save_path, 'epoch-last.pth')):
        last_checkpoint = os.path.join(save_path, 'epoch-last.pth')
        sv_file = torch.load(last_checkpoint)
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        total_it = sv_file['total_it'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, milestones=config['multi_step_lr']['milestones'],
                                       gamma=config['multi_step_lr']['gamma'], last_epoch=(epoch_start - 1))
        exist = True
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        total_it = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    return model, optimizer, epoch_start, total_it, lr_scheduler, exist


def train(train_loader, model, optimizer, leave_pbar, total_it, tbar):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    pbar = tqdm.tqdm(total=len(train_loader), leave=leave_pbar, desc='train', dynamic_ncols=True)

    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div  # for standardization
        # pred = model(inp, batch['coord'], batch['cell']) # if LIIF model
        pred = model(inp)  # if EDSR model

        gt = (batch['gt'] - gt_sub) / gt_div  # for standardization
        loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_it += 1
        pbar.update()
        pbar.set_postfix(dict(total_it=total_it))

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        dist_dict = {'loss': loss.item(), 'lr': cur_lr}
        tbar.set_postfix(dist_dict)
        tbar.refresh()

    return train_loss.item(), total_it


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)

    yaml_path = os.path.join(save_path, 'config.yaml')
    if not os.path.exists(yaml_path):
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, total_it, lr_scheduler, exist = prepare_training(save_path)
    train_loader, val_loader = make_data_loaders(exist)

    if not exist:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
        log(str(summary(model, (1, 1, 64, 1024))), screen=False)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    if config.get('eval_type') is None:
        acc_str = 'psnr'
    else:
        acc_str = config.get('eval_type')

    with tqdm.trange(epoch_start, epoch_max + 1, desc='epochs', dynamic_ncols=True, leave=True) as tbar:
        for epoch in tbar:
            t_epoch_start = timer.t()
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            leave_pbar = epoch + 1 == epoch_max
            train_loss, total_it = train(train_loader, model, optimizer, leave_pbar, total_it, tbar)

            if lr_scheduler is not None:
                lr_scheduler.step()

            log_info.append('train: loss={:.4f}'.format(train_loss))
            writer.add_scalars('loss', {'train': train_loss}, epoch)

            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model

            model_spec = config['model']
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            sv_file = {'model': model_spec,
                       'optimizer': optimizer_spec,
                       'epoch': epoch,
                       'total_it': total_it}

            torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

            if (epoch_save is not None) and (epoch % epoch_save == 0):
                torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

            if (epoch_val is not None) and (epoch % epoch_val == 0):
                if n_gpus > 1 and (config.get('eval_bsize') is not None):
                    model_ = model.module
                else:
                    model_ = model

                val_res = eval_psnr(val_loader, model_,
                                    data_norm=config['data_norm'],
                                    eval_type=config.get('eval_type'),
                                    eval_bsize=config.get('eval_bsize'))

                log_info.append(f'val: {acc_str}={val_res:.4f}')
                writer.add_scalars(acc_str, {'val': val_res}, epoch)

                if val_res > max_val_v:
                    max_val_v = val_res
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log(', '.join(log_info))
            writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
