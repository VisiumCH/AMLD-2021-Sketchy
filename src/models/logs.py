import torch
from tensorboardX import SummaryWriter
import os

import torch.nn as nn
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ScalarLogger(object):
    '''Logs Scalars in tensorboard'''

    def __init__(self, log_dir, force=False):
        # clean previous logged data under the same directory name
        self._remove(log_dir, force)

        # create the summary writer object
        self._writer = SummaryWriter(log_dir)

        self.global_step = 0

    def __del__(self):
        self._writer.close()

    def add_scalar(self, name, scalar_value):
        assert isinstance(scalar_value, float), type(scalar_value)
        self._writer.add_scalar(name, scalar_value, self.global_step)

    def add_image(self, name, img_tensor):
        assert isinstance(img_tensor, torch.Tensor), type(img_tensor)
        self._writer.add_image(name, img_tensor, self.global_step)

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path, force):
        """ param <path> could either be relative or absolute. """
        if not os.path.exists(path):
            return
        elif os.path.isfile(path) and force:
            os.remove(path)  # remove the file
        elif os.path.isdir(path) and force:
            import shutil
            shutil.rmtree(path)  # remove dir and all contains
        else:
            print('Logdir contains data. Please, set `force` flag to overwrite it.')
            import sys
            sys.exit(0)


class AttentionLogger(object):
    '''Logs some images to visulatise attenction module in tensorboard'''

    def __init__(self, valid_sk_data, valid_im_data, logger, dict_class, args):
        self.logger = logger
        self.dict_class = dict_class
        self.args = args
        self.select_images(valid_sk_data, valid_im_data)

    def select_images(self, valid_sk_data, valid_im_data):
        '''Save some random images to plot attention at defferent epochs'''
        rand_samples_sk = np.random.randint(0, high=len(valid_sk_data), size=5)
        rand_samples_im = np.random.randint(0, high=len(valid_im_data), size=5)
        for i in range(len(rand_samples_sk)):
            sk, _, lbl_sk = valid_sk_data[rand_samples_sk[i]]
            im, _, lbl_im = valid_im_data[rand_samples_im[i]]
            if self.args.cuda:
                sk, im = sk.cuda(), im.cuda()
            if i == 0:
                sk_log = sk.unsqueeze(0)
                im_log = im.unsqueeze(0)
                sk_lbl_log = [lbl_sk]
                im_lbl_log = [lbl_im]
            else:
                sk_log = torch.cat((sk_log, sk.unsqueeze(0)), dim=0)
                im_log = torch.cat((im_log, im.unsqueeze(0)), dim=0)
                sk_lbl_log.append(lbl_sk)
                im_lbl_log.append(lbl_im)

        self.sk_log = sk_log
        self.im_log = im_log
        self.sk_lbl_log = sk_lbl_log
        self.im_lbl_log = im_lbl_log

    def plot_attention(self, im_net, sk_net):
        '''Log the attention images in tensorboard'''
        _, attn_im = im_net(self.im_log)
        attn_im = nn.Upsample(size=(self.im_log[0].size(1), self.im_log[0].size(2)),
                              mode='bilinear', align_corners=False)(attn_im)
        attn_im = attn_im - attn_im.view((attn_im.size(0), -1)
                                         ).min(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        attn_im = 1 - attn_im/attn_im.view((attn_im.size(0), -1)
                                           ).max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        _, attn_sk = sk_net(self.sk_log)
        attn_sk = nn.Upsample(size=(self.sk_log[0].size(1), self.sk_log[0].size(2)),
                              mode='bilinear', align_corners=False)(attn_sk)
        attn_sk = attn_sk - attn_sk.view((attn_sk.size(0), -1)
                                         ).min(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        attn_sk = attn_sk/attn_sk.view((attn_sk.size(0), -1)
                                       ).max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for i in range(self.im_log.size(0)):
            plt_im = torch.cat([self.im_log[i], attn_im[i]], dim=0)
            nam = list(self.dict_class.keys())[list(self.dict_class.values()).index(self.im_lbl_log[i])]
            self.logger.add_image('im{}_{}'.format(i, nam), plt_im)

            nam = list(self.dict_class.keys())[list(self.dict_class.values()).index(self.sk_lbl_log[i])]
            plt_im = self.sk_log[i]*attn_sk[i]
            self.logger.add_image('sk{}_{}'.format(i, nam), plt_im)