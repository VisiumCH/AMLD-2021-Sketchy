import os

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn


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


class Logger(object):
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

    def add_embedding(self, embedding, metadata, label_img):
        self._writer.add_embedding(embedding, metadata=metadata, label_img=label_img,
                                   global_step=self.global_step)

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


class EmbeddingLogger(object):
    '''Logs the images in the latent space'''

    def __init__(self, valid_sk_data, valid_im_data, logger, dict_class, args):
        self.logger = logger
        self.dict_class = dict_class
        self.args = args
        (self.sketchy_limit_im, self.sketchy_limit_sk,
         self.tuberlin_limit_im, self.tuberlin_limit_sk) = get_limits(args.dataset, valid_sk_data, valid_im_data)

        self.select_embedding_images(valid_sk_data, valid_im_data, args.embedding_number, args)

    def select_embedding_images(self, valid_sk_data, valid_im_data, number_images, args):
        '''Save some random images to plot attention at defferent epochs'''
        sk_log, im_log, sk_lbl_log, im_lbl_log, index_sk, index_im = select_images(
            valid_sk_data, valid_im_data, number_images, args)

        self.sk_log = sk_log
        self.im_log = im_log

        # Convert class number to class name
        self.lbl = [get_labels_name(self.dict_class, value, index_im[i], args,
                                    self.sketchy_limit_im, self.tuberlin_limit_im)
                    for i, value in enumerate(im_lbl_log)]

        self.lbl.extend([get_labels_name(self.dict_class, value, index_sk[i], args,
                                         self.sketchy_limit_sk, self.tuberlin_limit_sk)
                         for i, value in enumerate(sk_lbl_log)])

    def plot_embeddings(self, im_net, sk_net):
        im_embedding, _ = im_net(self.im_log)
        sk_embedding, _ = sk_net(self.sk_log)

        all_embeddings = np.concatenate((im_embedding, sk_embedding), axis=0)
        all_images = np.concatenate((self.im_log, self.sk_log), axis=0)
        self.logger.add_embedding(all_embeddings, self.lbl, all_images)


class AttentionLogger(object):
    '''Logs some images to visulatise attenction module in tensorboard'''

    def __init__(self, valid_sk_data, valid_im_data, logger, dict_class, args):
        self.logger = logger
        self.dict_class = dict_class
        self.args = args
        (self.sketchy_limit_im, self.sketchy_limit_sk,
         self.tuberlin_limit_im, self.tuberlin_limit_sk) = get_limits(args.dataset, valid_sk_data, valid_im_data)
        self.select_attn_images(valid_sk_data, valid_im_data, args.attn_number, args)

    def select_attn_images(self, valid_sk_data, valid_im_data, number_images, args):
        '''Save some random images to plot attention at defferent epochs'''
        sk_log, im_log, sk_lbl_log, im_lbl_log, index_sk, index_im = select_images(
            valid_sk_data, valid_im_data, number_images, args)

        self.sk_log = sk_log
        self.im_log = im_log
        self.sk_lbl_log = sk_lbl_log
        self.im_lbl_log = im_lbl_log
        self.index_sk = index_sk
        self.index_im = index_im

    def plot_attention(self, im_net, sk_net):
        '''Log the attention images in tensorboard'''

        attn_im = self.process_attention(im_net, self.im_log)
        attn_sk = - self.process_attention(sk_net, self.sk_log)

        for i in range(self.im_log.size(0)):  # for each image-sketch pair

            plt_im = self.add_heatmap_on_image(self.im_log[i], attn_im[i])
            class_names = get_labels_name(self.dict_class, self.im_lbl_log[i], self.index_im[i],
                                          self.args, self.sketchy_limit_im, self.tuberlin_limit_im)
            self.logger.add_image('im{}_{}'.format(i, class_names), plt_im)

            plt_im = self.add_heatmap_on_image(self.sk_log[i], attn_sk[i])
            class_names = get_labels_name(self.dict_class, self.sk_lbl_log[i], self.index_sk[i],
                                          self.args, self.sketchy_limit_sk, self.tuberlin_limit_sk)
            self.logger.add_image('sk{}_{}'.format(i, class_names), plt_im)

    def process_attention(self, net, im):
        _, attn = net(im)
        attn = nn.Upsample(size=(im[0].size(1), im[0].size(2)), mode='bilinear', align_corners=False)(attn)
        min_attn = attn.view((attn.size(0), -1)).min(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        max_attn = attn.view((attn.size(0), -1)).max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return (attn - min_attn) / (max_attn - min_attn)

    def add_heatmap_on_image(self, im, attn):
        heat_map = attn.squeeze().detach().numpy()
        im = im.detach().numpy()
        im = np.transpose(im, (1, 2, 0))

        # Heatmap + Image on figure
        fig, ax = plt.subplots()
        ax.imshow(im)
        ax.imshow(255 * heat_map, alpha=0.8, cmap='Spectral_r')
        ax.axis('off')

        # Get value from canvas to pytorch tensor format
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot = np.transpose(image_from_plot, (2, 0, 1))
        return torch.tensor(image_from_plot.copy())


def select_images(valid_sk_data, valid_im_data, number_images, args):
    '''Save some random images to plot attention at defferent epochs'''
    rand_samples_sk = np.random.randint(0, high=len(valid_sk_data), size=number_images)
    rand_samples_im = np.random.randint(0, high=len(valid_im_data), size=number_images)
    for i in range(len(rand_samples_sk)):
        sk, _, lbl_sk = valid_sk_data[rand_samples_sk[i]]
        im, _, lbl_im = valid_im_data[rand_samples_im[i]]

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

    return sk_log, im_log, sk_lbl_log, im_lbl_log, rand_samples_sk, rand_samples_im


def get_limits(dataset, valid_sk_data, valid_im_data):
    if dataset == 'sk+tu' or dataset == 'sk+tu+qd':
        sketchy_images = valid_im_data.sketchy_limit_images
        sketchy_sketch = valid_sk_data.sketchy_limit_sketch
    else:
        sketchy_images = None
        sketchy_sketch = None

    if dataset == 'sk+tu+qd':
        tuberlin_images = valid_im_data.tuberlin_limit_images
        tuberlin_sketch = valid_sk_data.tuberlin_limit_sketch
    else:
        tuberlin_images = None
        tuberlin_sketch = None

    return sketchy_images, sketchy_sketch, tuberlin_images, tuberlin_sketch


def get_labels_name(dict_class, number_labels, idx, args, sketchy_limit, tuberlin_limit):

    if sketchy_limit is None:  # single dataset
        pass
    else:  # multiple datasets
        if idx < sketchy_limit:  # sketchy dataset
            dict_class = dict_class[0]
        else:
            if tuberlin_limit is None or idx < tuberlin_limit:  # tuberlin dataset
                dict_class = dict_class[1]
            else:  # quickdraw dataset
                dict_class = dict_class[2]

    return list(dict_class.keys())[list(dict_class.values()).index(number_labels)]
