import os

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

from src.data.utils import get_loader, get_dict
from src.models.utils import get_limits, get_dataset_dict

NUM_CLOSEST = 4


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

        # To load losses during training
        self.log_step = 0

    def __del__(self):
        self._writer.close()

    # During training every args.log_interval images
    def add_scalar_training(self, name, scalar_value):
        assert isinstance(scalar_value, float), type(scalar_value)
        self._writer.add_scalar(name, scalar_value, self.log_step)
        self.log_step += 1

    # At the end of each epoch
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


class InferenceLogger(object):
    '''Logs the images in the latent space'''

    def __init__(self, valid_sk_data, valid_im_data, logger, dict_class, args):
        self.logger = logger
        self.dict_class = dict_class
        self.args = args
        sketchy_limit_sk, tuberlin_limit_sk = get_limits(args.dataset, valid_sk_data, 'sketch')
        self.sk_log, self.sk_class_names, self.sk_indexes = select_images(
            valid_sk_data, args.inference_number, dict_class, sketchy_limit_sk, tuberlin_limit_sk)

    def plot_inference(self, similarity, images_fnames, images_classes):
        images_fnames = [image for batch_image in images_fnames for image in batch_image]
        im_similarity = np.array([similarity[index, :] for index in self.sk_indexes])
        arg_sorted_sim = np.array([(-im_sim).argsort() for im_sim in im_similarity])

        for i, sk in enumerate(self.sk_log):
            self.sorted_fnames = [images_fnames[j] for j in arg_sorted_sim[i][0:NUM_CLOSEST]]
            self.sorted_classes = [images_classes[i] for j in arg_sorted_sim[i][0:NUM_CLOSEST]]

            fig, axes = plt.subplots(1, NUM_CLOSEST, figsize=(25, 12))
            axes[0].imshow(sk.permute(1, 2, 0).numpy())
            axes[0].set(title='Sketch \n Label: ' + self.sk_class_names[i])
            axes[0].axis('off')

            for j in range(1, NUM_CLOSEST):
                dataset = self.sorted_fnames[j-1].split('/')[-4]
                loader = get_loader(dataset)

                dict_class = get_dict(dataset, self.dict_class)
                class_name = list(dict_class.keys())[list(dict_class.values()).index(self.sorted_classes[j-1])]
                im = loader(self.sorted_fnames[j-1])

                axes[j].imshow(im)
                axes[j].set(title='Closest image ' + str(j) + '\n Label: ' + class_name)
                axes[j].axis('off')

            plt.subplots_adjust(wspace=0.25, hspace=-0.35)

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image_from_plot = np.transpose(image_from_plot, (2, 0, 1))
            infer_plt = torch.tensor(image_from_plot.copy())

            self.logger.add_image('Inference_{}_{}'.format(i, self.sk_class_names[i]), infer_plt)


class EmbeddingLogger(object):
    '''Logs the images in the latent space'''

    def __init__(self, valid_sk_data, valid_im_data, logger, dict_class, args):
        self.logger = logger

        sketchy_limit_im, tuberlin_limit_im = get_limits(args.dataset, valid_im_data, 'image')
        sketchy_limit_sk, tuberlin_limit_sk = get_limits(args.dataset, valid_sk_data, 'sketch')

        self.sk_log, self.sk_class_names, _ = select_images(
            valid_sk_data, args.embedding_number, dict_class, sketchy_limit_sk, tuberlin_limit_sk)
        self.im_log, self.im_class_names, _ = select_images(
            valid_im_data, args.embedding_number, dict_class, sketchy_limit_im, tuberlin_limit_im)

        self.all_images = np.concatenate((self.sk_log, self.im_log), axis=0)
        self.all_classes = np.concatenate((self.sk_class_names, self.im_class_names), axis=0)

    def plot_embeddings(self, im_net, sk_net):
        sk_embedding, _ = sk_net(self.sk_log)
        im_embedding, _ = im_net(self.im_log)

        all_embeddings = np.concatenate((sk_embedding, im_embedding), axis=0)
        self.logger.add_embedding(all_embeddings, self.all_classes, self.all_images)


class AttentionLogger(object):
    '''Logs some images to visulatise attenction module in tensorboard'''

    def __init__(self, valid_sk_data, valid_im_data, logger, dict_class, args):
        self.logger = logger

        sketchy_limit_im, tuberlin_limit_im = get_limits(args.dataset, valid_im_data, 'image')
        sketchy_limit_sk, tuberlin_limit_sk = get_limits(args.dataset, valid_sk_data, 'sketch')

        self.sk_log, self.sk_class_names, _ = select_images(
            valid_sk_data, args.attn_number, dict_class, sketchy_limit_sk, tuberlin_limit_sk)
        self.im_log, self.im_class_names, _ = select_images(
            valid_im_data, args.attn_number, dict_class, sketchy_limit_im, tuberlin_limit_im)

    def plot_attention(self, im_net, sk_net):
        '''Log the attention images in tensorboard'''

        attn_im = self.process_attention(im_net, self.im_log)
        attn_sk = self.process_attention(sk_net, self.sk_log)

        for i in range(self.im_log.size(0)):  # for each image-sketch pair

            plt_im = self.add_heatmap_on_image(self.im_log[i], attn_im[i])
            self.logger.add_image('im{}_{}'.format(i, self.im_class_names[i]), plt_im)

            plt_sk = self.add_heatmap_on_image(self.sk_log[i], attn_sk[i])
            self.logger.add_image('sk{}_{}'.format(i, self.sk_class_names[i]), plt_sk)

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


def select_images(valid_data, number_images, all_dict_class, sketchy_limit_im, tuberlin_limit_im):
    '''Save some random images to plot attention at defferent epochs'''
    class_names = []
    rand_samples = np.random.randint(0, high=len(valid_data), size=number_images)
    for i in range(len(rand_samples)):
        im, _, label = valid_data[rand_samples[i]]

        if i == 0:
            im_log = im.unsqueeze(0)
        else:
            im_log = torch.cat((im_log, im.unsqueeze(0)), dim=0)

        dict_class = get_dataset_dict(all_dict_class, rand_samples[i], sketchy_limit_im, tuberlin_limit_im)
        class_name = list(dict_class.keys())[list(dict_class.values()).index(label)]
        class_names.append(class_name)

    return im_log, class_names, rand_samples
