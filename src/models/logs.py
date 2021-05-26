import os

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch

from src.data.utils import get_loader, get_dict, get_limits, get_dataset_dict
from src.models.utils import normalise_attention

NUM_CLOSEST = 5


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
    """Logs data in tensorboard"""

    def __init__(self, log_dir, force=False):
        # create the summary writer object
        self._writer = SummaryWriter(log_dir)

        self.global_step = 0

        # To load losses during training
        self.log_step = 0

    def __del__(self):
        self._writer.close()

    def add_scalar_training(self, name, scalar_value):
        """ Log a scalar every  args.log_interval images during training """
        assert isinstance(scalar_value, float), type(scalar_value)
        self._writer.add_scalar(name, scalar_value, self.log_step)
        self.log_step += 1

    def add_scalar(self, name, scalar_value):
        """ Log a scalar at the end of an epoch """
        assert isinstance(scalar_value, float), type(scalar_value)
        self._writer.add_scalar(name, scalar_value, self.global_step)

    def add_image(self, name, img_tensor):
        """ Log an image at the end of an epoch """
        assert isinstance(img_tensor, torch.Tensor), type(img_tensor)
        self._writer.add_image(name, img_tensor, self.global_step)

    def add_embedding(self, embedding, metadata, label_img):
        """ Log embeddings in the latent space at the end of an epoch """
        self._writer.add_embedding(
            embedding,
            metadata=metadata,
            label_img=label_img,
            global_step=self.global_step,
        )

    def step(self):
        """ Increment training step """
        self.global_step += 1


class InferenceLogger(object):
    """ Inference of the closest images of a selected sketches at different epochs"""

    def __init__(self, valid_sk_data, valid_im_data, logger, dict_class, args):
        self.logger = logger
        self.dict_class = dict_class
        self.args = args
        sketchy_limit_sk, tuberlin_limit_sk = get_limits(
            args.dataset, valid_sk_data, "sketches"
        )
        self.sk_log, self.sk_class_names, self.sk_indexes = select_images(
            valid_sk_data,
            args.inference_number,
            dict_class,
            sketchy_limit_sk,
            tuberlin_limit_sk,
        )

    def plot_inference(self, similarity, images_fnames, images_classes):
        """
        Logs an inference plot to tensorboard:
            The sketch is on the left
            The NUM_CLOSEST images in the latent space are on the right
        """
        images_fnames = [
            image for batch_image in images_fnames for image in batch_image
        ]
        arg_sorted_sim = np.array(
            [(-(similarity[index, :])).argsort() for index in self.sk_indexes]
        )

        for i, sk in enumerate(self.sk_log):
            # inference
            self.sorted_fnames = [
                images_fnames[j] for j in arg_sorted_sim[i][0:NUM_CLOSEST]
            ]
            self.sorted_classes = [
                images_classes[j] for j in arg_sorted_sim[i][0:NUM_CLOSEST]
            ]

            fig, axes = plt.subplots(1, NUM_CLOSEST, figsize=(NUM_CLOSEST * 4, 10))
            axes[0].imshow(sk.permute(1, 2, 0).numpy())
            axes[0].set_title("Sketch \n Label: " + self.sk_class_names[i], fontsize=16)
            axes[0].axis("off")

            for j in range(1, NUM_CLOSEST):
                dataset = self.sorted_fnames[j - 1].split("/")[-4]
                loader = get_loader(dataset)

                dict_class = get_dict(dataset, self.dict_class)
                class_name = list(dict_class.keys())[
                    list(dict_class.values()).index(self.sorted_classes[j - 1])
                ]
                im = loader(self.sorted_fnames[j - 1])

                axes[j].imshow(im)
                axes[j].set_title(
                    "Closest image " + str(j) + "\n Label: " + class_name, fontsize=16
                )
                axes[j].axis("off")

            plt.subplots_adjust(wspace=0.25, hspace=-0.35)

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            image_from_plot = np.transpose(image_from_plot, (2, 0, 1))
            infer_plt = torch.tensor(image_from_plot.copy())

            # Close image
            plt.cla()
            plt.close(fig)

            self.logger.add_image(
                "Inference_{}_{}".format(i, self.sk_class_names[i]), infer_plt
            )


class AttentionLogger(object):
    """Logs some images to visualise attenction module in tensorboard"""

    def __init__(self, valid_sk_data, valid_im_data, logger, dict_class, args):
        self.logger = logger

        sketchy_limit_im, tuberlin_limit_im = get_limits(
            args.dataset, valid_im_data, "images"
        )
        sketchy_limit_sk, tuberlin_limit_sk = get_limits(
            args.dataset, valid_sk_data, "sketches"
        )

        self.sk_log, self.sk_class_names, _ = select_images(
            valid_sk_data,
            args.attn_number,
            dict_class,
            sketchy_limit_sk,
            tuberlin_limit_sk,
        )
        self.im_log, self.im_class_names, _ = select_images(
            valid_im_data,
            args.attn_number,
            dict_class,
            sketchy_limit_im,
            tuberlin_limit_im,
        )

    def plot_attention(self, im_net, sk_net):
        """Log the attention images and sketches in tensorboard"""

        _, attn_im = im_net(self.im_log)
        attn_im = normalise_attention(attn_im, self.im_log)

        _, attn_sk = im_net(self.sk_log)
        attn_sk = normalise_attention(attn_sk, self.sk_log)

        for i in range(self.im_log.size(0)):  # for each image-sketch pair

            plt_im = add_heatmap_on_image(self.im_log[i], attn_im[i])
            self.logger.add_image("im{}_{}".format(i, self.im_class_names[i]), plt_im)

            plt_sk = add_heatmap_on_image(self.sk_log[i], attn_sk[i])
            self.logger.add_image("sk{}_{}".format(i, self.sk_class_names[i]), plt_sk)


class EmbeddingLogger(object):
    """Logs the images and sketches embeddings in the latent space"""

    def __init__(self, valid_sk_data, valid_im_data, logger, dict_class, args):
        self.logger = logger

        sketchy_limit_im, tuberlin_limit_im = get_limits(
            args.dataset, valid_im_data, "images"
        )
        sketchy_limit_sk, tuberlin_limit_sk = get_limits(
            args.dataset, valid_sk_data, "sketches"
        )

        self.sk_log, self.sk_class_names = select_images_from_class(
            args,
            valid_sk_data,
            args.embedding_number,
            dict_class,
            sketchy_limit_sk,
            tuberlin_limit_sk,
        )
        self.im_log, self.im_class_names = select_images_from_class(
            args,
            valid_im_data,
            args.embedding_number,
            dict_class,
            sketchy_limit_im,
            tuberlin_limit_im,
        )

        self.all_images = np.concatenate((self.sk_log, self.im_log), axis=0)
        self.all_classes = np.concatenate(
            (self.sk_class_names, self.im_class_names), axis=0
        )

    def plot_embeddings(self, im_net, sk_net):
        """ Compute images embeddings and log them in tensorboard """
        sk_embedding, _ = sk_net(self.sk_log)
        im_embedding, _ = im_net(self.im_log)

        all_embeddings = np.concatenate((sk_embedding, im_embedding), axis=0)
        self.logger.add_embedding(all_embeddings, self.all_classes, self.all_images)


def select_images(
    valid_data, number_images, all_dict_class, sketchy_limit_im, tuberlin_limit_im
):
    """Select some random images/sketch for tensorboard plots """
    class_names = []
    rand_samples = np.random.randint(0, high=len(valid_data), size=number_images)
    for i in range(len(rand_samples)):
        im, _, label = valid_data[rand_samples[i]]

        if i == 0:
            im_log = im.unsqueeze(0)
        else:
            im_log = torch.cat((im_log, im.unsqueeze(0)), dim=0)

        dict_class = get_dataset_dict(
            all_dict_class, rand_samples[i], sketchy_limit_im, tuberlin_limit_im
        )
        class_name = list(dict_class.keys())[list(dict_class.values()).index(label)]
        class_names.append(class_name)

    return im_log, class_names, rand_samples


def add_heatmap_on_image(im, attn):
    """
    Creates a plot with three subplots: the image, the heatmap of the attention and both superposed
    Args:
        - im: image to plot with attention
        - attn: attention values on image
    Return:
        - tensor of image to show in tensorboard
    """
    heat_map = attn.squeeze().detach().numpy()
    im = im.detach().numpy()
    im = np.transpose(im, (1, 2, 0))

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    # Image on figure
    axes[0].imshow(im)
    axes[0].set(title="Original")
    axes[0].axis("off")

    # Heatmap on figure
    axes[1].imshow(255 * heat_map, cmap="Spectral_r")
    axes[1].set(title="Heatmap")
    axes[1].axis("off")

    # Heatmap + Image on figure
    axes[2].imshow(im)
    axes[2].set(title="Original + Heatmap")
    axes[2].imshow(255 * heat_map, alpha=0.7, cmap="Spectral_r")
    axes[2].axis("off")

    # Get value from canvas to pytorch tensor format
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    image_from_plot = np.transpose(image_from_plot, (2, 0, 1))

    # Close image
    plt.cla()
    plt.close(fig)

    return torch.tensor(image_from_plot.copy())


def select_images_from_class(
    args, valid_data, number_images, all_dict_class, sketchy_limit_im, tuberlin_limit_im
):
    """Select some random images/sketch for tensorboard plots """
    class_names = []

    embedding_file = os.path.join(args.data_path, "embeddings_class.txt")
    with open(embedding_file, "r") as f:
        embedding_classes = f.read()
        embedding_classes = embedding_classes.split("\n")

    while len(class_names) < number_images:
        random_sample = np.random.randint(0, high=len(valid_data))
        im, _, label = valid_data[random_sample]

        dict_class = get_dataset_dict(
            all_dict_class, random_sample, sketchy_limit_im, tuberlin_limit_im
        )
        class_name = list(dict_class.keys())[list(dict_class.values()).index(label)]

        if class_name in embedding_classes:
            class_names.append(class_name)

            if len(class_names) == 1:
                im_log = im.unsqueeze(0)
            else:
                im_log = torch.cat((im_log, im.unsqueeze(0)), dim=0)

    return im_log, class_names
