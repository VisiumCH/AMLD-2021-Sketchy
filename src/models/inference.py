import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


from src.data.loader_factory import load_data
from src.data.utils import default_image_loader
from src.models.encoder import EncoderCNN
from src.models.utils import load_checkpoint
from src.models.test import get_test_data
from src.models.metrics import get_similarity


def get_model(args, best_checkpoint):
    im_net = EncoderCNN(out_size=args.emb_size, attention=True)
    sk_net = EncoderCNN(out_size=args.emb_size, attention=True)

    if args.cuda:
        checkpoint = torch.load(best_checkpoint)
    else:
        checkpoint = torch.load(best_checkpoint, map_location='cpu')

    im_net.load_state_dict(checkpoint['im_state'])
    sk_net.load_state_dict(checkpoint['sk_state'])

    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        im_net = nn.DataParallel(im_net, device_ids=list(range(args.ngpu)))
        sk_net = nn.DataParallel(sk_net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        im_net, sk_net = im_net.cuda(), sk_net.cuda()

    return im_net, sk_net


class Inference():

    def __init__(self, model_path):

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.loader = default_image_loader

        self.im_net, self.sk_net = get_model(args, model_path)
        self.prepare_image_embeddings()

    def prepare_image_embeddings(self):
        '''Get all images and compute their embeddings'''

        _, [_, _], [_, test_im_data], dict_class = load_data(args, self.transform)
        print("Length Image: {}".format(len(test_im_data)))
        print("Classes: {}".format(test_sk_data.get_class_dict()))

        test_im_loader = DataLoader(test_im_data, batch_size=1)
        self.images_fnames, self.images_embeddings, self.images_classes = get_test_data(test_im_loader, im_net, args)

    def inference_sketch(self, sketch_fname, plot=True):
        ''' For now just process a sketch but TODO decide how to proceed later'''

        sketch = self.transform(self.loader(sketch_fname)).unsqueeze(0)  # unsqueeze because 1 sketch (no batch)
        sketch_embedding, _ = self.sk_net(sketch)
        self.get_closest_images(sketch_embedding)

        if plot:
            self.plot_closest(sketch_fname)

    def get_closest_images(self, sketch_embedding):
        '''
        Based on a sketch embedding, retrieve the index of the closest images
        '''

        similarity = get_similarity(sketch_embedding, self.images_embeddings)
        arg_sorted_sim = (-similarity).argsort()

        self.sorted_fnames = [self.images_fnames[i][0] for i in arg_sorted_sim[0]]

    def plot_closest(self, sketch_fname):
        fig, axes = plt.subplots(1, NUM_CLOSEST + 1)

        sk = mpimg.imread(sketch_fname)
        axes[0].imshow(sk)
        axes[0].set(title='Sketch')

        for i in range(1, NUM_CLOSEST + 1):
            im = mpimg.imread(self.sorted_fnames[i-1])
            axes[i].imshow(im)
            axes[i].set(title='Closest image ' + str(i))

        plt.subplots_adjust(wspace=0.25, hspace=-0.35)


def main():
    # TODO:modify when better idea of process
    inference = Inference(args.load)

    sketch_fname = '../io/data/raw/Sketchy/sketch/tx_000000000000/bat/n02139199_1332-1.png'
    closest_images = inference.inference_sketch(sketch_fname, plot=True)

    sketch_fname = '../io/data/raw/Sketchy/sketch/tx_000000000000/door/n03222176_681-1.png'
    closest_images = inference.inference_sketch(sketch_fname, plot=True)

    sketch_fname = '../io/data/raw/Sketchy/sketch/tx_000000000000/giraffe/n02439033_67-1.png'
    closest_images = inference.inference_sketch(sketch_fname, plot=True)

    sketch_fname = '../io/data/raw/Sketchy/sketch/tx_000000000000skyscraper/n04233124_498-1.png'
    closest_images = inference.inference_sketch(sketch_fname, plot=True)

    sketch_fname = '../io/data/raw/Sketchy/sketch/tx_000000000000/wheelchair/n04576002_150-2.png'
    closest_images = inference.inference_sketch(sketch_fname, plot=True)


if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    # Check Test and Load
    if args.load is None:
        raise Exception('Cannot test without loading a model.')

    main()
