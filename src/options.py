import argparse


class Options():

    def __init__(self, test=False):
        # MODEL SETTINGS
        parser = argparse.ArgumentParser(description='Zero-shot Sketch Based Retrieval',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # Positional arguments
        parser.add_argument(
            "dataset",
            type=str,
            choices=["sketchy_extend", "tuberlin_extend", "both"],
            help="Choose between (Sketchy)."
        # Model parameters
        parser.add_argument('--data_path', '-dp', type=str, default='io/data/raw', help='Dataset root path.')
        parser.add_argument('--emb_size', type=int, default=256, help='Embedding Size.')
        parser.add_argument('--grl_lambda', type=float, default=0.5, help='Lambda used to normalize the GRL layer.')
        parser.add_argument('--nopretrain', action='store_true', help='Loads a pretrained model (Default: False).')
        # Optimization options
        parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
        parser.add_argument('--batch_size', '-b', type=int, default=10, help='Batch size.')
        parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='The Learning Rate.')
        parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
        parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
        parser.add_argument('--schedule', type=int, nargs='+', default=[],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--w_domain', type=float, default=1, help='Domain loss Weight.')
        parser.add_argument('--w_triplet', type=float, default=1, help='Triplet loss Weight.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--save', '-s', type=str, default='io/models', help='Folder to save checkpoints.')
        parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
        parser.add_argument('--early_stop', '-es', type=int, default=20, help='Early stopping epochs.')
        # Acceleration
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
        # i/o
        parser.add_argument('--log', type=str, default='io/models/', help='Log folder.')
        parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                            help='How many batches to wait before logging training status')
        parser.add_argument('--attn', action='store_false', help='Attention module (Default: True).')
        parser.add_argument('--plot', action='store_true', help='Qualitative results (Default: False).')

        # Test
        if test:
            parser.add_argument('--num_retrieval', type=int, default=10, help='Number of images to be retrieved')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
