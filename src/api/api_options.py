import argparse


class ApiOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Api for Sketchy Web App",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("name", type=str, help="Name of the training")
        parser.add_argument(
            "--ngpu", type=int, default=1, help="0 = CPU, 1 = CUDA, 1 < DataParallel"
        )
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
