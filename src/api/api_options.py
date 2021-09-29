import argparse


class ApiOptions:
    def __init__(self):
        self.default = argparse.Namespace(
            name="quickdraw_training",
            ngpu=1,
            port=5000,
        )
        parser = argparse.ArgumentParser(
            description="Api for Sketchy Web App",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--name",
            default=self.default.name,
            type=str,
            help="Name of the training",
        )
        parser.add_argument(
            "--ngpu", type=int, default=self.default.ngpu, help="0 = CPU, 1 = CUDA, 1 < DataParallel"
        )
        parser.add_argument("--port", type=int, default=self.default.port, help="Flask API port")
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
