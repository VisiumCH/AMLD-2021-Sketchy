import argparse


class ApiOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Api for Sketchy Web App",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("name", type=str, help="Name of the training")
        parser.add_argument("--epoch", type=str, default="00012", help="Epoch to load model")
        parser.add_argument("--log", type=str, default="io/models/", help="Log folder")
        self.parser = parser
    
    def parse(self):
        return self.parser.parse_args()
