# -*- coding: utf-8 -*-
"""
    Parse input arguments
"""

import argparse


class Options:
    def __init__(self, test=False):
        # MODEL SETTINGS
        parser = argparse.ArgumentParser(
            description="Zero-shot Sketch Based Retrieval",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        # Positional arguments
        parser.add_argument(
            "dataset",
            type=str,
            choices=["sketchy_extend", "tuberlin_extend", "quickdraw_extend"],
            help="Choose between (Sketchy).",
        )
        # Model parameters
        parser.add_argument(
            "--data_path",
            "-dp",
            type=str,
            default="io/data/raw/",
            help="Dataset root path.",
        )
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        parser.add_argument('--save', '-s', type=str, default='io/models', help='Folder to save checkpoints.')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
