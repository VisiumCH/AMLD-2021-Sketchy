from src.models.inference.inference import Inference
import matplotlib.pyplot as plt
import numpy as np


class ApiInference(Inference):
    def __init__(self, args, dataset_type):
        super().__init__(args, dataset_type)

    def get_closest(self, number):
        images, labels = [], []
        for index in range(number):
            image, label = self.prepare_image(index)
            images.append(image)
            labels.append(label)
        return images, labels

    def get_attention(self, sk):
        """ Find the closest images of a sketch and plot it """
        self.get_heatmap()

        fig, ax = plt.subplots(frameon=False)

        ax.imshow(sk, aspect="auto")
        ax.imshow(255 * self.heat_map, alpha=0.7, cmap="Spectral_r", aspect="auto")
        ax.axis("off")
        plt.tight_layout()

        fig.canvas.draw()
        attention = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        attention = attention.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.show()

        fig.clf()
        plt.close("all")

        return attention
