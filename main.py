import argparse
import os

import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(
    prog="Image Sorter",
    description="Sorts the images in the specified directory by color content",
)

parser.add_argument(
    "directory",
    type=str,
    help="The directory of images to sort",
)
parser.add_argument(
    "--recursive", "-r",
    default=False,
    type=bool,
    help="If set, all directories are recursively travelled, otherwise, they are ignored",
    dest="recursive",
)
parser.add_argument(
    "--thumbnail-size", "-t",
    default=32,
    type=int,
    help="The size of the thumbnail that all images will be compressed to as to compare images, larger sizes are more accurate but more computationally intensive",
    dest="thumbnail_size",
)


class Cluster:
    def __init__(self, filepath: str, thumbnail: np.array) -> None:
        self.filepaths = [filepath]
        self.thumbnail = thumbnail

    def add(self, other: Cluster) -> Cluster:
        self.thumbnail = (self.thumbnail * len(self.filepaths) + other.thumbnail * len(other.filepaths)) / (len(self.filepaths) + len(other.filepaths))
        self.filepaths += other.filepaths

        return self


class ClusterAdjacencyMatrix:
    def __init__(self, clusters: list[Cluster]) -> None:
        self.clusters: list[Cluster] = clusters
        self.n: int = len(clusters)
        self.adjmat: np.ndarray = np.zeros((self.n * (self.n - 1)) // 2)

    def _build(self):
        ...

    def _index(self, i: int, j: int) -> int:
        """Convert a pair of indices to a single index in one-dimensional array. This is
        for a upper triangular matrix with zeros on the diagonal.
        """
        if i > j:
            i, j = j, i

        return (i * self.n) - ((i * (i + 1)) // 2) + j - i - 1

    def _ij(self, index: int) -> tuple[int, int]:
        i = int(floor(self.n - 2 - np.sqrt(-8 * index + 4 * self.n * (self.n - 1) - 7) / 2. - 0.5))
        j = index + i + 1 - self.n * (self.n - 1) // 2 + (self.n - i) * ((self.n - i) - 1) // 2

        return i, j

    def set(self, i: int, j: int, value: float):
        if i == j:
            raise ValueError("Adjacency matrix should not be set as it is always 0")

        index = self._index(i, j)
        self.adjmat[index] = value

    def get(self, i: int, j: int) -> float:
        if i == j:
            return 0

        index = self._index(i, j)
        
        return self.adjmat[index]


def get_fingerprint(filepath: str, thumbnail_size: int) -> float:
    print(f"Calculating fingerprint for {filepath}")
    with Image.open(filepath) as image:
        image.resize((thumbnail_size, thumbnail_size))

        # band_r \in {0, 1}, band_g \in {1, 2}, band_b \in {2, 3}
        band_r = np.array(image.getdata(band=0)) / 255
        band_g = (np.array(image.getdata(band=1)) / 255) + 1
        band_b = (np.array(image.getdata(band=2)) / 255) + 2

    bands = band_r + band_g + band_b
    fingerprint = (bands + np.arange(len(bands))).sum()

    return fingerprint


def main() -> None:
    args = parser.parse_args()
    fingerprints = []

    for filename in os.listdir(args.directory):
        filepath = os.path.join(args.directory, filename)

        try:
            fingerprint = get_fingerprint(filepath, args.thumbnail_size)
        except IsADirectoryError as error:
            # if args.recursive: image_sort(filepath)
            continue

        fingerprints.append({"filename": filename, "fingerprint": fingerprint})

    fingerprints.sort(key=lambda x: x.get("fingerprint", 0))
    fingerprints_adj = np.array(
        [
            [abs(row["fingerprint"] - col["fingerprint"]) for col in fingerprints]
            for row in fingerprints
        ]
    )
    breakpoint()

    # After sorting, create a matrix that calculates the difference between all images
    # and cluster them


if __name__ == "__main__":
    main()
