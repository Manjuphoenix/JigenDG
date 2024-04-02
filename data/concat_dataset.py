import bisect
import warnings

from torch.utils.data import Dataset

# This is a small variant of the ConcatDataset class, which also returns dataset index
from data.JigsawLoader import JigsawTestDatasetMultiple


class ConcatDataset(Dataset):           # This is called while loading the training or validation dataset.........
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):       # e is of type data.JigsawLoader.JigsawDataset which is an object....
        r, s = [], 0
        for e in sequence:         # This loads one image after another image............
            l = len(e)
            r.append(l + s)              # r will hold the total length of the entire dataset be it train or test......
            s += l
            # print("E: ", e, " L: ", l, " R: ", r, " S: ", s, "----------------------------------")
        return r

    def isMulti(self):
        return isinstance(self.datasets[0], JigsawTestDatasetMultiple)

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        # print("----------------------------", datasets)
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)          # This holds the total size of train, val and test dataset...........

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # print(self.cumulative_sizes, "**********************88")
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)           # dataset_idx holds the information of if the dataset is for trian (0) or validation (1) or testing (2)...
        # print(self.cumulative_sizes, "cummulative sizes", idx, dataset_idx, "idx and dataset_idx................")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        # print(self.cumulative_sizes, idx, dataset_idx, sample_idx, "----------------")
        return self.datasets[dataset_idx][sample_idx], dataset_idx

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
