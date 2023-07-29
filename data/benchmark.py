import os
import glob
import data


# for vairous benchmark datasets
class Benchmark(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root
        if phase == "test" and opt.test_dataset != "":
            self.name = opt.test_dataset.split('_')[1]
        else:
            self.name = opt.dataset.split('_')[1]

        dir_MASK, dir_IMG = self.get_subdir()
        self.MASK_paths = sorted(glob.glob(os.path.join(root, dir_MASK, "*.png")))
        if 'HKU' in dir_IMG:
            self.IMG_paths = sorted(glob.glob(os.path.join(root, dir_IMG, "*.png")))
        else:
            self.IMG_paths = sorted(glob.glob(os.path.join(root, dir_IMG, "*.jpg")))

        super().__init__(phase, opt)

    def get_subdir(self):

        if 'DUTS' in self.name:
            dir_IMG = self.name + '/DUTS-TE-Image/'
            dir_MASK = self.name + '/DUTS-TE-Mask/'
        else:
            dir_IMG = self.name + '/images/'
            dir_MASK = self.name + '/GT/'

        if 'SOD' in dir_IMG:
            dir_MASK = dir_IMG
        if 'PASCAL-S' in dir_IMG:
            dir_MASK = dir_IMG

        return dir_MASK, dir_IMG