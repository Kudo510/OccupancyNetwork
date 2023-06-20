"""PyTorch datasets for loading ShapeNet voxels and ShapeNet point clouds from disk"""
import torch
from pathlib import Path
import json
import numpy as np
import trimesh

from custom_occnet.data.binvox_rw import read_as_3d_array
from custom_occnet.util.misc import remove_nans

class ShapeNetVox(torch.utils.data.Dataset):
    """
    Dataset for loading ShapeNet Voxels from disk
    """

    num_classes = 13  # we'll be performing a 13 class classification problem
    dataset_path = Path("custom_occnet/data/ShapeNet")  # path to voxel data - make sure you've downloaded the ShapeNet voxel models to the correct path
    class_name_mapping = json.loads(Path("custom_occnet/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        """
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']

        self.items = Path(f"custom_occnet/data/splits/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of data corresponding to the shape. In particular, this dictionary has keys
                 "name", given as "<shape_category>/<shape_identifier>",
                 "voxel", a 1x32x32x32 numpy float32 array representing the shape
                 "label", a number in [0, 12] representing the class of the shape
        """
        # TODO Get item associated with index, get class, load voxels with ShapeNetVox.get_shape_voxels
        item = self.items[index]
        

        # Hint: since shape names are in the format "<shape_class>/<shape_identifier>", the first part gives the class

        item_class = item.split("/")[0]
        #print(item_class)
        # read voxels from binvox format on disk as 3d numpy arrays
        voxels = ShapeNetVox.get_shape_voxels(item)
        return {
            "name": item,
            "voxel": voxels[np.newaxis, :, :, :],  # we add an extra dimension as the channel axis, since pytorch 3d tensors are Batch x Channel x Depth x Height x Width
            "label": ShapeNetVox.classes.index(item_class)  # label is 0 indexed position in sorted class list, e.g. 02691156 is label 0, 02828884 is label 1 and so on.
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO Implement
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['voxel'] = batch['voxel'].to(device)
        batch['label'] = batch['label'].to(device)

    @staticmethod
    def get_shape_voxels(shapenet_id):
        """
        Utility method for reading a ShapeNet voxel grid from disk, reads voxels from binvox format on disk as 3d numpy arrays
        :param shapenet_id: Shape ID of the form <shape_class>/<shape_identifier>, e.g. 03001627/f913501826c588e89753496ba23f2183
        :return: a numpy array representing the shape voxels
        """
        with open(ShapeNetVox.dataset_path / shapenet_id / "model.binvox", "rb") as fptr:
            voxels = read_as_3d_array(fptr).astype(np.float32)
        return voxels


class ShapeImplicit(torch.utils.data.Dataset):
    """
    Dataset for loading deep sdf training samples
    """

    dataset_path = Path("custom_occnet/data/ShapeNet")  # path to sdf data for ShapeNet sofa class - make sure you've downloaded the processed data at appropriate path

    def __init__(self, num_sample_points, split):
        """
        :param num_sample_points: number of points to sample for sdf values per shape
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']

        self.num_sample_points = num_sample_points
        #self.items = Path(f"custom_occnet/data/splits/sofas/{split}.txt").read_text().splitlines()  # keep track of shape identifiers based on split
        self.items = Path(f"custom_occnet/data/splits/{split}.txt").read_text().splitlines()
    def __getitem__(self, index):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of sdf data corresponding to the shape. In particular, this dictionary has keys
                 "name", shape_identifier of the shape
                 "indices": index parameter
                 "points": a num_sample_points x 3  pytorch float32 tensor containing sampled point coordinates
                 "sdf", a num_sample_points x 1 pytorch float32 tensor containing sdf values for the sampled points
        """

        # get shape_id at index
        item = self.items[index]

        # get path to sdf data
        sdf_samples_path = ShapeImplicit.dataset_path / item / "points.npz"

        # read points and their sdf values from disk
        # TODO: Implement the method get_sdf_samples
        points, occupancies = self.get_points_occupancies(sdf_samples_path)

        return {
            "name": item,       # identifier of the shape
            "indices": index,   # index parameter
            "points": points,   # points, a tensor with shape num_sample_points x 3
            "occupancies": occupancies  # sdf values, a tensor with shape num_sample_points x 1
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO: Implement
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['sdf'] = batch['sdf'].to(device)
        batch['indices'] = batch['indices'].to(device)

    def get_points_occupancies(self, path_to_sdf):
        """
        Utility method for reading an sdf file; the SDF file for a shape contains a number of points, along with their sdf values
        :param path_to_sdf: path to sdf file
        :return: a pytorch float32 torch tensor of shape (num_sample_points, 4) with each row being [x, y, z, sdf_value at xyz]
        """
        npz = np.load(path_to_sdf)
        # keys = npz.files
        # for key in keys:
        #     print(key)
        points= npz["points"]
        occupancies=npz["occupancies"]
        #print (occupancies[0:20],occupancies.shape)

        return points,occupancies


