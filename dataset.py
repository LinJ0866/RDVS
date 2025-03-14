import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np


def randomCrop(image, label, flow, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    label = Image.fromarray(label)
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), np.array(label.crop(random_region)), flow.crop(random_region), depth.crop(
        random_region)


class Dataset(data.Dataset):
    def __init__(self, datasets, dataset_root='../data', mode='train', transform=None, return_size=True):
        self.return_size = return_size
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        self.mode = mode
        for (i, dataset) in enumerate(datasets):

            if dataset == 'rdvs':
                lable_rgb = 'rgb'
                lable_depth = 'Depth'
                lable_gt = 'ground-truth'
                lable_flow = 'FLOW'

                if mode == 'train':
                    data_dir = os.path.join(dataset_root, 'RDVS/train')
                else:
                    data_dir = os.path.join(dataset_root, 'RDVS/test')
            elif dataset == 'vidsod_100':
                lable_rgb = 'rgb'
                lable_depth = 'depth'
                lable_gt = 'gt'
                lable_flow = 'flow'
                
                if mode == 'train':
                    data_dir = os.path.join(dataset_root, 'vidsod_100/train')
                    data_dir = '/home/linj/workspace/vsod/datasets/vidsod_100/train'
                else:
                    data_dir = os.path.join(dataset_root, 'vidsod_100/test')
            elif dataset == 'dvisal':
                lable_rgb = 'RGB'
                lable_depth = 'Depth'
                lable_gt = 'GT'
                lable_flow = 'flow'

                data_dir = os.path.join(dataset_root, 'DViSal_dataset/data')

                if mode == 'train':
                    dvi_mode = 'train'
                else:
                    dvi_mode = 'test_all'
            else:
                raise 'dataset is not support now.'
            
            if dataset == 'dvisal':
                with open(os.path.join(data_dir, '../', dvi_mode+'.txt'), mode='r') as f:
                    subsets = set(f.read().splitlines())
            else:
                subsets = os.listdir(data_dir)
            
            for video in subsets:
                video_path = os.path.join(data_dir, video)
                rgb_path = os.path.join(video_path, lable_rgb)
                depth_path = os.path.join(video_path, lable_depth)
                gt_path = os.path.join(video_path, lable_gt)
                flow_path = os.path.join(video_path, lable_flow)
                frames = os.listdir(rgb_path)
                frames = sorted(frames)
                for frame in frames[:-1]:
                    data = {}
                    data['img_path'] = os.path.join(rgb_path, frame)
                    if os.path.isfile(data['img_path']):
                        data['gt_path'] = os.path.join(gt_path, frame.replace('jpg', 'png'))
                        data['depth_path'] = os.path.join(depth_path, frame.replace('jpg', 'png'))
                        data['flow_path'] = os.path.join(flow_path, frame)
                        data['split'] = video
                        data['dataset'] = dataset
                        self.datas_id.append(data)
        self.transform = transform

    def __getitem__(self, item):

        assert os.path.exists(self.datas_id[item]['img_path']), (
            '{} does not exist'.format(self.datas_id[item]['img_path']))
        assert os.path.exists(self.datas_id[item]['gt_path']), (
            '{} does not exist'.format(self.datas_id[item]['gt_path']))
        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            pass
            # DUTS Depth
            # assert os.path.exists(self.datas_id[item]['depth_path']), (
            #     '{} does not exist'.format(self.datas_id[item]['depth_path']))
        else:
            assert os.path.exists(self.datas_id[item]['depth_path']), (
                '{} does not exist'.format(self.datas_id[item]['depth_path']))
            assert os.path.exists(self.datas_id[item]['flow_path']), (
                '{} does not exist'.format(self.datas_id[item]['flow_path']))

        image = Image.open(self.datas_id[item]['img_path']).convert('RGB')
        label = Image.open(self.datas_id[item]['gt_path']).convert('L')
        label = np.array(label)

        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            flow = np.zeros((image.size[1], image.size[0], 3))
            flow = Image.fromarray(np.uint8(flow))
            depth = np.zeros((image.size[1], image.size[0], 3))
            depth = Image.fromarray(np.uint8(depth))
            # DUTS Depth
            # depth = Image.open(self.datas_id[item]['depth_path']).convert('RGB')
        else:
            flow = Image.open(self.datas_id[item]['flow_path']).convert('RGB')
            depth = Image.open(self.datas_id[item]['depth_path']).convert('RGB')

        if label.max() > 0:
            label = label / 255

        w, h = image.size
        size = (h, w)

        sample = {'image': image, 'label': label, 'flow': flow, 'depth': depth}
        if self.mode == 'train':
            sample['image'], sample['label'], sample['flow'], sample['depth'] = randomCrop(sample['image'],
                                                                                           sample['label'],
                                                                                           sample['flow'],
                                                                                           sample['depth'])
        else:
            pass

        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)
        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            sample['flow'] = torch.zeros((3, 448, 448))
            # DUTS Depth
            sample['depth'] = torch.zeros((3, 448, 448))
        name = self.datas_id[item]['gt_path'].split('/')[-1]
        sample['dataset'] = self.datas_id[item]['dataset']
        sample['split'] = self.datas_id[item]['split']
        sample['name'] = name

        return sample

    def __len__(self):
        return len(self.datas_id)
