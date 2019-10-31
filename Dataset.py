import os
import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image
from torchvision import transforms


class YTBDatasetCNN(torch.utils.data.Dataset):
    def __init__(self, path="../aligned_images_DB", img_size=160, person_frames=400, seed=40, step=4,
                 **kwargs):

        """
         Initia the Dataset, load all image path to array.
         Args:
             path (string): Directory with all face images.
             train : If traning model
             transform (callable, optional): Optional transform to be applied
                 on a sample.
             person_frames: take how much frames each person (person_frames=person_video*videp_frames)
         """

        # super().__init__(**kwargs)
        path = os.path.abspath(path)
        self.path = path
        self.step = step
        self.person_frames = person_frames
        self.seed = seed
        np.random.seed(seed=self.seed)

        self.transforms1 = transforms.Compose([
            #                 transforms.Resize((img_size,img_size)),
            transforms.CenterCrop(int(img_size)),
            transforms.ToTensor(),
            #                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # get all directories for different people  (include absolute path)
        all_dir = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        all_dir_path = []
        for person_dir in all_dir:  # person_dir  一个人的视频所在的大文件夹
            path_video_dirs = []
            video_dirs = [d for d in os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, d))]
            # video_dirs 一个人的所有视频的n个文件夹(not include absolute path)
            video_dirs.sort()
            # 遍历每个人的所有的视频
            for video_dir in video_dirs:
                face_dirs = []
                img_dir = os.path.join(person_dir, video_dir)  # include absolute path
                fs = [f for f in os.listdir(img_dir) if
                      f.lower().endswith('.jpg')]  # get all img  (not include absolute path)
                fs.sort()
                for f in fs:  # 遍历每个视频的所有图片
                    # TODO:   filter some image with a stride
                    #                     index=np.random.choice(len(video_dirs), 1)[0]
                    #                     os.path.join(img_dir,fs[index])
                    face_dirs.append(os.path.join(img_dir, f))

                path_video_dirs.append(face_dirs)
            all_dir_path.append(path_video_dirs)  # put all people face together
        self.all_dir_path = all_dir_path

    def __len__(self):
        return len(self.all_dir_path) * self.person_frames

    def __getitem__(self, idx):
        # 设置每个人取 400张图片
        person_idx = idx // self.person_frames
        label = person_idx

        person_dir = self.all_dir_path[person_idx]

        idx = np.random.choice(len(person_dir))
        video_dir = person_dir[idx]

        idx = np.random.choice(len(video_dir))
        imgpath = video_dir[idx]
        name = os.path.basename(os.path.dirname(os.path.dirname(imgpath)))

        img = Image.open(imgpath)
        img = self.transforms1(img)
        img = torch.tensor(img)
        return img, label, name


class YTBDatasetVer(torch.utils.data.Dataset):
    """
    youtube数据集加载 verification task使用
    """

    def __init__(self, csv_file='../splits.txt', root_dir='../aligned_images_DB', img_size=160, num_frames=100,
                 train=True, transform=True, test_set_index=10, seed=66, img_item=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training = train
        self.transform = transform
        if img_item:
            self.num_frames = 1
        else:
            self.num_frames = num_frames
        self.img_item = img_item
        split_df = pd.read_csv(csv_file)
        if self.training:
            self.split_df = split_df[split_df['split number'] != test_set_index]
        else:
            self.split_df = split_df[split_df['split number'] == test_set_index]
        self.root_dir = root_dir
        self.transform = transform
        self.seed = seed
        self.rgb_transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.CenterCrop(int(img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        np.random.seed(seed=self.seed)

    def __len__(self):
        return len(self.split_df)

    def load_face_from_dir(self, dir_img):
        # get all img file name
        fs = [f for f in os.listdir(dir_img) if f.lower().endswith('.jpg')]
        img_group = []
        i = 0
        while i < self.num_frames:
            #             np.random.shuffle(fs)
            index = np.random.choice(len(fs))
            #             for f in fs:
            img = Image.open(os.path.join(dir_img, fs[index]))
            # if image has an alpha color channel, get rid of it
            if self.transform:
                img = self.rgb_transform(img)
            if img.shape[2] == 4:
                img = img[:, :, 0:3]
            if self.img_item:
                return torch.tensor(img)
            img_group.append(img)
            i += 1
            # get max frames
            if i >= self.num_frames - 1:
                break;
        data_face = torch.stack(img_group, dim=0)
        return data_face

    def load_video_from_dir(self, dir_img):
        # get all img file name
        fs = [f for f in os.listdir(dir_img) if f.lower().endswith('.jpg')]
        img_group = []
        i = 0
        while i < self.num_frames:
            #             np.random.shuffle(fs)
            index = np.random.choice(len(fs))
            #             for f in fs:
            img = Image.open(os.path.join(dir_img, fs[index]))
            # if image has an alpha color channel, get rid of it
            if self.transform:
                img = self.rgb_transform(img)
            if img.shape[2] == 4:
                img = img[:, :, 0:3]
            if self.img_item:
                return torch.tensor(img)
            img_group.append(img)
            i += 1
            if i >= self.num_frames - 1:
                break;
        data_face = torch.stack(img_group, dim=0)
        return data_face

    def __getitem__(self, idx):
        # Get first face
        face_dir = os.path.join(self.root_dir,
                                self.split_df.iloc[idx, 2].strip())
        data_face1 = self.load_face_from_dir(face_dir)

        # Get second face
        face_dir = os.path.join(self.root_dir,
                                self.split_df.iloc[idx, 3].strip())

        data_face2 = self.load_face_from_dir(face_dir)
        # Gat label of this data pair 1;same persopn 0: not same
        label = torch.tensor(float(self.split_df.iloc[idx, 4]))
        return data_face1, data_face2, label


class YTBDatasetCNN_RGBDiff(torch.utils.data.Dataset):
    def __init__(self, path="../aligned_images_DB", train=True, img_size=224, step=6, person_frames=400, seed=40,
                 **kwargs):

        """
         Initia the Dataset, load two kind of training source from image.
         Args:
             path (string): Directory with all face images.
             num_frames (int): Frames each person
             train : If traning model
             transform (callable, optional): Optional transform to be applied
                 on a sample.
             person_frames: take how much frames each person (person_frames=person_video*videp_frames)
         """

        # super().__init__(**kwargs)
        path = os.path.abspath(path)
        self.path = path
        self.step = step
        self.training = train
        self.person_frames = person_frames
        self.seed = seed
        np.random.seed(seed=self.seed)

        self.transforms1 = transforms.Compose([
            #                 transforms.Resize((256,256)),
            transforms.CenterCrop(int(img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # get all directories for different people  (include absolute path)
        all_dir = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        all_dir_path = []
        for person_dir in all_dir:  # person_dir  一个人的视频所在的大文件夹
            path_video_dirs = []
            video_dirs = [d for d in os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, d))]
            # video_dirs 一个人的所有视频的n个文件夹(not include absolute path)
            video_dirs.sort()
            # 遍历每个人的所有的视频
            for video_dir in video_dirs:
                face_dirs = []
                img_dir = os.path.join(person_dir, video_dir)  # include absolute path
                fs = [f for f in os.listdir(img_dir) if
                      f.lower().endswith('.jpg')]  # get all img  (not include absolute path)
                fs.sort()
                for f in fs:  # 遍历每个视频的所有图片
                    # TODO:   filter some image with a stride
                    #                     index=np.random.choice(len(video_dirs), 1)[0]
                    #                     os.path.join(img_dir,fs[index])
                    face_dirs.append(os.path.join(img_dir, f))

                path_video_dirs.append(face_dirs)
            all_dir_path.append(path_video_dirs)  # put all people face together
        self.all_dir_path = all_dir_path

    def __len__(self):
        return len(self.all_dir_path) * self.person_frames

    def __getitem__(self, idx):
        # 设置每个人取 400张图片
        person_idx = idx // self.person_frames
        label = person_idx

        person_dir = self.all_dir_path[person_idx]

        idx = np.random.choice(len(person_dir))
        video_dir = person_dir[idx]

        idx = np.random.choice(len(video_dir) - self.step)
        imgpath = video_dir[idx]
        imgpath2 = video_dir[idx + self.step]

        img = Image.open(imgpath)
        img = self.transforms1(img)
        img2 = Image.open(imgpath2)
        img2 = self.transforms1(img2)
        displacement = img - img2
        return img, displacement, label


class YTBDatasetVer_RGBdiff(torch.utils.data.Dataset):
    """
    youtube数据集加载 verification task使用
    """

    def __init__(self, csv_file='../splits.txt', root_dir='../aligned_images_DB', img_size=160, num_frames=100,
                 train=True, transform=True, test_set_index=10, step=6, seed=66, img_item=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training = train
        self.transform = transform
        if img_item:
            self.num_frames = 1
        else:
            self.num_frames = num_frames
        self.img_item = img_item
        split_df = pd.read_csv(csv_file)
        if self.training:
            self.split_df = split_df[split_df['split number'] != test_set_index]
        else:
            self.split_df = split_df[split_df['split number'] == test_set_index]
        self.root_dir = root_dir
        self.transform = transform
        self.seed = seed
        self.rgb_transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.CenterCrop(int(img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.step = step
        np.random.seed(seed=self.seed)

    def __len__(self):
        return len(self.split_df)

    def load_face_from_dir(self, dir_img):
        # get all img file name
        fs = [f for f in os.listdir(dir_img) if f.lower().endswith('.jpg')]
        img_group = []
        i = 0
        while i < self.num_frames:
            #             np.random.shuffle(fs)
            index = np.random.choice(len(fs))
            #             for f in fs:
            img = Image.open(os.path.join(dir_img, fs[index]))
            # if image has an alpha color channel, get rid of it
            if self.transform:
                img = self.rgb_transform(img)
            if img.shape[2] == 4:
                img = img[:, :, 0:3]
            if self.img_item:
                return torch.tensor(img)
            img_group.append(img)
            i += 1
            # get max frames
            if i >= self.num_frames - 1:
                break;
        data_face = torch.stack(img_group, dim=0)
        return data_face

    def load_video_from_dir(self, dir_img):
        # get all img file name
        fs = [f for f in os.listdir(dir_img) if f.lower().endswith('.jpg')]
        fs.sort()
        data_group = []
        i = 0
        while i < self.num_frames:
            #             np.random.shuffle(fs)
            idx = np.random.choice(len(fs) - self.step)
            imgpath = os.path.join(dir_img, fs[idx])
            imgpath2 = os.path.join(dir_img, fs[idx + self.step])
            img = Image.open(imgpath)
            # if image has an alpha color channel, get rid of it
            img2 = Image.open(imgpath2)
            if self.transform:
                img = self.rgb_transform(img)
                img2 = self.rgb_transform(img2)
            if img.shape[2] == 4:
                img = img[:, :, 0:3]
            if self.img_item:
                return img
            displacement = img - img2
            data = torch.cat((img, displacement), dim=0)
            data_group.append(data)
            i += 1
            if i >= self.num_frames - 1:
                break;
        data_face = torch.stack(data_group, dim=0)
        return data_face

    def __getitem__(self, idx):
        # Get first face
        face_dir = os.path.join(self.root_dir,
                                self.split_df.iloc[idx, 2].strip())
        data_face1 = self.load_video_from_dir(face_dir)

        # Get second face
        face_dir = os.path.join(self.root_dir,
                                self.split_df.iloc[idx, 3].strip())

        data_face2 = self.load_video_from_dir(face_dir)
        # Gat label of this data pair 1;same persopn 0: not same
        label = torch.tensor(float(self.split_df.iloc[idx, 4]))
        return data_face1, data_face2, label


if __name__ == "__main__":
    batch_size = 3
    dataset = YTBDatasetVer()
    dataload = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=2)
    g2 = iter(dataload)
    print(len(dataset))
    batch_size = 3
    x1, x2, y = next(g2)
    print('data shape', x1.shape, x2.shape, y)

    batch_size = 3
    dataset = YTBDatasetCNN()
    # dataset1 =YouTubeBeDataset()
    dataload = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    len(dataset)
    g2 = iter(dataload)
    x1, y, name = next(g2)
    print('data shape', x1.shape, y)
    print("name rank:", y[1], 'name:', name[1])
