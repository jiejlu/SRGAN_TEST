from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize


def is_image_file(filename):  # 判断是否为图片
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):  # 计算有效的裁剪大小，确保裁剪后的大小是缩放的整数倍
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):  # 随机裁剪
    return Compose([
        RandomCrop(crop_size),
        ToTensor()
    ])


def train_lr_transform(crop_size, upscale_factor):  # 缩小图片
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():  # 调整图像大小后中心裁剪为400×400
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])



class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))  # 对原始高分辨率图像裁剪至指定大小
        lr_image = self.lr_transform(hr_image)  # 再将裁剪后的高分辨率图像缩小
        return lr_image, hr_image
        # 返回元组（包含1.缩小后得到的低分辨图片 2.原始高分辨图片）

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # self.normalize_hr = normalize_hr()

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size  # 获取测试集中高分辨率图像的尺寸
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)  # 根据小的尺寸计算有效裁剪大小
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)  # 高分辨率图片中心裁剪
        lr_image = lr_scale(hr_image)  # 缩小裁剪后的高分辨率图像
        hr_restore_img = hr_scale(lr_image)  # 插值得到插值型的高分辨率图

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
        # 返回元组（包含1.验证集中缩小得到的低分辨图片 2.相应低分辨图片双三次插值得到的高分辨图片 3.高分辨图片）

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]


    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)  # 低分辨率图片进行双三次插值放大的图片

        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
        # 返回元组（包含1.测试数据集中的低分辨率测试图片 2.低分辨率测试图片双三次插值法获得的高分辨图片 3.测试数据集中的原始的高分辨图片）

    def __len__(self):
        return len(self.lr_filenames)
