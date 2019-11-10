import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader
from config import opt

""" 数据集：DogCat（继承自Dataset）"""
'''提供__init__，__getitem__，__len__方法'''
class DogCat(data.Dataset):
    '''
    self.imgs: 经排序过的数据集的完整路径（list）
    self.test: 是否为test set
    self.transforms：操作序列，可以使用默认操作或认为指定
    '''
    def __init__(self, root, transforms=None, train=True, test=False):  # train=True训练集；test=True测试集，否则验证集
        self.test = test
        # （1）数据地址获取与sort
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        '''数据集示例：（E:\\SOURCE\\kaggle-GogCat）'''
        '''test1: 12500.jpg（1~12500）'''
        '''train: dog.12499.jpg（0~12499）'''
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('\\')[-1])) # lambda只是指定sort规则
        else:  # imgs存的仍然是完整路径
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # （2）划分训练验证集（Validation : Test = 3 : 7）
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        # （3）transforms
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if self.test or not train:  # test or validation set
                self.transforms = T.Compose([
                    T.Resize(224),  # Resize the input PIL Image to the given size.
                    T.CenterCrop(224),  # Crops the given PIL Image at the center.
                    T.ToTensor(),
                    normalize
                ])
            else:  # train set，在训练集中加入随机裁剪和随机反转等
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),  # Crop the given PIL Image to random size and aspect ratio.
                    T.RandomHorizontalFlip(0.5), # Horizontally flip the given PIL Image randomly with a given probability.
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        '''
        :param index: 索引
        :return:
            test set: PIL图片 + 序号（如10,jpg返回10)
            else : PIL图片 + label（1：dog， 0：cat）
        '''
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('\\')[-1])
        else:
            label = 1 if 'dog' in img_path.split('\\')[-1] else 0
        # print(img_path)
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    path = 'E:/SOURCE/kaggle-GogCat/train'
    train_data = DogCat(path, train=True, test=False)
    val_data = DogCat(path, train=False, test=False)
    train_dataloader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=False,
                                  num_workers=opt.num_workers)
    val_dataloader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    for data,label in train_dataloader:
        print(label)

    print("*********************************************************")
    for data, label in val_dataloader:
        print(label)