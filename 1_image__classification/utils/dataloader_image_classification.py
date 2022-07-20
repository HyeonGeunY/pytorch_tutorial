#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import os.path as osp
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image


# In[4]:


class ImageTransform():
    """
    화상 전처리 클래스. 훈련시, 검증시의 동작이 다르다.
    화상 크기를 리사이즈하고, 색상을 표준화한다.
    훈련시에는 RandomResizedCrop과 RandomHorizontalFlip으로 데이터를 확장한다.

    Attributes
    ----------
    resize : int
        리사이즈 대상 화상의 크기.
    mean : (R, G, B)
        각 색상 채널의 평균값.
    std : (R, G, B)
        각 색상 채널의 표준 편차.
    """
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale =(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), # torch로 수학적 연산을 하기 전 텐서로 바꿔준다.
                transforms.Normalize(mean, std) 
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize), # 크기를 맞춰주기 위한 crop
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드를 지정.
        """
        
        return self.data_transform[phase](img)


# In[6]:

def make_datapath_list(phase="train"):
    """
    데이터의 경로를 저장한 리스트를 작성한다.

    Parameters
    ----------
    phase : 'train' or 'val'
        훈련 데이터 또는 검증 데이터를 지정

    Returns
    -------
    path_list : list
        데이터 경로를 저장한 리스트
    """
    
    rootpath = './data/hymenoptera_data/'
    target_path = osp.join(rootpath+phase+"/**/*.jpg")
    print(target_path)
    
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)
        
    return path_list


# In[3]:


class HymenopteraDataset(data.Dataset):
    """
    개미와 벌 화상의 Dataset 클래스, Pytorch의 Dataset 클래스를 상속한다.
    
    Attributes
    --------------
    file_list : 리스트
        화상 경로를 저장한 리스트
    transform : object
        전처리 클래스의 인스턴스
    phase : 'train' or 'test'
        학습인지 훈련인지를 결정한다.. 
    
    """
    
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        """화상 개수를 반환"""
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        전처리한 화상의 Tensor 형식의 데이터와 라벨을 취득
        """
        
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        img_transformed = self.transform(img, self.phase)
        
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]
            
        # 라벨을 숫자로 변경
        
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1
            
            
        return img_transformed, label
        

