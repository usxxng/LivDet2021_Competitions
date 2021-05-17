import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

'''
image classification용 CSV 파일 만들때 주의할점
아래 두개는 반드시 포함해야한다. 

target: 클래스 번호. 예: {0, 1}
image_name: 이미지 파일 이름

'''

def get_df_stone(k_fold, source_path, out_dim = 1, use_meta = False, use_ext = False):
    '''

    ##### get DataFrame
    데이터베이스 관리하는 CSV 파일을 읽어오고, 교차 validation을 위해 분할함
    stone 데이터셋을 위해 수정된 함수

    :param k_fold: argument에서 받은 k_fold 값
    :param out_dim: 네트워크 출력 개수
    :param data_dir: 데이터 베이스 폴더
    :param data_folder: 데이터 폴더
    :param use_meta: meta 데이터 사용 여부
    :param use_ext: 외부 추가 데이터 사용 여부

    :return:
    :target_idx 양성을 판단하는 인덱스 번호
    '''
    print('--source_path--')
    print(source_path)
    df_test = pd.read_csv(source_path, delimiter='\n', names=['filepath'])

    '''
    ####################################################
    메타 데이터를 사용하는 경우 (나이, 성별 등)
    ####################################################
    '''
    if use_meta:
        df_train, df_test, meta_features, n_meta_features = get_meta_data_stoneproject(df_train, df_test)
    else:
        meta_features = None
        n_meta_features = 0


    '''
    ####################################################
    class mapping - 정답 레이블을 기록 (csv의 target)
    ####################################################
    '''
    # target2idx = {d: idx for idx, d in enumerate(sorted(df_train.target.unique()))}
    # df_train['target'] = df_train['target'].map(target2idx)
    #
    # # test data의 레이블 정보 (없는 경우엔 필요없음)
    # if 'target' in df_test.columns:
    #     df_test['target'] = df_test['target'].map(target2idx)

    # CSV 기준 정상은 0, 비정상은 1
    target_idx = 0

    return df_test, meta_features, n_meta_features, target_idx


def get_meta_data_stoneproject(df_train, df_test):
    '''
    ####################################################
    메타 데이터를 사용할 경우 세팅 함수
    (이미지를 표현하는 정보: 성별, 나이, 사람당 사진수, 이미지 크기 등)
    ####################################################
    '''

    # One-hot encoding of targeted feature
    concat = pd.concat([df_train['targeted_feature'], df_test['targeted_feature']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)

    # Sex features - 1과 0으로 변환
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)

    # Age features - [0,1] 사이의 값으로 변환
    df_train['age'] /= 90
    df_test['age'] /= 90
    df_train['age'] = df_train['age'].fillna(0)
    df_test['age'] = df_test['age'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)

    # n_image per user
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)

    # image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
        train_sizes[i] = os.path.getsize(img_path)
    df_train['image_size'] = np.log(train_sizes)
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)

    # df_train.columns에서 모든 정보를 가져옴
    meta_features = ['sex', 'age', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features




class MMC_ClassificationDataset(Dataset):
    '''
    MMC_ClassificationDataset 클래스
    일반적인 이미지 classification을 위한 데이터셋 클래스
        class 내가만든_데이터셋(Dataset):
            def __init__(self, csv, mode, meta_features, transform=None):
                # 데이터셋 초기화

            def __len__(self):
                # 데이터셋 크기 리턴
                return self.csv.shape[0]

            def __getitem__(self, index):
                # 인덱스에 해당하는 이미지 리턴
    '''

    def __init__(self, csv, mode, meta_features, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode # train / valid
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 이미지 tranform 적용
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        #image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  ### 이부분!!###
        image = image.transpose(2, 0, 1)

        # 메타 데이터를 쓰는 경우엔 image와 함께 텐서 생성
        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            # Test 의 경우 정답을 모르기에 데이터만 리턴
            return data
        else:
            # training 의 경우 CSV의 스톤여부를 타겟으로 보내줌
            return data, torch.tensor(self.csv.iloc[index].target).long(), row.id



def get_transforms(image_size):
    '''
    albumentations 라이브러리 사용함
    https://github.com/albumentations-team/albumentations

    TODO: FAST AUTO AUGMENT
    https://github.com/kakaobrain/fast-autoaugment
    DATASET의 AUGMENT POLICY를 탐색해주는 알고리즘

    TODO: Unsupervised Data Augmentation for Consistency Training
    https://github.com/google-research/uda

    TODO: Cutmix vs Mixup vs Gridmask vs Cutout
    https://www.kaggle.com/saife245/cutmix-vs-mixup-vs-gridmask-vs-cutout

    '''
    transforms_train = albumentations.Compose([
        # albumentations.Transpose(p=0.5),
        # albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),


        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        # albumentations.GaussianBlur(blur_limit=3, p=1.0),
        #albumentations.MedianBlur(blur_limit=(attack_strength, attack_strength), always_apply=True),
        #albumentations.ToGray(always_apply=True),
        # albumentations.GaussianBlur(blur_limit=(attack_strength, attack_strength),always_apply=True),
        # albumentations.GaussNoise(var_limit=(attack_strength * 10.0, attack_strength * 10.0),always_apply=True),
        #albumentations.ChannelShuffle(always_apply=True),
        albumentations.Normalize()
    ])
    return transforms_train, transforms_val
