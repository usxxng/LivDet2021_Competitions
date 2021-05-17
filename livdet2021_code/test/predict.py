import os
os.environ["PYTORCH_JIT"] = "0"
import sys

# from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC
from models import Effnet_MMC

from dataset1 import get_df_stone, get_transforms, MMC_ClassificationDataset

from utils.util import *

from tqdm import tqdm

# def parse_args -> class
# Can be used as dictionary (ClassInstance.__dict__)
class ModelInfo:
    def __init__(self):
        self.kernel_type = '5fold_b5_10ep'

        #self.data_dir = './data/'
        #self.data_folder = './data/'

        self.image_size = 456
        self.enet_type = 'tf_efficientnet_b5_ns' #tf_efficientnet_b3_ns
        self.batch_size = 4
        self.num_workers = 0
        self.out_dim = 2
        self.use_amp = True
        self.use_ext = True
        self.k_fold = 5

        self.use_meta = False #True
        self.DEBUG = True
        self.model_dir = '.\\weights'
        self.log_dir = './logs'
        self.sub_dir = './subs'
        self.eval = "best" #choices=['best', 'best_no_ext', 'final']
        self.n_test = 1
        self.CUDA_VISIBLE_DEVICES = '0'
        self.n_meta_dim = '512,128'

        
        self.source_dir = sys.argv[2]
        self.output_dir = sys.argv[3]

        print('--Get Model Info--')


args = ModelInfo()

def main():
    df_test, meta_features, n_meta_features, target_idx = get_df_stone(
        k_fold = args.k_fold,
        source_path = args.source_dir,
        out_dim = args.out_dim,
        use_meta = args.use_meta,
        use_ext = args.use_ext
    )

    print(df_test)
    print(meta_features)
    print(n_meta_features)
    print(target_idx)

    transforms_train, transforms_val = get_transforms(args.image_size)

    print(transforms_val)

    if args.DEBUG:
        df_test = df_test.sample(args.batch_size * 3)

    print(df_test)

    dataset_test = MMC_ClassificationDataset(df_test, 'test', meta_features, transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

    PROBS = []
    folds = range(args.k_fold)
    for fold in folds:
    
        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
            print('hi')
        elif args.eval == 'best_no_ext':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_no_ext_fold{fold}.pth')
        if args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

        model = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )

        model = model.to(device)

        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
            print('GOOD!!!!')
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)

        model.eval()

        PROBS = []
        TARGETS = []
        with torch.no_grad():
            for (data) in tqdm(test_loader):

                if args.use_meta:
                    data, meta = data
                    data, meta = data.to(device), meta.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(args.n_test):
                        l = model(get_trans(data, I), meta)
                        probs += l.softmax(1)
                else:
                    data = data.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(args.n_test):
                        l = model(get_trans(data, I))
                        probs += l.softmax(1)

                probs /= args.n_test

                PROBS.append(probs.detach().cpu())

        PROBS = torch.cat(PROBS).numpy()


        #mkcsv
        df_test['target'] = PROBS[:, target_idx]

        change_txt = open(sys.argv[3],'w')

        for i in df_test['target']:
            change_txt.write(str(int(float(i)*100))+'\n')

        change_txt.close()
        

if __name__ == '__main__':
    ## need some try-catch to argv len

    # os.makedirs(args.sub_dir, exist_ok=True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if 'efficientnet' in args.enet_type:
        ModelClass = Effnet_MMC
        print('efficientnet')
    # elif args.enet_type == 'resnest101':
    #     ModelClass = Resnest_MMC
    #     print('resnest101')
    # elif args.enet_type == 'seresnext101':
    #     ModelClass = Seresnext_MMC
    #     print('seresnext101')
    else:
        raise NotImplementedError()

    print(args.__dict__)

    device = torch.device('cuda')

    main()

