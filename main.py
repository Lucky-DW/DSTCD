from dataloader.sar_opt_datasets import Dataset_self
import torch
import os
import argparse
import random
import numpy as np
from utils.metric import get_metric, get_confusion_matrix
from utils.LRadjust import PolyLR
import torch.nn as nn
import time
from utils.loss import BoundaryLoss, torch_MS_SSIM, SurfaceLoss, GradLoss
from model.CDNet import opt_sar_CDNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置随机种子函数
def set_seed(seed=42):
    """
    设置随机种子以确保实验可重复
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 设置CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子已设置为: {seed}")

#参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--train_path',
                        default='/home/september/code/3/dataset/3/noto/2/train',
                        help='train data path')
    parser.add_argument('--val_path',
                        default='/home/september/code/3/dataset/3/noto/2/val',
                        help='val data path')
    parser.add_argument('--train_batch_size', default=16, help='validate data path')
    parser.add_argument('--val_batch_size', default=2, help='validate data path')
    parser.add_argument('--ckpt_path', default=None, help='checkpoint path')
    parser.add_argument('--work_dir', default='result', help='the dir to save checkpoint and logs')
    parser.add_argument('--epoch', default=100, help='Total Epoch')
    parser.add_argument('--lr', default=0.0001, help='Initial learning rate')
    parser.add_argument('--Transfer_net', default=True, help='Transfer_net')
    parser.add_argument('--Structure', default=True, help='Structure moudle')
    parser.add_argument('--Sementic', default=True, help='Sementic moudle')
    parser.add_argument('--path_name', default=None, help='path_name')
    # 添加随机种子参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)
    
    # 设置DataLoader的随机种子（通过worker_init_fn）
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset=Dataset_self(args.train_path),
        batch_size=args.train_batch_size,
        shuffle=True, 
        num_workers=4,
        worker_init_fn=worker_init_fn  # 添加worker初始化函数
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset=Dataset_self(args.val_path),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn
    )
    
    model = opt_sar_CDNet(Transfer=args.Transfer_net,
                          Structure=args.Structure, Sementic=args.Sementic)
    model.to(device)
    
    loss_fuc = BoundaryLoss()
    loss_fuc.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    poly_lr_scheduler = PolyLR(optimizer, max_iter=args.epoch)
    
    minoitor = float('inf')
    begin_time = time.time()
    save_path = os.path.join(args.work_dir, args.path_name)
    os.mkdir(save_path)
    
    # 在日志文件中记录随机种子
    with open(os.path.join(save_path, str(begin_time) + '.txt'), 'a') as file:
        file.write(f'Random seed: {args.seed}\n')
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        file.write('Epoch'+'\t'+'lr' + '\t' + 'Train_Loss' + '\t' + 'Val_Loss' +
                    '\t' + 'Accuracy' + '\t' + 'F1_score' + '\t' + 'Kappa' + '\t'
                   + 'IoU' + '\n')
        file.close()
    
    for i in range(1, args.epoch+1):
        Train_Loss = 0
        model.train()
        for Iter, (image1, image2, gt) in enumerate(data_loader_train):
            image1 = image1.to(device, dtype=torch.float)
            image2 = image2.to(device, dtype=torch.float)
            gt = gt.to(device, dtype=torch.float)
            pre1, pre2 = model(image1, image2)
            
            loss = loss_fuc(pre1, pre2, gt)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            Train_Loss += loss.data
        
        Train_Loss = Train_Loss/(Iter+1)
        print("Epoch:", "%04d" % (i), "train_loss =", "%.4f" % (Train_Loss))
        
        if True:
            Val_Loss = 0
            model.eval()
            with torch.no_grad():
                confusion_matrix = [0,0,0,0]
                for Iter_val, (image1_val, image2_val, gt_val) in enumerate(data_loader_val):
                    image1_val = image1_val.to(device, dtype=torch.float)
                    image2_val = image2_val.to(device, dtype=torch.float)
                    gt_val = gt_val.to(device, dtype=torch.float)
                    pre_val1, pre_val2 = model(image1_val, image2_val)
                    
                    loss = loss_fuc(pre_val1, pre_val2, gt_val)
                    Val_Loss += loss.data
                    pre_val = (pre_val2 > 0.5).float()
                    pre_val = pre_val.cpu().detach().numpy()
                    gt_val = gt_val.cpu().detach().numpy()
                    confusion_matrix = np.sum([confusion_matrix, get_confusion_matrix(pre_val, gt_val)],
                                            axis=0).tolist()
            
            accuracy, f1_score, iou, precision, recall = get_metric(confusion_matrix)
            Val_Loss = Val_Loss / (Iter_val + 1)
            poly_lr_scheduler.step()
            
            # model_path = save_path + '/epoch'+str(i)+'_model.pth'
            # torch.save(model, model_path)
            
            # if Val_Loss <= minoitor and i > 5:
            if i > 5:
                minoitor = Val_Loss
                best_model_path = save_path + '/best_epoch'+ str(i) + '_model.pth'
                torch.save(model, best_model_path)
            
            # # 保存随机状态以便完全复现
            # if i == 1:  # 只保存第一次的随机状态
            #     random_state_path = save_path + '/random_state.pth'
            #     torch.save({
            #         'python_random_state': random.getstate(),
            #         'numpy_random_state': np.random.get_state(),
            #         'torch_random_state': torch.get_rng_state(),
            #         'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            #     }, random_state_path)
            
            print("Epoch:", "%04d" % (i), "val_loss =", "%.4f" % (Val_Loss),
                  "accuracy =", "%.4f" % (accuracy), "f1_score =", "%.4f" % (f1_score),
                  "iou =", "%.4f" % (iou), "precision =", "%.4f" % (precision),"recall =", "%.4f" % (recall))
        
        with open(os.path.join(save_path, str(begin_time)+'.txt'), 'a') as file:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            file.write(f'{i}\t{lr:.8f}\t{Train_Loss:.4f}\t{Val_Loss:.4f}'
                       f'\t{accuracy:.4f}\t{f1_score:.4f}\t{iou:.4f}'
                       f'\t{precision:.4f}\t{recall:.4f}\n')
            file.close()