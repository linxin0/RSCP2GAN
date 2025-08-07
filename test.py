import torch
import torch.distributed
gpus = ','.join([str(i) for i in [0, 1]])
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
from PIL import Image
from torchvision.transforms import transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as uData
from networks3 import _NetG_DOWN, DnCNN, UNetD,sample_generator,\
    Discriminator1,VGGStyleDiscriminator128,_NetD,NLayerDiscriminator,APBSN,DBSNl,Deam
from datasets.DenoisingDatasets import BenchmarkTrain, BenchmarkTest
from math import ceil
from torch.autograd import Variable
from torchvision.transforms.functional import rotate,vflip
from utils import *

from loss import mean_match, get_gausskernel, gradient_penalty, \
    mean_match_1, GANLoss, GANLoss_v2, PerceptualLoss, GANLoss_v3, L1Loss, log_SSIM_loss, mse_loss

import warnings
from pathlib import Path
import commentjson as json
from GaussianSmoothLayer import GaussionSmoothLayer

# filter warnings
device = torch.device('cuda:0'if torch.cuda.is_available()else'cpu')
warnings.simplefilter('ignore', Warning, lineno=0)

# default dtype
torch.set_default_dtype(torch.float32)

_C = 3
_modes = ['train', 'val']


img_path = '/media/sr617/新加卷/linshi/code_train/polyu_fast_test/polyu_noisy'
targeet_path = '/media/sr617/新加卷/linshi/code_train/polyu_fast_test/polyu_gt'

img_list = sorted(os.listdir(img_path))
num_img = len(img_list)

def psnr(pred, gt):
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)

def data_augmentation(image, mode):
    '''
    Performs dat augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image

       0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = vflip(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = rotate(image,90)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = rotate(image,90)
        out = vflip(out)
    elif mode == 4:
        # rotate 180 degree
        out = rotate(image,180)
    elif mode == 5:
        # rotate 180 degree and flip
        out = rotate(image,180)
        out = vflip(out)
    elif mode == 6:
        # rotate 270 degree
        out = rotate(image,270)
    elif mode == 7:
        # rotate 270 degree and flip
        out = rotate(image,270)
        out = vflip(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out
def redata_augmentation(image, mode):
    '''
    Performs dat augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = vflip(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = rotate(image,270)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = vflip(image)
        out = rotate(out,270)
        # out = vflip(out)
    elif mode == 4:
        # rotate 180 degree
        out = rotate(image,180)
    elif mode == 5:
        # rotate 180 degree and flip
        out= rotate(image,180)
        out = vflip(out)
    elif mode == 6:
        # rotate 270 degree
        out = rotate(image,90)
    elif mode == 7:
        # rotate 270 degree and flip
        out = vflip(image)
        out = rotate(out,90)

    else:
        raise Exception('Invalid choice of image transformation')

    return out
from restormer import Restormer
def main():
    # set parameters
    with open('./configs/DANet_v5.json', 'r') as f:
        args = json.load(f)

    # set the available GPUs
    # set the available GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args["gpu_id"]
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # build up the E
    # netE = torch.nn.DataParallel(DBSNl()).cuda()
    device_ids = [i for i in range(torch.cuda.device_count())]
    netE = Restormer()
    netE= nn.DataParallel(netE, device_ids=device_ids)
    netE = netE.cuda()
    # netD = torch.nn.DataParallel(UNetD(3)).cuda()
    netD = Restormer()
    netD = nn.DataParallel(netD, device_ids=device_ids)
    netD = netD.cuda()
    print('for')
    # build up the generator
    # netG = torch.nn.DataParallel(_NetG_DOWN(stride=1)).cuda()
    netG = _NetG_DOWN(stride=1)
    netG = nn.DataParallel(netG, device_ids=device_ids)
    netG = netG.cuda()


    criterionGAN = GANLoss(args['gan_mode']).cuda()
    init_weights(netG, init_type='normal',init_gain=0.02)

    net = {'D': netD}
    # optimizer
    # optimizerE = optim.Adam(netE.parameters(), lr=args['lr_E'])
    optimizerD = optim.Adam(netD.parameters(), lr=args['lr_D'])
    # optimizerD1 = optim.Adam(netD.parameters(), lr=args['lr_D1'])
    optimizer = {'D': optimizerD}
    if args['resume']:
        if Path(args['resume']).is_file():
            print('=> Loading checkpoint {:s}'.format(str(Path(args['resume']))))
            checkpoint = torch.load(str(Path(args['resume'])), map_location='cpu')
            args['epoch_start'] = checkpoint['epoch']
            netD.load_state_dict(checkpoint['model_state_dict']['D'])

            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args['resume'], checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args['epoch_start'] = 0
        if not Path(args['log_dir']).is_dir():
            # shutil.rmtree(args['log_dir'])
            Path(args['log_dir']).mkdir()
        if not Path(args['model_dir']).is_dir():
            # shutil.rmtree(args['model_dir'])
            Path(args['model_dir']).mkdir()

    for key, value in args.items():
        print('{:<15s}: {:s}'.format(key, str(value)))

    # making dataset
    datasets = {'train': BenchmarkTrain(h5_file=args['SIDD_train_h5_noisy'],
                                        length=2000 * args['batch_size'] * args['num_critic'],
                                        pch_size=args['patch_size'],
                                        mask=False),
                'val': BenchmarkTest(args['SIDD_test_h5'])}

    global kernel
    print('\nBegin training with GPU: ' + (args['gpu_id']))
    train_epoch(net, datasets, optimizer, args, criterionGAN)

def train_epoch(net, datasets, optimizer, args, criterionGAN):

    print('-' * 150)
    PSNR = 0
    SSIM = 0

    transform = transforms.ToTensor()

    for img in img_list:
        image = Image.open(img_path + '/' + img).convert('RGB')
        target = Image.open(targeet_path + '/' + img).convert('RGB')
        image = transform(image)
        target = transform(target)
        image = image.cuda()
        target = target.cuda()
        [A, B, C] = image.shape
        image = image.reshape([1, A, B, C])
        [A, B, C] = target.shape
        target = target.reshape([1, A, B, C])


        with torch.no_grad():
            B, C, H, W = image.size()
            _, _, h_old, w_old = image.size()
            h_pad = (h_old // 8 + 1) * 8 - h_old
            w_pad = (w_old // 8 + 1) * 8 - w_old
            img_lq = torch.cat([image, torch.flip(image, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            im_denoise = net['D'](img_lq)
            pre = im_denoise[:, :, :H, :W]

        processed_image = pre
        reconstruct_tar = target

        processed_image.clamp_(0.0, 1.0)
        reconstruct_tar.clamp_(0.0, 1.0)

        psnr_out = psnr(processed_image, reconstruct_tar)
        ssim_iter = batch_SSIM(processed_image, reconstruct_tar)
        SSIM += ssim_iter
        PSNR += psnr_out
    print("PSNR =", PSNR / num_img)
    print("SSIM =", SSIM / num_img)

print('Reach the maximal epochs! Finish training')

# 不更新梯度，判别器中使用
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or no
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def adjust_learning_rate(epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt['lr'] * (opt['gamma'] ** ((epoch) // opt['lr_decay']))
    # lr = opt['lr']
    return lr


if __name__ == '__main__':
    main()