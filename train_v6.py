#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49
import torch
import torch.distributed
# GPUs
gpus = ','.join([str(i) for i in [0,1]])
# torch.distributed.init_process_group(backend="nccl")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
from PIL import Image
from torchvision.transforms import transforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as uData
import matplotlib.pyplot as plt
from JigsawNet import JigsawNet
from networks3.Discriminator import NLayerDiscriminator
from networks3 import _NetG_DOWN, DnCNN, UNetD,sample_generator,\
    Discriminator1,VGGStyleDiscriminator128,_NetD,NLayerDiscriminator,APBSN,DBSNl,Deam
from datasets.DenoisingDatasets import BenchmarkTrain, BenchmarkTest
from math import ceil
from torch.autograd import Variable
from torchvision.transforms.functional import rotate,vflip
from utils import *
import itertools
import random
from loss import mean_match, get_gausskernel, gradient_penalty, \
    mean_match_1, GANLoss, GANLoss_v2, PerceptualLoss, GANLoss_v3, L1Loss, log_SSIM_loss, mse_loss
import torchvision.utils as utils1
# from torch.utils.tensorboard import SummaryWriter
import shutil
import warnings
from pathlib import Path
import commentjson as json
from GaussianSmoothLayer import GaussionSmoothLayer
import torchvision.utils as utils

# filter warnings

device = torch.device('cuda:0'if torch.cuda.is_available()else'cpu')
warnings.simplefilter('ignore', Warning, lineno=0)

# default dtype
torch.set_default_dtype(torch.float32)

_C = 3
_modes = ['train', 'val']
BGBlur_kernel = [3, 9, 15]

### change1: GaussionSmoothLayer(3, k_size, 25).cuda()
# BlurNet = [GaussionSmoothLayer(3, k_size, 25) for k_size in BGBlur_kernel]
BlurWeight = [0.01,0.1,1.]

BlurNet = [GaussionSmoothLayer(3, k_size, 25).cuda() for k_size in BGBlur_kernel]
# BlurWeight = [1.]
from restormer import Restormer, ResLocal
img_path = '/media/sr617/新加卷/linshi/code_train/dataset/noisy'
targeet_path = '/media/sr617/新加卷/linshi/code_train/dataset/gt_new'

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

    print('start')
    device_ids = [i for i in range(torch.cuda.device_count())]
    netE = Restormer()
    netE= nn.DataParallel(netE, device_ids=device_ids)
    netE = netE.cuda()

    netD = Restormer()
    netD = nn.DataParallel(netD, device_ids=device_ids)
    netD = netD.cuda()
    print('for')
    # build up the generator
    netG = _NetG_DOWN(stride=1)
    netG = nn.DataParallel(netG, device_ids=device_ids)
    netG = netG.cuda()

    # build up the discriminator

    netP = NLayerDiscriminator(6)
    netP = nn.DataParallel(netP, device_ids=device_ids)
    netP = netP.cuda()

    criterionGAN = GANLoss(args['gan_mode']).cuda()
    init_weights(netG, init_type='normal',init_gain=0.02)
    init_weights(netP, init_type='normal', init_gain=0.02)

    net = {'E':netE,'D': netD, 'G': netG, 'P': netP}

    # optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=args['lr_G'])
    optimizerD = optim.Adam(netD.parameters(), lr=args['lr_D'])
    optimizerP = optim.Adam(netP.parameters(), lr=args['lr_P'])
    optimizer = {'D': optimizerD, 'G': optimizerG, 'P': optimizerP}
    if args['resume']:
        if Path(args['resume']).is_file():
            print('=> Loading checkpoint {:s}'.format(str(Path(args['resume']))))
            checkpoint = torch.load(str(Path(args['resume'])), map_location='cpu')
            args['epoch_start'] = checkpoint['epoch']
            netE.load_state_dict(checkpoint['model_state_dict']['D'])
            netD.load_state_dict(checkpoint['model_state_dict']['D'])
            netG.load_state_dict(checkpoint['model_state_dict']['G'])
            netP.load_state_dict(checkpoint['model_state_dict']['P'])

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

    # build the Gaussian kernel for loss
    global kernel
    kernel = get_gausskernel(args['ksize'], chn=_C).cuda()
    # train model
    print('\nBegin training with GPU: ' + (args['gpu_id']))
    train_epoch(net, datasets, optimizer, args, criterionGAN)

def train_epoch(net, datasets, optimizer, args, criterionGAN):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss().cuda()
    loss_ssim = log_SSIM_loss().cuda()
    blurkernel = GaussionSmoothLayer(5,15,9).cuda()
    batch_size = {'train': args['batch_size'], 'val': 4}
    data_loader = {phase: uData.DataLoader(datasets[phase], batch_size=batch_size[phase],
                                           shuffle=True, num_workers=0, pin_memory=True) for phase in
                   _modes}
    data_set_gt = BenchmarkTrain(h5_file=args['SIDD_train_h5_gt'],
                                 length=2000 * args['batch_size'] * args['num_critic'],
                                 pch_size=args['patch_size'],
                                 mask=False)
    # todo gt dataset has no key()
    data_loader_gt = uData.DataLoader(data_set_gt, batch_size=batch_size['train'],
                                      shuffle=True, num_workers=0, pin_memory=True)

    num_data = {phase: len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}

    # L1_criterion = L1Loss().to(device)
    # content_criterion = nn.L1Loss().to(device)
    # perception_criterion = PerceptualLoss().to(device)

    for epoch in range(args['epoch_start'], args['epochs']):
        loss_epoch = {x: 0 for x in ['PL', 'DL', 'GL']}
        subloss_epoch = {x: 0 for x in
                         ['loss_GAN_DG', 'loss_l1', 'perceptual_loss', 'loss_bgm', 'loss_GAN_P_real', 'loss_GAN_P_fake']}
        mae_epoch = {'train': 0, 'val': 0}

        # optE, optD, optP1, optP2, optP3 ,optG = optimizer['E'] ,optimizer['D'], optimizer['P1'], optimizer['P2'],optimizer['P3'], optimizer['G']
        optD, optP, optG =  optimizer['D'],optimizer['P'], optimizer['G']


        tic = time.time()
        # train stage
        net['D'].train()
        net['G'].train()
        net['P'].train()


        lr_D = optimizer['D'].param_groups[0]['lr']
        lr_G = optimizer['G'].param_groups[0]['lr']
        lr_P = optimizer['P'].param_groups[0]['lr']

        if lr_D < 1e-6:
            sys.exit('Reach the minimal learning rate')
        phase = 'train'
        iter_GD = 0

        for ii, data in enumerate(data_loader[phase]):
        # for ii,data in enumerate(zip(data_loader[phase])):

            im_noisy,_ = [x.cuda() for x in data]
            _,im_gt = [x.cuda() for x in data1]
            ################################
            #training generator
            #############################
            # optimizer['E'].zero_grad()
            optimizer['G'].zero_grad()
            optimizer['D'].zero_grad()
            rec_x2 = net['D'](im_noisy.detach())
            # Reb-SC
            aa = random.randint(0, 7)
            bb = random.randint(0, 7)
            xx = random.randint(0, 7)
            yy = random.randint(0, 7)
            with torch.set_grad_enabled(False):
                im_noisy_1 = data_augmentation(im_noisy, xx)
                im_noisy_2 = data_augmentation(im_noisy, yy)
                im_noisy_3 = data_augmentation(im_noisy, aa)
                im_noisy_4 = data_augmentation(im_noisy, bb)

                tizao_1_1 = net['E'](im_noisy_1)
                tizao_1_2 = net['E'](im_noisy_2)
                tizao_1_3 = net['E'](im_noisy_3)
                tizao_1_4 = net['E'](im_noisy_4)

                tizao_1_1 = redata_augmentation(tizao_1_1,xx)
                tizao_1_2 = redata_augmentation(tizao_1_2, yy)
                tizao_1_3 = redata_augmentation(tizao_1_3, aa)
                tizao_1_4 = redata_augmentation(tizao_1_4, bb)

                tizao_1 = (tizao_1_1 + tizao_1_2 + tizao_1_3 + tizao_1_4) / 4

            fake_im_noisy1 = net['G'](im_gt, (im_noisy-tizao_1))

            rec_x1 = net['D'](fake_im_noisy1.detach())



            with torch.set_grad_enabled(False):
                # tizao_2 = net['E'](fake_im_noisy1)
                fake_im_noisy1_1 = data_augmentation(fake_im_noisy1, xx)
                fake_im_noisy1_2 = data_augmentation(fake_im_noisy1, yy)
                fake_im_noisy1_3 = data_augmentation(fake_im_noisy1, aa)
                fake_im_noisy1_4 = data_augmentation(fake_im_noisy1, bb)

                tizao_2_1 = net['E'](fake_im_noisy1_1)
                tizao_2_2 = net['E'](fake_im_noisy1_2)
                tizao_2_3 = net['E'](fake_im_noisy1_3)
                tizao_2_4 = net['E'](fake_im_noisy1_4)

                tizao_2_1 = redata_augmentation(tizao_2_1, xx)
                tizao_2_2 = redata_augmentation(tizao_2_2, yy)
                tizao_2_3 = redata_augmentation(tizao_2_3, aa)
                tizao_2_4 = redata_augmentation(tizao_2_4, bb)

                tizao_2 = (tizao_2_1 + tizao_2_2 + tizao_2_3 + tizao_2_4) / 4
                    # lim_noisy2 = rotate(fake_im_noisy1, 90)
            # lim_denoise3 = net['E'](lim_noisy2)
            # denoisebian1 = net['E'](bian1)
            # denoisebian2 = net['E'](bian2)
            # unbian1 =  redata_augmentation(denoisebian1,xx)
            # unbian2 = redata_augmentation(denoisebian2, yy)
            # lim_denoise3 = rotate(lim_denoise3, 270)

            # tizao_2 = (lim_denoise3 + tizao_2) / 2


            # fake_im_noisy2 = net['G'](rec_x2, im_noisy)
            fake_im_noisy2 = net['G'](rec_x2, (im_noisy-tizao_1))
            fake_im_noisy3 = net['G'](rec_x2, (fake_im_noisy1-tizao_2))

            # fake_im_noisy3 = net['G'](rec_x2, (tizao_2 - fake_im_noisy1))
            # fake_im_noisy3 = net['G'](rec_x2, fake_im_noisy1)
            # fake_im_noisy4 = net['G'](rec_x1, fake_im_noisy1)
            fake_im_noisy4 = net['G'](rec_x1, (fake_im_noisy1-tizao_2))
            # fake_im_noisy4 = net['G'](rec_x1, (fake_im_noisy1 - tizao_2))

            # 不更新梯度
            set_requires_grad([net['P']], False)
            # set_requires_grad([net['E']], False)

            subloss_epoch['perceptual_loss'] += 0
            # he = torch.cat([fake_im_noisy1-rec_x2, fake_im_noisy1], dim=1)
            adversarial_loss1 = criterionGAN(net['P'](fake_im_noisy1), True)

            adversarial_loss = adversarial_loss1
            # identity_loss = criterion(net['G'](im_noisy),im_noisy)
            # 保留输入和输出颜色组成的一致性（cyclegan中应用）
            identity_loss = 0

            bgm_loss1 = 0
            bgm_loss2 = 0
            bgm_loss  = 0
# 求BCM
            for index, weight in enumerate(BlurWeight):
                out_b1 = BlurNet[index](im_gt)
                out_real_b1 = BlurNet[index](fake_im_noisy1)
                # out_b2 = BlurNet[index](rec_x2)
                # out_real_b2 = BlurNet[index](fake_im_noisy2)
                grad_loss_b1 = criterion(out_b1, out_real_b1)
                # grad_loss_b2 = criterion(out_b2, out_real_b2)
                bgm_loss1 += weight * (grad_loss_b1)
                # bgm_loss2 += weight * (grad_loss_b2)
                bgm_loss  += bgm_loss1 + bgm_loss2
            # loss_G = adversarial_loss1 * args['adversarial_loss_factor'] + \
            loss_G =  adversarial_loss * args['adversarial_loss_factor'] + \
                     bgm_loss1 * args['bgm_loss'] + \
                     bgm_loss2 * args['bgm_loss']

            los_ssim1 = loss_ssim(rec_x2, tizao_1)
            loss_recon1 = criterion(rec_x2, tizao_1)
            los_ssim2 = loss_ssim(rec_x1, tizao_2)
            loss_recon2 = criterion(rec_x1, tizao_2)
            los_ssim = loss_ssim(rec_x1, im_gt)
            loss_recon = criterion(rec_x1, im_gt)
            loss_D = loss_recon + loss_recon1 + loss_recon2 + los_ssim + los_ssim1 + los_ssim2
            loss_G.backward()
            loss_D.backward()
            optimizer['G'].step()
            optimizer['D'].step()
            loss_epoch['DL'] += loss_D.item()
            loss_epoch['GL'] += loss_G.item()
            subloss_epoch['loss_GAN_DG'] += adversarial_loss.item()
            subloss_epoch['loss_bgm'] += bgm_loss.item()




            ##########################
            # training discriminator #
            ##########################
            if (ii+1) % args['num_critic'] == 0:
                # 对tensor和tensor计算出来的其他tensor求导且保存在grad中，便于优化器更新参数
                set_requires_grad([net['P']], True)

                pred_real1 = net['P'](im_noisy)
                loss_P_real1 = criterionGAN(pred_real1, True)
                loss_P_real = loss_P_real1

                pred_fake1 = net['P'](fake_im_noisy1.detach())
                pred_fake2 = net['P'](fake_im_noisy3.detach())
                pred_fake3 = net['P'](fake_im_noisy4.detach())
                pred_fake4 = net['P'](fake_im_noisy2.detach())

                loss_P_fake1 = criterionGAN(pred_fake1, False)
                loss_P_fake2 = criterionGAN(pred_fake2, False)
                loss_P_fake3 = criterionGAN(pred_fake3, False)
                loss_P_fake4 = criterionGAN(pred_fake4, False)
                loss_P_fake = loss_P_fake1 + loss_P_fake3 + loss_P_fake4 + loss_P_fake2
                loss_P = (loss_P_real + loss_P_fake) * 0.5
                loss_P.backward()

                optimizer['P'].step()
                optimizer['P'].zero_grad()



                loss_epoch['PL'] += loss_P.item()
                subloss_epoch['loss_GAN_P_real'] += loss_P_real.item()
                subloss_epoch['loss_GAN_P_real'] += loss_P_fake.item()

                if (ii + 1) % args['print_freq'] == 0:
                    template = '[Epoch:{:>2d}/{:<3d}] {:s}:{:0>5d}/{:0>5d},' + \
                                   ' PL:{:>6.6f}, GL:{:>6.6f}, DL:{:>6.6f}, ' \
                                   'loss_GAN_G:{:>6.6f},' + \
                                   'loss_bgm:{:>6.9f}, loss_P_real:{:>6.4f}, ' \
                                   'loss_P_fake:{:>6.4f}, indentity_loss:{:>6.4f}'
                    print(template.format(epoch + 1, args['epochs'], phase, ii + 1, num_iter_epoch[phase],
                                              loss_P.item(), loss_G.item(),loss_D.item(),
                                              # loss_P1.item(), loss_G.item(), loss_D.item(),
                                              adversarial_loss.item(), bgm_loss1.item(), loss_P_real.item(), loss_P_fake.item(),identity_loss))

        loss_epoch['GL'] /= (ii + 1)
        # loss_epoch['EL'] /= (ii + 1)

        subloss_epoch['loss_GAN_DG'] /= (ii + 1)
        subloss_epoch['loss_bgm'] /= (ii + 1)
        subloss_epoch
        loss_epoch['PL'] /= (ii + 1)
        # loss_epoch['P2L'] /= (ii + 1)
        subloss_epoch['loss_GAN_P_real'] /= (ii + 1)
        subloss_epoch['loss_GAN_P_fake'] /= (ii + 1)

        template = '{:s}: PL:{:>6.6f}, GL:{:>6.6f},loss_GAN_DG:{:>6.6f}, ' + \
                   ' loss_bgm:{:>6.4f}, loss_P_real:{:>6.4f}, ' \
                   'loss_P_fake:{:>6.4f}, lrDG/P:{:.2e}/{:.2e}'
        print(template.format(phase, loss_epoch['PL'], loss_epoch['GL'], subloss_epoch['loss_GAN_DG'],
                              subloss_epoch['loss_bgm'],
                              subloss_epoch['loss_GAN_P_real'],
                              subloss_epoch['loss_GAN_P_fake'], lr_D, lr_P))

        net['G'].eval()
        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}'
            .format(
            epoch,
            lr=lr_D,
            loss=loss_epoch['DL']))

        print('-' * 150)

        # test stage
        net['G'].eval()
        net['D'].eval()
        psnr_per_epoch = ssim_per_epoch = 0
        phase = 'val'
        PSNR = 0
        SSIM = 0

        transform = transforms.ToTensor()

        for img in img_list:
            image = Image.open(img_path + '/' + img).convert('RGB')
            # print(image)
            target = Image.open(targeet_path + '/' + img).convert('RGB')
            image = transform(image)
            target = transform(target)
            image = image.cuda()
            target = target.cuda()

            save_path = './generate_out'
            save_path_1 = save_path + '/' + img

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
        print('-' * 150)

        # save model
        model_prefix = 'model_'
        save_path_model = str(Path(args['model_dir']) / (model_prefix + str(epoch + 1)))
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': {x: net[x].state_dict() for x in ['D',  'G', 'P']},
            'optimizer_state_dict': {x: optimizer[x].state_dict() for x in ['D', 'P', 'G']},
            # 'lr_scheduler_state_dict': {x: lr_scheduler[x].state_dict() for x in ['D', 'P', 'G']}
        }, save_path_model)
        model_prefix = 'model_state_'
        save_path_model = str(Path(args['model_dir']) / (model_prefix + str(epoch + 1) + 'PSNR{:.2f}_SSIM{:.4f}'.
                                                         format(psnr_per_epoch, ssim_per_epoch) + '.pt'))
        torch.save({x: net[x].state_dict() for x in ['D', 'G', 'P']}, save_path_model)

        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))

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
