import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from prep import printProgressBar
from measure import compute_measure
from losses import HybridLoss, CharbonnierLoss


class TeeLogger:
    def __init__(self, log_path=None):
        self.log_path = log_path
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def write(self, message):
        print(message)
        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, secs)


def moving_average(values, window):
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode='same')


def epoch_average(values, iters_per_epoch):
    if len(values) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    epoch_means = []
    epoch_end_iters = []
    for start in range(0, len(values), iters_per_epoch):
        chunk = values[start:start + iters_per_epoch]
        if len(chunk) > 0:
            epoch_means.append(float(np.mean(chunk)))
            epoch_end_iters.append(start + len(chunk))
    return np.array(epoch_means, dtype=np.float32), np.array(epoch_end_iters, dtype=np.int32)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_arr(arr,patch_size,stride=32):    ## 512*512 to 32*32
    pad = (16, 16, 16, 16) # pad by (0, 1), (2, 1), and (3, 3)
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _,_,h,w = arr.shape
    num = h//stride - 1
    arrs = torch.zeros(num*num,1,patch_size,patch_size)

    for i in range(num):
        for j in range(num):
            arrs[i*num+j,0] = arr[0,0,i*stride:i*stride+patch_size,j*stride:j*stride+patch_size]
    return arrs

def agg_arr(arrs, size, stride=32):  ## from 32*32 to size 512*512
    arr = torch.zeros(size, size)
    n,_,h,w = arrs.shape
    num = size//stride
    for i in range(num):
        for j in range(num):
            arr[i*stride:(i+1)*stride,j*stride:(j+1)*stride] = arrs[i*num+j,:,16:48,16:48]
  #return arr
    return arr.unsqueeze(0).unsqueeze(1)

class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.max_train_iters = args.max_train_iters
        self.resume_iters = args.resume_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size
        self.logger = TeeLogger(args.log_file if args.log_file else os.path.join(self.save_path, 'train.log'))
        self.model_name = args.model_name.lower()
        self.loss_name = args.loss_name.lower()
        self.optimizer_name = args.optimizer.lower()
        self.scheduler_name = args.scheduler.lower()
        self.weight_decay = args.weight_decay
        self.naf_width = args.naf_width
        self.naf_enc_blocks = args.naf_enc_blocks
        self.naf_middle_blocks = args.naf_middle_blocks
        self.naf_dec_blocks = args.naf_dec_blocks
        self.naf_dropout = args.naf_dropout
        self.restormer_dim = args.restormer_dim
        self.restormer_num_blocks = args.restormer_num_blocks
        self.restormer_num_refinement_blocks = args.restormer_num_refinement_blocks
        self.restormer_heads = args.restormer_heads
        self.restormer_ffn_expansion_factor = args.restormer_ffn_expansion_factor
        self.grad_accum_steps = max(1, args.grad_accum_steps)
        self.use_amp = bool(args.use_amp and self.device.type == 'cuda')

        self.model = self.build_model(for_test=False)
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            self.logger.write('Use {} GPUs'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.lr = args.lr
        self.criterion = self.build_loss()
        self.optimizer = self.build_optimizer()
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

    def build_model(self, for_test=False):
        if self.model_name == 'ctnafnet':
            from advanced_ct_denoiser import CTNAFNet
            return CTNAFNet(
                img_channel=1,
                width=self.naf_width,
                enc_blocks=tuple(self.naf_enc_blocks),
                middle_blocks=self.naf_middle_blocks,
                dec_blocks=tuple(self.naf_dec_blocks),
                dropout=self.naf_dropout
            )
        raise ValueError('This project only supports model_name=ctnafnet, got {}'.format(self.model_name))

    def build_loss(self):
        if self.loss_name == 'mse':
            return nn.MSELoss()
        if self.loss_name == 'charbonnier':
            return CharbonnierLoss()
        if self.loss_name == 'hybrid':
            return HybridLoss()
        raise ValueError('Unsupported loss_name: {}'.format(self.loss_name))

    def build_optimizer(self):
        if self.optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError('Unsupported optimizer: {}'.format(self.optimizer_name))


    def save_model(self, iter_):
        prefix = 'T2T_vit' if self.model_name == 'ctformer' else self.model_name
        f = os.path.join(self.save_path, '{}_{}iter.ckpt'.format(prefix, iter_))
        torch.save(self.model.state_dict(), f)


    def load_model(self, iter_):
        device = torch.device('cpu')
        candidates = [
            os.path.join(self.save_path, '{}_{}iter.ckpt'.format(self.model_name, iter_)),
            os.path.join(self.save_path, 'T2T_vit_{}iter.ckpt'.format(iter_)),
        ]
        model_path = next((path for path in candidates if os.path.exists(path)), None)
        if not model_path:
            raise FileNotFoundError('Checkpoint not found for iter {} in {}'.format(iter_, self.save_path))
        self.model.load_state_dict(torch.load(model_path, map_location=device))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def infer_by_tiles(self, x, tile_size=64, stride=32, batch_size=16):
        _, _, height, width = x.shape
        if height != width:
            raise ValueError('Tile inference currently expects square inputs, got {}x{}'.format(height, width))

        tiles = split_arr(x.detach().cpu(), tile_size, stride=stride).to(self.device)
        preds = torch.zeros_like(tiles)

        for start in range(0, tiles.shape[0], batch_size):
            end = min(start + batch_size, tiles.shape[0])
            preds[start:end] = self.model(tiles[start:end])

        return agg_arr(preds, height, stride=stride).to(self.device)


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result, slice_id=None):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)
        if slice_id:
            f.text(0.98, 0.98, 'Slice: {}'.format(slice_id), ha='right', va='top', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        safe_name = str(slice_id) if slice_id else 'result_{}'.format(fig_name)
        f.savefig(os.path.join(self.save_path, 'fig', '{}.png'.format(safe_name)))
        plt.close()


    def train(self):
        NumOfParam = count_parameters(self.model)
        self.logger.write('trainable parameter: {}'.format(NumOfParam))
        self.logger.write('device: {}'.format(self.device))
        self.logger.write('dataset size: {}'.format(len(self.data_loader.dataset)))
        self.logger.write('model_name: {}'.format(self.model_name))
        self.logger.write('loss_name: {}'.format(self.loss_name))
        self.logger.write('optimizer: {} (lr={}, weight_decay={})'.format(self.optimizer_name, self.lr, self.weight_decay))
        self.logger.write('use_amp: {}, grad_accum_steps: {}'.format(self.use_amp, self.grad_accum_steps))

        if self.resume_iters > 0:
            self.logger.write('resume model: {}'.format(self.resume_iters))
            self.load_model(self.resume_iters)

        train_losses = []
        total_iters = self.resume_iters
        start_time = time.time()
        loss_all = []
        stop_training = False
        iters_per_epoch = len(self.data_loader)
        planned_total_iters = self.max_train_iters if self.max_train_iters > 0 else max(1, (self.num_epochs - 1) * iters_per_epoch)
        scheduler = None
        if self.scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(self.optimizer, T_max=planned_total_iters, eta_min=max(self.lr * 0.01, 1e-6))
        start_epoch = (self.resume_iters // iters_per_epoch) + 1
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, self.num_epochs):
            self.model.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)   ## expand one dimension given the dimension 0  4->[1,4]
                y = y.unsqueeze(0).float().to(self.device)   ## copy data to device

                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)  ## similar to reshape
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                with autocast('cuda', enabled=self.use_amp):
                    pred = self.model(x)
                    if self.loss_name == 'mse':
                        loss = self.criterion(pred, y) * 100 + 1e-4
                    else:
                        loss = self.criterion(pred, y)

                loss_value = float(loss.item())
                train_losses.append(loss_value)
                loss_all.append(loss_value)

                scaled_loss = loss / self.grad_accum_steps
                self.scaler.scale(scaled_loss).backward()

                if total_iters % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                # print
                if total_iters % self.print_iters == 0:
                    elapsed = time.time() - start_time
                    progress = min(total_iters / planned_total_iters, 1.0)
                    eta = (elapsed / total_iters) * (planned_total_iters - total_iters) if total_iters > 0 else 0
                    self.logger.write(
                        "STEP [{}], EPOCH [{}/{}], ITER [{}/{}], PROGRESS: {:.2f}%, LOSS: {:.8f}, ELAPSED: {}, ETA: {}".format(
                            total_iters, epoch, self.num_epochs, iter_ + 1,
                            len(self.data_loader), progress * 100.0, loss_value,
                            format_seconds(elapsed), format_seconds(eta)
                        )
                    )
                # learning rate decay
                #print(total_iters)  
                if scheduler is None and self.decay_iters > 0 and total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if self.save_iters > 0 and total_iters % self.save_iters == 0:
                    self.logger.write("save model: {}".format(total_iters))
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
                if self.max_train_iters > 0 and total_iters >= self.max_train_iters:
                    stop_training = True
                    break
            if stop_training:
                break
        if total_iters % self.grad_accum_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        self.save_model(total_iters)
        np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
        self.logger.write("total_iters: {}".format(total_iters))
        self.save_loss_plot(np.array(loss_all), total_iters)

    def save_loss_plot(self, loss_values, total_iters):
        if len(loss_values) == 0:
            return
        iters_per_epoch = max(1, len(self.data_loader))
        epoch_loss, epoch_x = epoch_average(loss_values, iters_per_epoch)
        final_loss = float(epoch_loss[-1])
        start_loss = float(epoch_loss[0])
        target_95 = start_loss - 0.95 * (start_loss - final_loss)
        reach_95_idx = int(np.where(epoch_loss <= target_95)[0][0]) if np.any(epoch_loss <= target_95) else len(epoch_loss) - 1
        reach_95_iter = int(epoch_x[reach_95_idx])
        reach_95_loss = float(epoch_loss[reach_95_idx])
        max_idx = int(np.argmax(epoch_loss))
        min_idx = int(np.argmin(epoch_loss))

        plt.figure(figsize=(12, 6))
        plt.plot(epoch_x, epoch_loss, color='darkred', linewidth=2.0, label='Epoch-average loss')
        plt.scatter([reach_95_iter], [reach_95_loss], color='royalblue', s=60, zorder=3, label='95% steady-state')
        plt.scatter([int(epoch_x[max_idx])], [float(epoch_loss[max_idx])], color='orange', s=60, zorder=3, label='Max epoch loss')
        plt.scatter([int(epoch_x[min_idx])], [float(epoch_loss[min_idx])], color='green', s=60, zorder=3, label='Min epoch loss')
        plt.axvline(reach_95_iter, color='royalblue', linestyle='--', linewidth=1.2, alpha=0.8)
        plt.title('{} Epoch-average Training Loss'.format(self.model_name.upper()))
        plt.xlabel('Training iteration')
        if self.loss_name == 'mse':
            plt.ylabel('Loss (MSE * 100 + 1e-4)')
        else:
            plt.ylabel('Loss ({})'.format(self.loss_name))
        plt.annotate('95% steady-state\niter={}\nloss={:.6f}'.format(reach_95_iter, reach_95_loss),
                     xy=(reach_95_iter, reach_95_loss),
                     xytext=(12, 12), textcoords='offset points', color='royalblue')
        plt.annotate('max\niter={}\nloss={:.6f}'.format(int(epoch_x[max_idx]), float(epoch_loss[max_idx])),
                     xy=(int(epoch_x[max_idx]), float(epoch_loss[max_idx])),
                     xytext=(12, -28), textcoords='offset points', color='darkorange')
        plt.annotate('min\niter={}\nloss={:.6f}'.format(int(epoch_x[min_idx]), float(epoch_loss[min_idx])),
                     xy=(int(epoch_x[min_idx]), float(epoch_loss[min_idx])),
                     xytext=(12, 12), textcoords='offset points', color='darkgreen')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'loss.png'))
        plt.close()

    def test(self):
        del self.model
        self.model = self.build_model(for_test=True)
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                if len(batch) == 3:
                    x, y, slice_id = batch
                    if isinstance(slice_id, (list, tuple)):
                        slice_id = slice_id[0]
                else:
                    x, y = batch
                    slice_id = str(i)
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                
                if self.model_name == 'ctformer':
                    arrs = split_arr(x, 64).to(self.device)
                    arrs[0:64] = self.model(arrs[0:64])
                    arrs[64:2*64] = self.model(arrs[64:2*64])
                    arrs[2*64:3*64] = self.model(arrs[2*64:3*64])
                    arrs[3*64:4*64] = self.model(arrs[3*64:4*64])
                    pred = agg_arr(arrs, 512).to(self.device)
                elif self.model_name == 'ctrestormer':
                    pred = self.infer_by_tiles(
                        x,
                        tile_size=self.patch_size,
                        stride=max(16, self.patch_size // 2),
                        batch_size=16
                    )
                else:
                    pred = self.model(x)
                

                #pred = x - pred# denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result, slice_id=slice_id)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            self.logger.write('')
            self.logger.write('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                             ori_ssim_avg/len(self.data_loader), 
                                                                                             ori_rmse_avg/len(self.data_loader)))
            self.logger.write('')
            self.logger.write('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                   pred_ssim_avg/len(self.data_loader), 
                                                                                                   pred_rmse_avg/len(self.data_loader)))
