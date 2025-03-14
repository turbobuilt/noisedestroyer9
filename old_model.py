import torch
import torch.nn as nn
import torch.nn.functional as F


from torch import nn
import torch
import torchaudio
import os
import glob
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from load_data_pedalboard_random import DataIterator
from stft import istft, stft
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


target_lr = .0003
d_model = 256
layers_count = 13
segment_length=2**15
sample_rate = 2**15
model_name = f"ManyConvNet_4_resnet_billion_{target_lr}_{sample_rate}_{d_model}_{layers_count}"
device = torch.device("cuda")
out_dir = f"{model_name}_out"
checkpoint_dir = f"{model_name}_checkpoints"
# target_lr = .0001

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        
        # First part of the block
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Second part of the block
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # If in_channels != out_channels, use a 1x1 convolution to match dimensions
        self.match_dimensions = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        identity = x
        if self.match_dimensions:
            identity = self.match_dimensions(identity)
        out = self.conv2(out)
        out = out + identity
        return out

class ManyConvNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_layers=8, base_filters=64, growth_factor=1.24):
        super(ManyConvNet, self).__init__()
        
        self.num_layers = num_layers
        self.growth_factor = growth_factor
        
        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else int(base_filters * (growth_factor ** (i - 1)))
            out_ch = int(base_filters * (growth_factor ** i))
            print("i", i, "in_ch", in_ch, "out_ch", out_ch)
            self.encoders.append(self.conv_block(in_ch, out_ch))
        
        self.decoders = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_ch = int(base_filters * (growth_factor ** (i + 1))) if i != num_layers - 1 else int(base_filters * (growth_factor ** i))
            out_ch = int(base_filters * (growth_factor ** i))
            print("decode i", i, "in_ch", in_ch, "out_ch", out_ch)
            self.decoders.append(self.out_conv_block(in_ch + out_ch, out_ch))  # Adjusted input channels for concatenation
        
        # Final Convolution
        self.final_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    

    def out_conv_block(self, in_channels, out_channels):
        return ResNetBlock(in_channels, out_channels)
        # return nn.Sequential(
        #     nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
    
    def forward(self, x):
        # Encoder
        enc_features = []
        for encoder in self.encoders:
            # print("x shape", x.shape)
            x = encoder(x)
            enc_features.append(x)
            x = F.max_pool1d(x, kernel_size=2)
        
        # Decoder
        for i, decoder in enumerate(self.decoders):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = torch.cat([x, enc_features[-(i + 1)]], dim=1)
            # print("x.shape", x.shape)
            x = decoder(x)
        
        # Final Convolution
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    writer = SummaryWriter(log_dir=f"runs/{model_name}", flush_secs=10)
    writer2 = SummaryWriter(log_dir=f"runs/{model_name}_r2", flush_secs=10)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    for f in glob.glob(f'{out_dir}/*'):
        os.remove(f)
    model = ManyConvNet(num_layers=layers_count, base_filters=d_model).to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("num trainable params is", params)

    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=target_lr)
    criterion = torch.nn.L1Loss()
    step=0

    total_steps=-1
    start_epoch = 0

    model = torch.compile(model, mode="max-autotune")
    checkpoints_sorted = glob.glob(f'{model_name}_checkpoints/*.pt')
    if len(checkpoints_sorted) > 0:
        checkpoints_sorted.sort(key=os.path.getmtime)
        print("loading checkpoint", checkpoints_sorted[-1],)
        checkpoint = torch.load(checkpoints_sorted[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps = checkpoint['total_steps']
        step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        print("loaded checkpoint", checkpoints_sorted[-1], "total steps", total_steps, "start_i", step, "epoch", start_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = target_lr
    warmup_steps = 10000

    test_dataset = DataIterator(sample_rate=sample_rate, segment_length=segment_length*2, max_examples_per_file=1, queue_depth=2, num_workers=2)

    for epoch in range(start_epoch, 5000):
        dataset = DataIterator(sample_rate=sample_rate, segment_length=segment_length, max_examples_per_file=100, queue_depth=100, num_workers=12)
        loop_time = timer()
        x_parts = []
        target_parts = []
        batch_size=1
        for i, (x_single, target_single) in enumerate(dataset, step):
            #batch
            x_parts.append(torch.from_numpy(x_single.copy()).bfloat16())
            target_parts.append(torch.from_numpy(target_single.copy()))
            if len(x_parts) < batch_size:
                continue
            x = torch.stack(x_parts, dim=0)
            target=torch.stack(target_parts, dim=0)
            x_parts = []
            target_parts = []
            start = timer()
            x = x.reshape(x.shape[0],1,x.shape[1]).to(device)
            target = target.reshape(target.shape[0],1,target.shape[1]).to(device)

            end = timer()
            step += 1
            total_steps += 1
            if total_steps < warmup_steps:
                lr = (target_lr / warmup_steps) * (total_steps+1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            out = None
            for repetition in range(1):
                optimizer.zero_grad()
                with autocast(dtype=torch.bfloat16):
                    if out is not None:
                        x = out.detach()
                        x.requires_grad_(True)
                    out = model(x)
                    loss = criterion(out, target)
                    loss.backward()
                loss_item = loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                
                if total_steps % 1000 == 0:
                    torchaudio.save(f"{out_dir}/e_{str(epoch)}_i_{str(total_steps)}_{repetition}_out.wav", out.detach().reshape(1,-1).cpu().float(), sample_rate)
                    torchaudio.save(f"{out_dir}/e_{str(epoch)}_i_{str(total_steps)}_{repetition}_in.wav", x.detach().reshape(1,-1).cpu().float(), sample_rate)
                    print("in shape", x.shape)
                    print("target shape", target.shape)
                    torchaudio.save(f"{out_dir}/e_{str(epoch)}_i_{str(total_steps)}_{repetition}_target.wav", target.detach().reshape(1,-1).cpu().float(), sample_rate)

                if total_steps % 50 == 0:
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']
                        break
                    print(out[0,0,0:4], out.max().item(), out.min().item())
                    # print(out[0,0,253:257])
                    print(f"epoch {epoch}, step: {step}, total_steps: {total_steps} loss: {str(loss.item())[:8]}, lr: {str(lr)} min {str(torch.min(out).item())}, max {str(torch.max(out).item())}, avg {str(torch.mean(out.abs()).item())}, t_min {str(torch.min(target).item())[:8]}, t_max {str(torch.max(target).item())[:8]}, t_avg {str(torch.mean(target.abs()).item())[:8]}")
                    if repetition == 0:
                        writer.add_scalar("main_loss", loss_item, total_steps)
                        writer.add_scalar("lr", lr, total_steps)
                        writer.add_scalar("min", torch.min(target).item(), total_steps)
                        writer.add_scalar("max", torch.max(target).item(), total_steps)
                    else:
                        writer2.add_scalar("main_loss", loss_item, total_steps)
                        writer2.add_scalar("lr", lr, total_steps)
                        writer2.add_scalar("min", torch.min(target).item(), total_steps)
                        writer2.add_scalar("max", torch.max(target).item(), total_steps)

                optimizer.step()
                


            if step % 5000 == 0 and step != 0:
                old_checkpoints = glob.glob(f"{model_name}_checkpoints/*")
                old_checkpoints.sort(key=os.path.getmtime)
                for f in old_checkpoints[:-1]:
                    os.remove(f)
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_steps': total_steps,
                    'loss': loss
                }, f"{model_name}_checkpoints/{str(epoch)}_{str(i)}.pt")

        step = 0