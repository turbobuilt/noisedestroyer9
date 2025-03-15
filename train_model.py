import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import glob
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
import argparse
import math

from model import create_audio_denoiser, AudioDenoiserTransformer
from load_data_pedalboard_random import DataIterator

# Set up Torch performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Audio Denoiser model")
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large', 'xl'], 
                        help="Size of the model to train")
    parser.add_argument('--input_dim', type=int, default=128, 
                        help="Input feature dimension")
    parser.add_argument('--segment_length', type=int, default=32000, 
                        help="Length of audio segments for training")
    parser.add_argument('--sample_rate', type=int, default=16000, 
                        help="Audio sample rate")
    parser.add_argument('--batch_size', type=int, default=1, 
                        help="Training batch size")
    parser.add_argument('--lr', type=float, default=0.0003, 
                        help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=10000, 
                        help="Number of warmup steps for LR scheduler")
    parser.add_argument('--device', type=str, default='cuda', 
                        help="Device to run training on")
    return parser.parse_args()

def main():
    args = parse_args()
    # for device check if cuda avaialbe
    if not torch.cuda.is_available():
        args.device = 'cpu'
    
    # Set up training configuration
    model_name = f"AudioDenoiser_{args.model_size}_{args.lr}_{args.sample_rate}"
    device = torch.device(args.device)
    out_dir = f"{model_name}_out"
    checkpoint_dir = f"{model_name}_checkpoints"
    
    # Set up TensorBoard logging
    writer = SummaryWriter(log_dir=f"runs/{model_name}", flush_secs=10)
    
    # Create output directories
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Clean previous output files
    for f in glob.glob(f'{out_dir}/*'):
        os.remove(f)
    
    # Create the model with the specified sample rate
    model = create_audio_denoiser(
        model_size=args.model_size,
        input_dim=args.input_dim,
        sample_rate=args.sample_rate  # Pass sample_rate to model creation
    ).to(device)
    
    # if cuda, compile
    if args.device == 'cuda':
        model = torch.compile(model, mode="max-autotune")
    # Print model param count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {model_name} has {num_params / 1e6:.2f}M parameters")
    print(f"Model {model_name} is running on {args.device}")

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()
    
    # Initialize training state
    step = 0
    total_steps = -1
    start_epoch = 0
    
    # # Compile model for performance if available (PyTorch 2.0+)
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model, mode="max-autotune")
    
    # Load checkpoint if available
    checkpoints_sorted = glob.glob(f'{checkpoint_dir}/*.pt')
    if len(checkpoints_sorted) > 0:
        checkpoints_sorted.sort(key=os.path.getmtime)
        print(f"Loading checkpoint {checkpoints_sorted[-1]}")
        checkpoint = torch.load(checkpoints_sorted[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps = checkpoint['total_steps']
        step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint {checkpoints_sorted[-1]}, total steps: {total_steps}, step: {step}, epoch: {start_epoch}")
    
    # Setup learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    
    # Create test dataset for evaluation
    test_dataset = DataIterator(
        sample_rate=args.sample_rate, 
        segment_length=args.segment_length*2, 
        max_examples_per_file=1, 
        queue_depth=2, 
        num_workers=2,
        batch_size=args.batch_size  # Pass batch_size parameter
    )
    
    # Training loop
    for epoch in range(start_epoch, 5000):
        # Create training dataset with batch_size
        dataset = DataIterator(
            sample_rate=args.sample_rate, 
            segment_length=args.segment_length, 
            max_examples_per_file=100, 
            queue_depth=100, 
            num_workers=7,
            batch_size=args.batch_size  # Pass batch_size parameter
        )
        
        loop_time = timer()
        
        for i, (x, target) in enumerate(dataset, step):
            # Process batch - data is already batched from DataIterator
            start = timer()

            # print(f"Processing batch {i} of epoch {epoch}, batch shape: {x.shape}")
            
            # Convert numpy arrays to torch tensors and move to device
            x = torch.from_numpy(x).float().to(device)
            target = torch.from_numpy(target).float().to(device)
            
            end = timer()
            step += 1
            total_steps += 1
            
            # Learning rate warmup
            if total_steps < args.warmup_steps:
                lr = (args.lr / args.warmup_steps) * (total_steps + 1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Training step
            optimizer.zero_grad()
            # if cuda is available then use autocast
            if args.device == 'cuda':
                with autocast():
                    # Feed through model with both noisy and clean audio
                    out = model(x, y=target)
                    loss = criterion(out, target)
            else:
                out = model(x, y=target)
                # print('out shape', out.shape, "target shape", target.shape)
                loss = criterion(out, target)
            
            loss.backward()
            loss_item = loss.item()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Save audio samples periodically
            if total_steps % 1000 == 0:
                torchaudio.save(
                    f"{out_dir}/e_{str(epoch)}_i_{str(total_steps)}_out.wav", 
                    out.detach().reshape(1, -1).cpu().float(), 
                    args.sample_rate
                )
                torchaudio.save(
                    f"{out_dir}/e_{str(epoch)}_i_{str(total_steps)}_in.wav", 
                    x.detach().reshape(1, -1).cpu().float(), 
                    args.sample_rate
                )
                torchaudio.save(
                    f"{out_dir}/e_{str(epoch)}_i_{str(total_steps)}_target.wav", 
                    target.detach().reshape(1, -1).cpu().float(), 
                    args.sample_rate
                )
            
            # Log metrics periodically
            if total_steps % 50 == 0:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    break
                writer.add_scalar("loss", loss_item, total_steps)
                writer.add_scalar("lr", lr, total_steps)
                writer.add_scalar("min", torch.min(out).item(), total_steps)
                writer.add_scalar("max", torch.max(out).item(), total_steps)
                
            print(f"epoch {epoch}, step: {step}, total_steps: {total_steps} "
                    f"loss: {str(loss_item)[:8]}, lr: {str(lr)} "
                    f"min {str(torch.min(out).item())}, max {str(torch.max(out).item())}, "
                    f"avg {str(torch.mean(out.abs()).item())}, "
                    f"t_min {str(torch.min(target).item())[:8]}, "
                    f"t_max {str(torch.max(target).item())[:8]}, "
                    f"t_avg {str(torch.mean(target.abs()).item())[:8]}")
                
            
            # Save checkpoint periodically
            if step % 5000 == 0 and step != 0:
                # Keep only the most recent checkpoint
                old_checkpoints = glob.glob(f"{checkpoint_dir}/*")
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
                }, f"{checkpoint_dir}/{str(epoch)}_{str(i)}.pt")
        
        # Reset step counter at the end of each epoch
        step = 0

if __name__ == "__main__":
    main()
