import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from torch.nn.modules.loss import MSELoss 
from data import ImageField, TextField, RawField
from data import COCO, DataLoader, Flick30k as Flickr30k, Flickr8k
import evaluation
from evaluation import PTBTokenizer, Cider

from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
from models.transformer.backbone_wrapper import FrozenBackboneModel
from models.transformer.clip_backbone import CLIPBackbone


import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss, MSELoss
from tqdm import tqdm
import argparse

import pickle
import numpy as np
import itertools
from shutil import copyfile
import shutil

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DistributedSampler

import csv
import datetime

def init_csv_logger(exp_name):
    filename = f'training_log_{exp_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','learning_rate', 'train_loss', 'train_loss_ce', 'train_loss_itm', 'val_loss', 'val_cider', 'val_bleu1', 
                        'val_bleu4', 'val_meteor', 'val_rouge', 'test_cider', 
                        'test_bleu1', 'test_bleu4', 'test_meteor', 'test_rouge'])
    return filename

def write_to_csv(filename, epoch,learning_rate, train_loss, train_loss_ce, train_loss_itm, val_loss, val_scores, test_scores):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            learning_rate, 
            train_loss,
            train_loss_ce,
            train_loss_itm,
            val_loss,
            val_scores.get('CIDEr', 0),
            val_scores.get('BLEU', [0,0,0,0])[0],
            val_scores.get('BLEU', [0,0,0,0])[3],
            val_scores.get('METEOR', 0),
            val_scores.get('ROUGE', 0),
            test_scores.get('CIDEr', 0),
            test_scores.get('BLEU', [0,0,0,0])[0],
            test_scores.get('BLEU', [0,0,0,0])[3],
            test_scores.get('METEOR', 0),
            test_scores.get('ROUGE', 0)
        ])

import torch.nn.functional as F

def evaluate_loss(model, dataloader, loss_fn, text_field, e, device):

    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                text_input = text_field.decode(captions[:, 1:].cpu(), join_words=True)
                out, _, _, _, _, _, _ = model(mode='xe', images=detections, seq=captions, text_input=text_input)
                captions_gt = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                
                loss = loss_fn[0](out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=f"{running_loss / (it + 1):.3f}")
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

def evaluate_metrics(model, dataloader, text_field, e, device):
    import itertools
    model.eval()
    seq_len = 20
    beam_size = 5
    gen = {}
    gts = {}

    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        for it, (images, caps_gt, captions) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model(mode='rl', images=images, max_len=seq_len, eos_idx=text_field.vocab.stoi['<eos>'], beam_size=beam_size, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def train_xe(model, dataloader, optim, text_field,  scheduler, loss_fn, e, device):
    model.train()
    scheduler.step()
    if device == 0:
        lrs = [group['lr'] for group in optim.state_dict()['param_groups']]
        print('Learning Rates:', lrs)
    
    running_loss = .0
    running_loss_ce = .0
    running_loss_itm = .0
    running_loss_tim = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            text_input = text_field.decode(captions[:, 1:].cpu(), join_words=True)
            out, img_features, txt_features, cls_features, text_features, itm_logits_pos, tim_logits_pos = model(mode='xe', images=detections, seq=captions, text_input=text_input, epoch=e)
            
            batch_size = captions.size(0)
            shift_amount = torch.randint(1, max(2, batch_size), (1,)).item() if batch_size > 1 else 1
            
            captions_neg = torch.roll(captions, shifts=shift_amount, dims=0)
            text_input_neg = text_field.decode(captions_neg[:, 1:].cpu(), join_words=True)
            _, _, _, _, _, itm_logits_neg, _ = model.module(mode='xe', images=detections, seq=captions_neg, text_input=text_input_neg, epoch=e, compute_decoder=False)
            
            detections_neg = torch.roll(detections, shifts=shift_amount, dims=0)
            _, _, _, _, _, _, tim_logits_neg = model.module(mode='xe', images=detections_neg, seq=captions, text_input=text_input, epoch=e, compute_decoder=False)
            
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()

            loss_ce = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

            pos_labels = torch.ones(itm_logits_pos.size(0), dtype=torch.long, device=device)
            neg_labels = torch.zeros(itm_logits_neg.size(0), dtype=torch.long, device=device)
            
            loss_itm_pos = F.cross_entropy(itm_logits_pos, pos_labels)
            loss_itm_neg = F.cross_entropy(itm_logits_neg, neg_labels)
            loss_itm = (loss_itm_pos + loss_itm_neg) / 2
            
            loss_tim_pos = F.cross_entropy(tim_logits_pos, pos_labels)
            loss_tim_neg = F.cross_entropy(tim_logits_neg, neg_labels)
            loss_tim = (loss_tim_pos + loss_tim_neg) / 2

            loss = loss_ce + loss_itm + loss_tim

            loss.backward()
            optim.step()
            this_loss = loss.item()
            running_loss += this_loss
            running_loss_ce += loss_ce.item()
            running_loss_itm += loss_itm.item()
            running_loss_tim += loss_tim.item()

            pbar.set_postfix(loss=f"{running_loss / (it + 1):.3f}", ce=f"{running_loss_ce / (it + 1):.3f}", 
                             itm=f"{running_loss_itm / (it + 1):.3f}", tim=f"{running_loss_tim / (it + 1):.3f}")
            pbar.update()

    loss = running_loss / len(dataloader)
    loss_ce = running_loss_ce / len(dataloader)
    loss_itm_avg = (running_loss_itm + running_loss_tim) / (2 * len(dataloader))
    return loss, loss_ce, loss_itm_avg



def train_scst(model, dataloader, optim, cider, text_field,  scheduler_rl, e, device):
    running_reward = .0
    running_reward_baseline = .0

    model.train()
    scheduler_rl.step()
    if device == 0:
        lrs = [group['lr'] for group in optim.state_dict()['param_groups']]
        print('Learning Rates:', lrs)

    running_loss = .0
    seq_len = 20
    beam_size = 5
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        for it, (detections, caps_gt, captions) in enumerate(dataloader):
            detections = detections.to(device)
            text = captions.to(device)
            outs, log_probs = model(mode='rl', images=detections, max_len=seq_len, eos_idx=text_field.vocab.stoi['<eos>'], beam_size=beam_size, out_size=beam_size)
            optim.zero_grad()
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))

            caps_gen = evaluation.PTBTokenizer.tokenize(caps_gen)
            caps_gt = evaluation.PTBTokenizer.tokenize(caps_gt)
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=f"{running_loss / (it + 1):.3f}", reward=running_reward / (it + 1),
                           reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


def _changeConfig(config):
    REF_BATCH = 50.0
    batchSize = config.batch_size
    scale = batchSize / REF_BATCH
    config.xe_base_lr *= scale
    config.rl_base_lr *= scale

def _generalConfig(rank: int, worldSize: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8084"
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    random.seed(127)
    torch.manual_seed(127)
    np.random.seed(127)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank)


def train(rank, worldSize, args):
    _generalConfig(rank, worldSize)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print('Rank{}: Transformer Training'.format(rank))
    if rank == 0:
        csv_filename = init_csv_logger(args.exp_name)
        print(f'Training log will be saved to: {csv_filename}')
        
        epoch_save_dir = 'saved_transformer_models/epoch_save'
        os.makedirs(epoch_save_dir, exist_ok=True)
        print(f'Epoch models will be saved to: {epoch_save_dir}')

    from data import ImageDetectionsField 
    
    if args.use_extracted_features:
        if rank == 0:
            print(f"Using pre-extracted features from {args.features_path}")
        image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
    else:
        image_field = ImageField(config=args)
        
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    if args.exp_name == 'COCO':
        dataset = COCO(image_field, text_field, args.img_root_path, args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'Flickr30k':
        dataset = Flickr30k(image_field, text_field, args.img_root_path, args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'Flickr8k':
        dataset = Flickr8k(image_field, text_field, args.img_root_path, args.annotation_folder, args.annotation_folder)
    else:
        dataset = Flickr30k(image_field, text_field, args.img_root_path, args.annotation_folder, args.annotation_folder)
    
    train_dataset, val_dataset, test_dataset = dataset.splits
    
    if not os.path.isfile('vocab.pkl'):
        print("Rank{}: Building vocabulary".format(rank))
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab.pkl', 'wb'))
    else:
        print('Rank{}: Loading from vocabulary'.format(rank))
        text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    if args.use_extracted_features:
        class IdentityBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x
        
        backbone = IdentityBackbone()
        if rank == 0:
            print("Using IdentityBackbone since features are pre-extracted.")
    else:
        backbone = CLIPBackbone(model_name=args.clip_model_path, device=device)
        
    current_visual_dim = 1024
    current_text_dim = 768
    
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': args.m}, d_in=current_visual_dim)
    decoder = TransformerDecoderLayer(len(text_field.vocab), 1000, 3, text_field.vocab.stoi['<pad>'])
    head_model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder, args.num_clusters, len(text_field.vocab), 1000, text_field.vocab.stoi['<pad>'], 
                             text_d_model=current_text_dim, visual_dim=current_visual_dim, clip_model_name=args.clip_model_path)
    
    if args.train_clip_text:
        print(f"Rank{rank}: Unfreezing CLIP Text Encoder")
        for p in head_model.text_encoder.parameters():
            p.requires_grad = True
    trainable_visual = (args.train_clip_visual or args.train_clip_visual_layers > 0) if not args.use_extracted_features else False
    
    model = FrozenBackboneModel(backbone, head_model, trainable=trainable_visual).to(device)
    if trainable_visual and rank == 0:
        print("Unfreezing CLIP Visual Encoder")
        if args.train_clip_visual_layers > 0:
            print(f"Applying partial unfreezing to last {args.train_clip_visual_layers} layers of CLIP Visual Encoder")
    elif args.use_extracted_features and rank == 0:
        print("Using extracted features: CLIP Visual Encoder skipped/frozen (Identity).")

    if args.train_clip_visual_layers > 0 and not args.use_extracted_features:
        backbone.partial_unfreeze(args.train_clip_visual_layers)
        
    model = torch.nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)
 
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'add_text':text_field})
    ref_caps_train = train_dataset.text
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'add_text':text_field})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'add_text':text_field})


    def lambda_lr(s):
        print("lr_s:", s)
        if s <= 4:
            lr = args.xe_base_lr * s / 4
        elif s <= 9:
            lr = args.xe_base_lr 
        elif s <= 12:
            lr = args.xe_base_lr * 0.2
        # elif s <= 15:
        #     lr = args.xe_base_lr * 0.2 * 0.2
        else:
            lr = args.xe_base_lr * 0.2 * 0.2
        return lr
    
    def lambda_lr_rl(s):
        print("rl_s:", s)
        refine_epoch = args.refine_epoch_rl
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 10:
            lr = args.rl_base_lr * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 
        return lr

    clip_visual_params = []
    clip_text_params = []
    rest_params = []
    
    clip_visual_ids = set(map(id, backbone.parameters()))
    clip_text_ids = set(map(id, head_model.text_encoder.parameters()))

    for p in model.parameters():
        if p.requires_grad:
            if id(p) in clip_visual_ids:
                clip_visual_params.append(p)
            elif id(p) in clip_text_ids:
                clip_text_params.append(p)
            else:
                rest_params.append(p)
    
    optimizer_grouped_parameters = [{'params': rest_params, 'lr': 1.0}]
    if clip_visual_params:
        if rank == 0:
            print(f"Optimizer: Tracking {len(clip_visual_params)} CLIP Visual parameters (lr=0.5)")
        optimizer_grouped_parameters.append({'params': clip_visual_params, 'lr': 0.5})
    if clip_text_params:
        if rank == 0:
            print(f"Optimizer: Tracking {len(clip_text_params)} CLIP Text parameters (lr=0.1)")
        optimizer_grouped_parameters.append({'params': clip_text_params, 'lr': 0.1})

    optim = Adam(optimizer_grouped_parameters, lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    optim_rl = Adam(optimizer_grouped_parameters, lr=1, betas=(0.9, 0.98))
    scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    loss_align = MSELoss()
    loss = (loss_fn, loss_align)
    use_rl = False
    best_cider = .0
    best_test_cider = 0.
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best or args.resume_best_test:
        if args.resume_last:
            fname = 'saved_transformer_models/%s_last.pth' % args.exp_name
        elif args.resume_best:
            fname = 'saved_transformer_models/%s_best.pth' % args.exp_name
        else:
            fname = 'saved_transformer_models/%s_best_test.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            use_rl = data['use_rl']

            if use_rl:
                try:
                    optim_rl.load_state_dict(data['optimizer'])
                    scheduler_rl.load_state_dict(data['scheduler'])
                except ValueError:
                    print('Optimizer parameter groups changed (likely due to unfreezing), resetting optimizer and scheduler...')
                    for _ in range(start_epoch):
                        scheduler_rl.step()
            else:
                try:
                    optim.load_state_dict(data['optimizer'])
                    scheduler.load_state_dict(data['scheduler'])
                except ValueError:
                    print('Optimizer parameter groups changed (likely due to unfreezing), resetting optimizer and scheduler...')
                    for _ in range(start_epoch):
                        scheduler.step()

            print('Resuming from epoch %d, validation loss %f, best cider %f, and best_test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))
            print('patience:', data['patience'])

            if args.force_rl:
                use_rl = True
                optim_rl = Adam(optimizer_grouped_parameters, lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
                for k in range(start_epoch - 1):
                    scheduler_rl.step()
                print('Forced to RL stage after resuming.')

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        trainSampler = DistributedSampler(train_dataset, worldSize, rank)
        trainSampler.set_epoch(e)
        dataloader_train = DataLoader(train_dataset, sampler=trainSampler, batch_size=args.batch_size, pin_memory=True, drop_last=False, num_workers=args.workers, prefetch_factor=2, persistent_workers=(args.workers > 0))

        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        dict_trainSampler = DistributedSampler(dict_dataset_train, worldSize, rank)
        dict_trainSampler.set_epoch(e)
        dict_dataloader_train = DataLoader(dict_dataset_train, sampler=dict_trainSampler, batch_size=args.batch_size // 5,  pin_memory=True, drop_last=False, num_workers=args.workers, persistent_workers=(args.workers > 0))
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)
        
        if not use_rl:
            train_loss, train_loss_ce, train_loss_itm = train_xe(model, dataloader_train, optim, text_field, scheduler, loss_fn, e, rank)
            if rank == 0:
                pass
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, cider_train, text_field, scheduler_rl, e, rank)
            train_loss_ce, train_loss_itm = 0.0, 0.0
            if rank == 0:
                pass

        val_loss = evaluate_loss(model, dataloader_val, loss, text_field, e, rank)
        if rank == 0:
            pass

        scores = evaluate_metrics(model, dict_dataloader_val, text_field, e, rank)
        val_cider = scores['CIDEr']
        if rank == 0:
            if not use_rl:
                current_lr = optim.param_groups[0]['lr']
            else:
                current_lr = optim_rl.param_groups[0]['lr']
            print("Validation scores", scores)

        test_scores = evaluate_metrics(model, dict_dataloader_test, text_field, e, rank)
        test_cider = test_scores['CIDEr']
        if rank == 0:
            print("Test scores", test_scores)
            write_to_csv(csv_filename, e, current_lr, train_loss, train_loss_ce, train_loss_itm, val_loss, scores, test_scores)
            
            if e > 100:
                epoch_save_path = f'saved_transformer_models/epoch_save/{args.exp_name}_epoch_{e}.pth'
                torch.save({
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    'epoch': e,
                    'val_loss': val_loss,
                    'val_cider': val_cider,
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
                    'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
                    'patience': patience,
                    'best_cider': best_cider,
                    'best_test_cider': best_test_cider,
                    'use_rl': use_rl,
                    'train_loss': train_loss if not use_rl else train_loss[0] if isinstance(train_loss, tuple) else train_loss,
                }, epoch_save_path)
                print(f"Saved epoch {e} model to {epoch_save_path}")
            
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False

        if patience == 15:
            if e < args.xe_least:
                if rank == 0:
                    print('special treatment, e = {}'.format(e))
                use_rl = False
                switch_to_rl = False
                patience = 0
            elif not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(optimizer_grouped_parameters, lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
                
                for k in range(e-1):
                    scheduler_rl.step()
                if rank == 0:
                    print("Switching to RL")
            else:
                if rank == 0:
                    print('patience reached.')
                exit_train = True

        if e == args.xe_most:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(optimizer_grouped_parameters, lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

                for k in range(e-1):
                    scheduler_rl.step()
                if rank == 0:
                    print("Switching to RL")
        if rank == 0:
            if switch_to_rl and not best:
                data = torch.load('saved_transformer_models/%s_best.pth' % args.exp_name)
                torch.set_rng_state(data['torch_rng_state'])
                torch.cuda.set_rng_state(data['cuda_rng_state'])
                np.random.set_state(data['numpy_rng_state'])
                random.setstate(data['random_rng_state'])
                model.load_state_dict(data['state_dict'])
                print('RL_Resuming from epoch %d, validation loss %f, best_cider %f, and best test_cider %f' % (
                    data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))

            torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'val_loss': val_loss,
                'val_cider': val_cider,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
                'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
                'patience': patience,
                'best_cider': best_cider,
                'best_test_cider': best_test_cider,
                'use_rl': use_rl,
            }, 'saved_transformer_models/%s_last.pth' % args.exp_name)

            if best:
                copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/%s_best.pth' % args.exp_name)
            if best_test:
                copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/%s_best_test.pth' % args.exp_name)

        if exit_train:
            break


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--exp_name', type=str, default='COCO') 
    parser.add_argument('--batch_size', type=int, default= 120) 
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')  
    parser.add_argument('--resume_best_test', action='store_true')
    parser.add_argument('--img_root_path', type=str, required=True)

    parser.add_argument('--annotation_folder', type=str, required=True)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--xe_least', type=int, default=15)
    parser.add_argument('--xe_most', type=int, default=25)
    parser.add_argument('--refine_epoch_rl', type=int, default=25)

    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--num_clusters', type=int, default=6) 
    parser.add_argument('--text2text', type=int, default=0)
    parser.add_argument('--force_rl', action='store_true', default=False, help='Force switch to RL stage after resuming')
    parser.add_argument('--train_clip_visual', action='store_true', default=False, help='Unfreeze CLIP visual encoder wholly')
    parser.add_argument('--train_clip_visual_layers', type=int, default=0, help='Number of deep layers to unfreeze in CLIP visual encoder (0 means use --train_clip_visual setting)')
    parser.add_argument('--train_clip_text', action='store_true', default=False, help='Unfreeze CLIP text encoder')
    parser.add_argument('--use_extracted_features', action='store_true', default=False, help='Use pre-extracted .npy features instead of CLIP visual encoder')
    parser.add_argument('--features_path', type=str, required=True, help='Path to pre-extracted features')
    parser.add_argument('--clip_model_path', type=str, required=True, help='Path to HF model or CLIP model name. Default ViT-B/32')
    args = parser.parse_args()
    print(args)
    worldSize = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    _changeConfig(args)
    print('\nDistribute config', args)
    mp.spawn(train, (worldSize, args), worldSize)
