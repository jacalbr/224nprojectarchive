import time, random, numpy as np, argparse, sys, re, os, math
from types import SimpleNamespace
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_para, model_eval_sts

from loss_fn import online_contrastive_loss, kl_loss, sym_kl_loss, inf_norm

TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

WARM_UP_EPOCH = 1


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        
        # bert output pooling
        self.pool = config.pool
        if self.pool == 'none':
            self.bert_out_select = 'pooler_output'
        else:
            self.bert_out_select = 'last_hidden_state'
        self.task_opt = config.task_opt
        #========================
        # sst
        #========================
        if config.task == 'sst':
            if config.task_opt == 'base':
                self.sst_head = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(config.hidden_dropout_prob)),
                    ('fc', nn.Linear(config.hidden_size, config.num_labels))
                ]))
        #========================
        # paras
        #========================
        elif config.task == 'para':
            if config.task_opt == 'base':
                self.para_haed = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(config.hidden_dropout_prob)),
                    ('fc', nn.Linear(config.hidden_size*2, 1))
                ]))
            elif config.task_opt == 'diff_concat':
                self.para_haed = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(config.hidden_dropout_prob)),
                    ('fc', nn.Linear(config.hidden_size*3, 1))
                ]))
            elif config.task_opt == 'online_contrastive':
                self.para_haed = nn.CosineSimilarity()
            elif config.task_opt == 'cross':
                self.para_haed = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(config.hidden_dropout_prob)),
                    ('fc', nn.Linear(config.hidden_size, 1))
                ]))
        #========================
        # sts
        #========================
        elif config.task == 'sts':
            if config.task_opt == 'base':
                self.sts_head = nn.CosineSimilarity()
            if config.task_opt =='cross':
                self.sts_head = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(config.hidden_dropout_prob)),
                    ('fc', nn.Linear(config.hidden_size, 1)),
                    ('sigmoid', nn.Sigmoid())
                ]))


    def forward(self, input_ids, attention_mask, token_type_ids=None):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        x = self.bert(input_ids, attention_mask, token_type_ids)
        out = x[self.bert_out_select]
        if self.pool == 'max':
            out = out.max(1)[0]
        elif self.pool == 'mean':
            out = out.mean(1)
        elif self.pool == 'attn':
            out = attention_pool(x)
        return out
        

    def forward_bert_encoder (self, embed, attention_mask):
        if self.pool == 'max':
            hidden = self.bert.encode(embed, attention_mask).max(1)[0]
        elif self.pool == 'mean':
            hidden = self.bert.encode(embed, attention_mask).mean(1)
        elif self.pool == 'attn':
            hidden = self.bert.encode(embed, attention_mask).mean(1)
            hidden = attention_pool(hidden)
        elif self.pool == 'none':
            sequence_output = self.bert.encode(embed, attention_mask)
            first_tk = sequence_output[:, 0]
            first_tk = self.bert.pooler_dense(first_tk)
            hidden = self.bert.pooler_af(first_tk)
        return hidden
    

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        features = self.forward(input_ids, attention_mask)
        logits = self.sst_head(features)
        return logits
    

    def predict_sentiment_smart (self, input_ids, attention_mask, labels,
                                 weight: float = 5.0,
                                 noise_var: float = 1e-5,
                                 step_size: float = 1e-3, 
                                 epsilon: float = 1e-5,):
        
        embed = self.bert.embed(input_ids)
        hidden = self.forward_bert_encoder(embed=embed, attention_mask=attention_mask)

        logits = self.sst_head(hidden)

        ce_loss = F.cross_entropy(logits, labels.view(-1), reduction='sum') / args.batch_size

        noise = torch.randn_like(embed, requires_grad = True) * noise_var 
        embed_perturbed = embed + noise
        hidden_perturbed = self.forward_bert_encoder(embed_perturbed, attention_mask)
        logits_perturbed = self.sst_head(hidden_perturbed)
        loss = F.mse_loss(logits_perturbed, logits.detach())
        noise_gradient, = torch.autograd.grad(loss, noise)
        step = noise + step_size * noise_gradient 
        step1_norm = inf_norm(step)
        noise = step / (step1_norm + epsilon)
        noise = noise.detach().requires_grad_()
        embed_perturbed = embed + noise
        hidden_perturbed = self.forward_bert_encoder(embed_perturbed, attention_mask)
        logits_perturbed = self.sst_head(hidden_perturbed)
        adv_loss = sym_kl_loss(logits_perturbed, logits)
        return ce_loss + weight * adv_loss



    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2 = None, attention_mask_2 = None,
                           token_type_ids=None):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        if self.task_opt == 'cross':
            features = self.forward(input_ids_1, attention_mask_1, token_type_ids)
            logits = self.para_haed(features)
            return logits
        else:
            features1 = self.forward(input_ids_1, attention_mask_1)
            features2 = self.forward(input_ids_2, attention_mask_2)
            if self.task_opt == 'base':
                logits = self.para_haed(torch.cat((features1, features2), dim=-1))
            elif self.task_opt == 'diff_concat':
                logits = self.para_haed(torch.cat((features1, features2, features1-features2), dim=-1))
            elif self.task_opt == 'online_contrastive':
                logits = self.para_haed(features1, features2)
            return logits
        

    def predict_paraphrase_smart_cross(self,
                                 input_ids, attention_mask, token_type_ids, labels,
                                 weight: float = 5.0,
                                 noise_var: float = 1e-5,
                                 step_size: float = 1e-3, 
                                 epsilon: float = 1e-5,):
        embed = self.bert.embed(input_ids, tk_type_ids=token_type_ids)
        hidden = self.forward_bert_encoder(embed=embed, attention_mask=attention_mask)

        logits = self.para_haed(hidden)

        ce_loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits), reduction='mean')

        noise = torch.randn_like(embed, requires_grad = True) * noise_var 
        embed_perturbed = embed + noise
        hidden_perturbed = self.forward_bert_encoder(embed_perturbed, attention_mask)
        logits_perturbed = self.para_haed(hidden_perturbed)
        loss = F.mse_loss(logits_perturbed, logits.detach())
        noise_gradient, = torch.autograd.grad(loss, noise)
        step = noise + step_size * noise_gradient 
        step1_norm = inf_norm(step)
        noise = step / (step1_norm + epsilon)
        noise = noise.detach().requires_grad_()
        embed_perturbed = embed + noise
        hidden_perturbed = self.forward_bert_encoder(embed_perturbed, attention_mask)
        logits_perturbed = self.para_haed(hidden_perturbed)
        adv_loss = sym_kl_loss(logits_perturbed, logits)
        return ce_loss + weight * adv_loss
        
    
    def predict_paraphrase_smart(self,
                                 input_ids_1, attention_mask_1,
                                 input_ids_2, attention_mask_2, labels,
                                 weight: float = 1.0,
                                 noise_var: float = 1e-5,
                                 step_size: float = 1e-3, 
                                 epsilon: float = 1e-5,):
        if self.task_opt != 'base' and self.pool == 'mean':
            print('currently it only support base task option')
        embed1 = self.bert.embed(input_ids_1)
        embed2 = self.bert.embed(input_ids_2)

        hidden1 = self.bert.encode(embed1, attention_mask_1).mean(1)
        hidden2 = self.bert.encode(embed2, attention_mask_2).mean(1)

        logits = self.para_haed(torch.cat((hidden1, hidden2), dim=-1))

        ce_loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits), reduction='mean')

        noise1 = torch.randn_like(embed1, requires_grad = True) * noise_var 
        noise2 = torch.randn_like(embed2, requires_grad = True) * noise_var 


        embed1_perturbed = embed1 + noise1
        embed2_perturbed = embed2 + noise2
        hidden1_perturbed = self.bert.encode(embed1_perturbed, attention_mask_1).mean(1)
        hidden2_perturbed = self.bert.encode(embed2_perturbed, attention_mask_2).mean(1)
        logits1_perturbed = self.para_haed(torch.cat((hidden1_perturbed, hidden2), dim=-1))
        logits2_perturbed = self.para_haed(torch.cat((hidden1, hidden2_perturbed), dim=-1))

        loss1 = kl_loss(logits1_perturbed, logits.detach())
        loss2 = kl_loss(logits2_perturbed, logits.detach())

        noise1_gradient, = torch.autograd.grad(loss1, noise1)
        noise2_gradient, = torch.autograd.grad(loss2, noise2)

        step1 = noise1 + step_size * noise1_gradient 
        step2 = noise2 + step_size * noise2_gradient 

        step1_norm = inf_norm(step1)
        step2_norm = inf_norm(step2)
        noise1 = step1 / (step1_norm + epsilon)
        noise2 = step2 / (step2_norm + epsilon)

        noise1 = noise1.detach().requires_grad_()
        noise2 = noise2.detach().requires_grad_()

        embed1_perturbed = embed1 + noise1
        embed2_perturbed = embed2 + noise2 
        hidden1_perturbed = self.bert.encode(embed1_perturbed, attention_mask_1).mean(1) 
        hidden2_perturbed = self.bert.encode(embed2_perturbed, attention_mask_2).mean(1) 
        logits_perturbed = self.para_haed(torch.cat((hidden1_perturbed, hidden2_perturbed), dim=-1))

        adv_loss = sym_kl_loss(logits_perturbed, logits)
        return ce_loss + weight * adv_loss



    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2=None, attention_mask_2=None,
                           token_type_ids=None):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        if self.task_opt == 'cross':
            features = self.forward(input_ids_1, attention_mask_1, token_type_ids=token_type_ids)
            scores = self.sts_head(features)
        else:
            features1 = self.forward(input_ids_1, attention_mask_1)
            features2 = self.forward(input_ids_2, attention_mask_2)
            scores = self.sts_head(features1, features2)
        return scores
    

    def predict_similarity_smart_cross(self,
                                 input_ids, attention_mask, token_type_ids, labels,
                                 weight: float = 20.0,
                                 noise_var: float = 1e-5,
                                 step_size: float = 1e-3, 
                                 epsilon: float = 1e-5,):
        embed = self.bert.embed(input_ids, tk_type_ids=token_type_ids)
        hidden = self.forward_bert_encoder(embed=embed, attention_mask=attention_mask)

        logits = self.sts_head(hidden)

        mse_loss = F.mse_loss(logits.view(-1), labels.view(-1).type_as(logits))

        noise = torch.randn_like(embed, requires_grad = True) * noise_var 
        embed_perturbed = embed + noise
        hidden_perturbed = self.forward_bert_encoder(embed_perturbed, attention_mask)
        logits_perturbed = self.sts_head(hidden_perturbed)
        loss = F.mse_loss(logits_perturbed, logits.detach())
        noise_gradient, = torch.autograd.grad(loss, noise)
        step = noise + step_size * noise_gradient 
        step1_norm = inf_norm(step)
        noise = step / (step1_norm + epsilon)
        noise = noise.detach().requires_grad_()
        embed_perturbed = embed + noise
        hidden_perturbed = self.forward_bert_encoder(embed_perturbed, attention_mask)
        logits_perturbed = self.sts_head(hidden_perturbed)
        adv_loss = sym_kl_loss(logits_perturbed, logits)
        return mse_loss + weight * adv_loss
    

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


#========================================================
# Additional function
#========================================================

def attention_pool (x):
    ftk = x['last_hidden_state'][:, 0]
    last_state = x['last_hidden_state'][:, 1:]
    e = torch.einsum('nj,nij->ni', ftk, last_state)
    a = torch.softmax(e, -1)
    out = ((a[..., None] * last_state).sum(1) + ftk) / 2
    return out

def cosine_update_lr (optimizer, initial_lr, epoch_progress, max_epoch):
    warm_up_epoch = WARM_UP_EPOCH
    if epoch_progress <= warm_up_epoch:
        lr = initial_lr * epoch_progress / warm_up_epoch
    else:
        lr = initial_lr * 0.5 * (1. + math.cos(math.pi * (epoch_progress - warm_up_epoch) / (max_epoch - warm_up_epoch)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr   


#========================================================
# SST: Stanford Sentiment Treebank Sentiment Analysis
# Given: SENT
# Predict: 0,1,2,3,4 (classification)
#========================================================
def train_sst (args, device):
    sst_train_data, num_labels, _, _ = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels, _, _= load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Set up model confdig
    base_config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                   'num_labels': len(num_labels),
                   'hidden_size': 768,
                   'data_dir': '.',
                   'option': args.option,
                   'task': 'sst',
                   'task_opt': args.sst_opt,
                   'pool': args.pool}
    if args.sst_opt == 'base':
        config = base_config
    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    initial_lr = lr
    lrs = []
    weight_decay = args.wd 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    max_epoch = args.epochs
    best_dev_acc = 0

    # for cosine scheduling
    num_steps_per_train_epoch = len(sst_train_dataloader)

    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        num_batches = 0
        batch_idx = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                    batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()

            if args.smart:
                loss = model.predict_sentiment_smart(b_ids, b_mask, b_labels)
            else:    
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            #====================cosine scheduling======================
            epoch_progress = epoch + min((batch_idx+1)/num_steps_per_train_epoch , 1)
            batch_idx += 1
            cosine_update_lr(optimizer, initial_lr, epoch_progress, max_epoch)
            if args.debug:
                lrs.append(optimizer.param_groups[0]["lr"])
            #===========================================================
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, 'sst-'+args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    if args.debug:
        plt.plot(np.arange(len(lrs)), lrs)
        plt.show()

#========================================================
# para: Paraphrase Detection
# Given: SENT1, SENT2
# Predict: 0, 1
#========================================================
def train_para (args, device):
    _, _, para_train_data, _ = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    _, _, para_dev_data, _= load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    para_train_data = SentencePairDataset(para_train_data, args, cross=args.cross)
    para_dev_data = SentencePairDataset(para_dev_data, args, cross=args.cross)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=256,
                                    collate_fn=para_dev_data.collate_fn)
    
    # Set up model confdig
    base_config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                   'hidden_size': 768,
                   'data_dir': '.',
                   'option': args.option,
                   'task': 'para',
                   'task_opt': args.para_opt,
                   'pool': args.pool}

    if args.para_opt == 'base':
        config = base_config
    elif args.para_opt == 'diff_concat':
        config = base_config
    elif args.para_opt == 'online_contrastive':
        config = base_config

    if args.cross:
        config['task_opt'] = 'cross'
    
    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    initial_lr = lr
    lrs = []
    weight_decay = args.wd 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    max_epoch = args.epochs
    best_dev_acc = 0

    # for cosine scheduling
    num_steps_per_train_epoch = len(para_train_dataloader)

    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        num_batches = 0
        batch_idx = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            if args.cross:
                (token_ids, token_type_ids, attention_mask,
                 labels,sent_ids) = (batch['token_ids'], batch['token_type_ids'],
                            batch['attention_mask'], batch['labels'], batch['sent_ids'])
                
                token_ids = token_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                if args.smart:
                    loss = model.predict_paraphrase_smart_cross(token_ids, attention_mask, token_type_ids, labels)
                else:
                    logits = model.predict_paraphrase(token_ids, attention_mask, token_type_ids=token_type_ids)
                    loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits), reduction='mean')

            else:
                (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                            batch['token_ids_2'], batch['attention_mask_2'],
                            batch['labels'], batch['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                if args.smart:
                    loss = model.predict_paraphrase_smart(b_ids1, b_mask1, b_ids2, b_mask2, b_labels)
                else:
                    logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                    if args.para_opt == 'online_contrastive':
                        loss = online_contrastive_loss(logits.view(-1), b_labels.view(-1), margin=0.6)
                    else:
                        loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.view(-1).type_as(logits), reduction='mean')
                    # loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            loss.backward()
            #====================cosine scheduling======================
            epoch_progress = epoch + min((batch_idx+1)/num_steps_per_train_epoch , 1)
            batch_idx += 1
            cosine_update_lr(optimizer, initial_lr, epoch_progress, max_epoch)
            if args.debug:
                lrs.append(optimizer.param_groups[0]["lr"])
            #===========================================================
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        train_acc, train_f1, *_ = model_eval_para(para_train_dataloader, model, device, cross=args.cross)
        dev_acc, dev_f1, *_ = model_eval_para(para_dev_dataloader, model, device, cross=args.cross)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, 'para-'+args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    if args.debug:
        plt.plot(np.arnage(len(lrs)), lrs)


#========================================================
# sts: Semantic Textual Similarity
# Given: SENT1, SENT2
# Predict: similarity score (-1 -> 1), label (0 -> 5) (regression)
#========================================================
def train_sts (args, device):
    _, _, _, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    _, _, _, sts_dev_data= load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True, cross=args.cross)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True, cross=args.cross)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    
    # Set up model confdig
    base_config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                   'hidden_size': 768,
                   'data_dir': '.',
                   'option': args.option,
                   'task': 'sts',
                   'task_opt': args.sts_opt,
                   'pool': args.pool}

    if args.sts_opt == 'base':
        config = base_config
    if args.cross:
        config['task_opt'] = 'cross'

    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    initial_lr = lr
    lrs = []
    weight_decay = args.wd 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    max_epoch = args.epochs
    best_dev_corr = 0

    # for cosine scheduling
    num_steps_per_train_epoch = len(sts_train_dataloader)

    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        num_batches = 0
        batch_idx = 0
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            if args.cross:
                (token_ids, token_type_ids, attention_mask,
                 labels,sent_ids) = (batch['token_ids'], batch['token_type_ids'],
                            batch['attention_mask'], batch['labels'], batch['sent_ids'])
                train_labels = labels/5.0
                
                token_ids = token_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                train_labels = train_labels.to(device)
                

                optimizer.zero_grad()

                if args.smart:
                    loss = model.predict_similarity_smart_cross(token_ids, attention_mask, token_type_ids, train_labels)
                else:
                    scores = model.predict_similarity(token_ids, attention_mask, token_type_ids=token_type_ids)
                    loss = F.mse_loss(scores.view(-1), train_labels.view(-1).type_as(scores))
            else:
                (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                            batch['token_ids_2'], batch['attention_mask_2'],
                            batch['labels'], batch['sent_ids'])
                # convert label to -1.0 -> 1.0
                train_labels = b_labels/5.0

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                train_labels = train_labels.to(device)

                optimizer.zero_grad()
                scores = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.mse_loss(scores.view(-1), train_labels.view(-1).type_as(scores))
            loss.backward()
            #====================cosine scheduling======================
            epoch_progress = epoch + min((batch_idx+1)/num_steps_per_train_epoch , 1)
            batch_idx += 1
            cosine_update_lr(optimizer, initial_lr, epoch_progress, max_epoch)
            if args.debug:
                lrs.append(optimizer.param_groups[0]["lr"])
            #===========================================================
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        train_corr, *_ = model_eval_sts(sts_train_dataloader, model, device, cross=args.cross)
        dev_corr, *_ = model_eval_sts(sts_dev_dataloader, model, device, cross=args.cross)
        if dev_corr > best_dev_corr:
            best_dev_corr = dev_corr
            save_model(model, optimizer, args, config, 'sts-'+args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train corr :: {train_corr :.3f}, dev acc :: {dev_corr :.3f}")
    if args.debug:
        plt.plot(np.arnage(len(lrs)), lrs)


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device(f'cuda:{args.gpu}') if args.use_gpu else torch.device('cpu')
    args.device = device

    if args.sst: 
        train_sst(args, device)
    if args.para:
        train_para(args, device)
    if args.sts:
        train_sts(args, device)
    
model_to_test={
    'sst': 'sst-finetune-5-5e-05-none-smart.pt', 
    'para': 'para-finetune-5-1e-05-none-cross-smart.pt',
    'sts': 'sts-finetune-15-0.0001-none-cross-smart.pt',
}

def test_model(args):
    with torch.no_grad():
        device = torch.device(f'cuda:{args.gpu}') if args.use_gpu else torch.device('cpu')
        models = {}
        for task in ['sst', 'para', 'sts']:
            # TODO: this only work when checkpoint is store at pwd
            file_path = model_to_test[task]
            saved = torch.load(file_path)
            config = saved['model_config']

            model = MultitaskBERT(config)
            model.load_state_dict(saved['model'])
            model = model.to(device)
            print(f"Loaded model to test from {file_path}")
            models[task] = model

        test_model_multitask(args, models, device)


def get_args() -> argparse.ArgumentParser: 
    parser = argparse.ArgumentParser()
    # sst I/O args
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")
    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")
    # para I/O args
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
    # sts input file args
    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")
    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")
    # training config
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--debug", action='store_true')
    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--wd", type=float, help="Weight decay for AdmaW", default=1e-4)
    # bert pooling
    parser.add_argument("--pool", type=str,
                        help='pooling method for output, none will be using first token',
                        choices=('none', 'max', 'mean', 'attn'), default="none")
    # task selection
    parser.add_argument("--sst", help='Train sst task', action='store_true')
    parser.add_argument("--para", help='Train para task', action='store_true')
    parser.add_argument("--sts", help='Train sts task', action='store_true')

    # options use to train subtask
    parser.add_argument('--sst_opt', choices=['base'], default='base')
    parser.add_argument('--para_opt', choices=['base', 'diff_concat', 'online_contrastive'], default='base')
    parser.add_argument('--smart', help='Use SMART regularization', action='store_true')
    parser.add_argument('--sts_opt', choices=['base'], default='base')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cross', help='cross encoder', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-{args.pool}' # save path
    if args.cross:
        args.filepath += '-cross'
    if args.smart:
        args.filepath += '-smart'
    args.filepath += '.pt'
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    if args.test:
        test_model(args)
