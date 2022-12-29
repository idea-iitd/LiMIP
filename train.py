"""
File adapted from https://github.com/ds4dm/learn2branch
"""
import os
import importlib
import argparse
import sys
import pathlib
import pickle
import numpy as np
from time import strftime
from shutil import copyfile
import gzip

import torch

import utilities
from utilities import log

from utilities_gcnn_torch import GCNNDataset as Dataset
from utilities_gcnn_torch import load_batch_gcnn as load_batch
import random
from datetime import datetime
import random
import pickle

random.seed(0)


def logits_to_memory(model, dataloader, top_k, optimizer=None):
    """
   
    """
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    memory_input_logits = []

    n_samples_processed = 0
    dataloader = torch.utils.data.DataLoader(dataloader, batch_size=1,
                                                   shuffle=False, num_workers=num_workers, collate_fn=load_batch)

    for batch in dataloader:
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = map(lambda x:x.to(device), batch)
        batched_states = (c, ei, ev, v, n_cs, n_vs)
        batch_size = n_cs.shape[0]
        weights /= batch_size # sum loss

        if optimizer:
            optimizer.zero_grad()
            _, logits = model(batched_states)  # eval mode
            logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands)  # apply padding now
            loss = _loss_fn(logits, best_cands, weights)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _, logits = model(batched_states)  # eval mode
                logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands), 0)  # filter candidate variables
                logits = model.pad_output(logits, n_cands)  # apply padding now
                loss = _loss_fn(logits, best_cands, weights)

                c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = c.cpu(), ei.cpu(), ev.cpu(), v.cpu(), n_cs.cpu(), n_vs.cpu(), n_cands.cpu(), cands.cpu(), best_cands.cpu(), cand_scores.cpu(), weights.cpu()
                logits = logits.cpu()
                storage_input = (c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights)
                memory_input_logits.append((storage_input, logits))
                pass

    

    return memory_input_logits

def process(model, dataloader, top_k, optimizer=None):
    """
   
    """
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = map(lambda x:x.to(device), batch)
        batched_states = (c, ei, ev, v, n_cs, n_vs)
        batch_size = n_cs.shape[0]
        weights /= batch_size # sum loss

        if optimizer:
            optimizer.zero_grad()
            _, logits = model(batched_states)  # eval mode
            logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands)  # apply padding now
            loss = _loss_fn(logits, best_cands, weights)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _, logits = model(batched_states)  # eval mode
                logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands), 0)  # filter candidate variables
                logits = model.pad_output(logits, n_cands)  # apply padding now
                loss = _loss_fn(logits, best_cands, weights)

        true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
        true_scores = true_scores.cpu().numpy()
        true_bestscore = true_bestscore.cpu().numpy()

        kacc = []
        for k in top_k:
            pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)

        mean_loss += loss.detach_().item() * batch_size
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc

def _loss_fn(logits, labels, weights):
    loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    return torch.sum(loss * weights)


def _loss_KLD(logits, logits_old, weights):
    loss = torch.nn.MSELoss(reduction='mean')(logits, logits_old)
    return loss

def reservoir_insert(reservoir_locations_array, memory_size, memory_input_logits, dict_memory_logits_kd, index_of_new_task):
    global samples_seen_for_memory


    if(index_of_new_task not in dict_memory_logits_kd):
        dict_memory_logits_kd[index_of_new_task] = []

    for iterator_memory_logits, new_CO_element in enumerate(memory_input_logits):
        samples_seen_for_memory +=1

        
        if( len(reservoir_locations_array) < memory_size ):
            reservoir_locations_array.append(index_of_new_task)
            dict_memory_logits_kd[index_of_new_task].append(new_CO_element)

        else:
            random_int = random.randint(0,samples_seen_for_memory)
            if(random_int< memory_size ):
                task_at_random_place = reservoir_locations_array[random_int]
                get_random_index_in_that_tasks_dict = random.randint(0, len(dict_memory_logits_kd[task_at_random_place])-1)
                dict_memory_logits_kd[task_at_random_place].pop(get_random_index_in_that_tasks_dict)
                reservoir_locations_array[random_int] = index_of_new_task
                dict_memory_logits_kd[index_of_new_task].append(new_CO_element)

def observe(t, dict_model_object, previous_data_loader, new_data_loader, dict_memory, dict_memory_logits_kd, reservoir_locations_array, device):
    net = dict_model_object['net']
    optimizer = dict_model_object['optimizer']
    current_task = dict_model_object['current_task']
    fisher_loss= dict_model_object['fisher_loss']
    fisher_att = dict_model_object['fisher_att']
    optpar = dict_model_object['optpar']
    

    lambda_l = dict_model_object['lambda_l']
    lambda_att = dict_model_object['lambda_att']
    
    old_task_weight = dict_model_object['old_task_weight']
    
    kl_weight = 1

    


    beta = 0
    task_loss_lambda = 1
    prob_of_old_task= 0.99
    net.train()

     
    samples_from_reservoir = dict_model_object['samples_from_reservoir']
    if(t==2):
        samples_from_reservoir=dict_model_object['samples_from_reservoir']


    if t != current_task:
        net.zero_grad()

        fisher_loss[current_task] = []
        fisher_att[current_task] = []
        optpar[current_task] = []
        

        mean_loss = 0
        mean_kacc = np.zeros(len(top_k))

        n_samples_processed = 0
        for batch in previous_data_loader:
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = map(lambda x: x.to(device),
                                                                                             batch)
       
            batched_states = (c, ei, ev, v, n_cs, n_vs)
            batch_size = n_cs.shape[0]
            weights /= batch_size  # sum loss

            _, logits, attention_weights1,attention_weights2 = model(batched_states,return_attention_weights=True)  # eval mode
            logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands),
                                     0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands)  # apply padding now
            loss = _loss_fn(logits, best_cands, weights)
            loss.backward()

        
            

            true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
            true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
            true_scores = true_scores.cpu().numpy()
            true_bestscore = true_bestscore.cpu().numpy()

            kacc = []
            for k in top_k:
                pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
                pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
                kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
            kacc = np.asarray(kacc)

            mean_loss += loss.detach_().item() * batch_size
            mean_kacc += kacc * batch_size
            n_samples_processed += batch_size

        mean_loss /= n_samples_processed
        mean_kacc /= n_samples_processed

        for p in net.parameters():
            pd = p.data.clone()
            pg = p.grad.data.clone().pow(2)
            pg =  pg*(batch_size*1.0) / n_samples_processed 
            
            optpar[current_task].append(pd)
            fisher_loss[current_task].append(pg)
            
            
        dict_model_object['current_task'] = index
            
       
    net.zero_grad()
 
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    loss = 0

    for iter_batch, batch in enumerate(new_data_loader):
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = map(lambda x: x.to(device),
                                                                                         batch)
   

        batched_states = (c, ei, ev, v, n_cs, n_vs)
        batch_size = n_cs.shape[0]
        weights /= batch_size  # sum loss

        if optimizer:
            optimizer.zero_grad()
        _, logits = model(batched_states)  # eval mode
        logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands),
                                 0)  # filter candidate variables
        logits = model.pad_output(logits, n_cands)  # apply padding now

        loss = _loss_fn(logits, best_cands, weights)

        chance = random.uniform(0, 1)
        old_task_batches_count = 0

        old_task_loss_scaler =1


        if(len(reservoir_locations_array)> 1 ):
            sampled_elements = random.sample(reservoir_locations_array, samples_from_reservoir)
            frequency_of_sampled_tasks = {x: sampled_elements.count(x) for x in sampled_elements}

            for task_old, frequency in frequency_of_sampled_tasks.items():
               
                KD_task_old = dict_memory_logits_kd[task_old]
                random.shuffle(KD_task_old)

                for iter_batch_old_task, (batch_old_task, logits_theta ) in enumerate(KD_task_old):
                    c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = map(
                        lambda x: x.to(device),
                        batch_old_task)
                    batched_states = (c, ei, ev, v, n_cs, n_vs)
                    batch_size = n_cs.shape[0]
                    weights /= batch_size  # sum loss
                                                                            
                    _, logits = model(batched_states)  # eval mode
                    logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands),
                                             0)  # filter candidate variables
                    logits = model.pad_output(logits, n_cands)  # apply padding now
                    
                    
                    logits_theta= logits_theta.to(device)

                    old_task_loss_KL = _loss_KLD(logits, logits_theta, weights)
                    logits_theta= logits_theta.cpu()

                    old_task_loss_KL = kl_weight*old_task_loss_KL


                    old_task_CE_loss = _loss_fn(logits, best_cands, weights)

                    loss_old_total =  old_task_loss_KL  #+     old_task_CE_loss
                    


                    loss_old_total = loss_old_total/samples_from_reservoir*1.0

                    loss_old_total= loss_old_total *old_task_weight

                    loss_old_total.backward()


                    if(frequency == iter_batch_old_task):

                        break

        loss.backward()


        for tt in range(t):
            for i, p in enumerate(net.parameters()):

                
                l =  fisher_loss[tt][i] 
                l = l * ((p - optpar[tt][i]).pow(2))
                fisher_loss_value =  (l.sum())*lambda_l
                fisher_loss_value.backward()

        optimizer.step()


        true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
        true_scores = true_scores.cpu().numpy()
        true_bestscore = true_bestscore.cpu().numpy()
  



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--model',
        help='GCNN model to be trained.',
        type=str,
        default='GAT_baseline_torch',
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--data_path',
        help='name of the folder where train and valid folders are present. Assumes `data/samples` as default.',
        type=str,
        default="data/samples_1.0",
    )
    parser.add_argument(
        '--l2',
        help='value of l2 regularizer',
        type=float,
        default=0.00001
    )
    parser.add_argument(
        '--sampled_batches',
        help='',
        type=int,
        default=150
    )

    parser.add_argument(
        '--number_of_epochs',
        help='',
        type=int,
        default=200
    )

    parser.add_argument(
        '--memory_size_buffer',
        help='',
        type=int,
        default=500
    )

    
    parser.add_argument(
        '-prob_seq','--prob_seq',
        help='prob_seq',
        type=str,
        default=None
    )

    
    parser.add_argument(
        '--lambda_l',
        help='lambda_l',
        type=int,
        default=100
    )
    
    parser.add_argument(
        '--lam_samples',
        help='lam_samples',
        type=int,
        default=300
    )
    
    
    parser.add_argument(
        '--lambda_att',
        help='lambda_att',
        type=float,
        default=0
    )
    
    parser.add_argument(
        '--old_task_weight',
        help='old_task_weight',
        type=float,
        default=1.5
    )
    
    parser.add_argument(
        '--samples_from_reservoir',
        help='samples_from_reservoir',
        type=int,
        default=30
    )
    
    
    
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 180
    epoch_size = 312
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 0.001
    patience = 15
    early_stopping = 30
    top_k = [1, 3, 5, 10]
    train_sample_limit = 150000
    valid_sample_limit = 30000
    num_workers = 10
    sampled_batches=args.sampled_batches
    sampled_task = batch_size*sampled_batches # randomly sample(for easier implementation)
    lambda_l = args.lambda_l
    lambda_att = args.lambda_att
    
    old_task_weight = args.old_task_weight
    
    number_of_epochs = args.number_of_epochs

    memory_size_buffer = args.memory_size_buffer # memory 


    samples_seen_for_memory = 0

    problem_folders = {

        
        
        
        'facdem_5_10_40_50':'100_100_5',
        'facdem_30_35_50_55':'100_100_5',
        'facdem_60_65_80_90':'100_100_5',
        'facdem_80_90_100_110':'100_100_5',
        
        
        'facdem_maxopen_80_90_100_110_95':'100_100_5',
        
        

              
        'indsetnewba_4_500': '4_500',
        'indsetnewba_4_750': '4_750',
        'indsetnewba_4_450': '4_450',
        'indsetnewba_5_450': '5_450',
        
        'indsetnewba_5_400': '5_400',
        'indsetnewba_5_350': '5_350',
        
        
        
        
        'setcover_densize_0.2': '700r_800c_0.2d',
        'setcover_densize_0.15': '700r_800c_0.15d',
        'setcover_densize_0.125': '700r_800c_0.125d',
        'setcover_densize_0.1': '700r_800c_0.1d',
        'setcover_densize_0.05': '700r_800c_0.05d',
        'setcover_densize_0.075': '700r_800c_0.075d',
        
    }
    
    problems_sequence = str(args.prob_seq).split('-') #['facilities','cauctions', 'setcover']

    print("problems_sequence" ,problems_sequence)

    # DIRECTORY NAMING
    modeldir = f"{args.model}"
    running_dir = f"trained_models/MODEL_{str('_'.join(problems_sequence))}/{modeldir}/{args.seed}"
    print(" running_dir ", running_dir)
    os.makedirs(running_dir)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"epoch_size: {epoch_size}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience }", logfile)
    log(f"early_stopping : {early_stopping }", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed: {args.seed}", logfile)
    log(f"l2 {args.l2}", logfile)

    

    ### NUMPY / TORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))


    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    importlib.reload(model)
    model = model.GATPolicy()
    del sys.path[0]
    model.to(device)


    ##;
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience, verbose=True)
    
    
    
    
    dict_model_object = {}
    dict_model_object['optimizer'] = optimizer
    dict_model_object['current_task'] = 0
    dict_model_object['fisher_loss'] = {}
    dict_model_object['fisher_att'] = {}
    dict_model_object['optpar'] = {}
    dict_model_object['net'] = model
    dict_model_object['scheduler'] = scheduler
    dict_model_object['lambda_l'] = lambda_l
    dict_model_object['lambda_att'] = lambda_att
    
    dict_model_object['old_task_weight'] = old_task_weight
    
    dict_model_object['samples_from_reservoir'] = args.samples_from_reservoir
    


    previous_data_loader = None
    dict_task_wise_valid_data_loaders = {}
    dict_memory = {}
    dict_memory_logits_kd = {}

    reservoir_locations_array = []

    for index, problem in enumerate(problems_sequence):

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience,
                                                               verbose=True)
        best_loss = np.inf
        plateau_count = 0
        
        dict_model_object['optimizer'] = optimizer
        dict_model_object['scheduler'] = scheduler

        print("index top ", index)
        print(" problem ", problem)
        problem_folder = problem_folders[problem]
        print(" problem_folder ", problem_folder)


        ### SET-UP DATASET ###
        dir = f'data/samples/{problem}/{problem_folder}'
        if args.data_path:
            dir = f"{args.data_path}/{problem}/{problem_folder}"
            

        train_files = list(pathlib.Path(f'{dir}/train').glob('sample_*.pkl'))
        valid_files = list(pathlib.Path(f'{dir}/valid').glob('sample_*.pkl'))

        log(f"{len(train_files)} training samples", logfile)
        log(f"{len(valid_files)} validation samples", logfile)

        train_files = [str(x) for x in train_files]
        valid_files = [str(x) for x in valid_files]

        valid_data = Dataset(valid_files)
        valid_data = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size,
                                shuffle = False, num_workers = num_workers, collate_fn = load_batch)

        dict_task_wise_valid_data_loaders[index]  = valid_data
        dict_task_wise_valid_data_loaders[index]  = valid_data

        pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
        pretrain_data = Dataset(pretrain_files)
        pretrain_data = torch.utils.data.DataLoader(pretrain_data, batch_size=pretrain_batch_size,
                                shuffle = False, num_workers = num_workers, collate_fn = load_batch)


        epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)

        sampled_memory_files = rng.choice(train_files, sampled_task, replace=False)


        train_data = Dataset(epoch_train_files)
        train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                        shuffle = True, num_workers = num_workers, collate_fn = load_batch)

        memory_data = Dataset(sampled_memory_files)

        dict_memory[index] = memory_data

        valid_loss, valid_kacc = process(model, valid_data, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

 
        continue_running = True
        
        for epoch in range(number_of_epochs):
            
            
            print("epoch ", epoch)

            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
            train_data = Dataset(epoch_train_files)
            train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                    shuffle = False, num_workers = num_workers, collate_fn = load_batch)

            observe(index,dict_model_object, previous_data_loader, train_data, dict_memory, dict_memory_logits_kd,reservoir_locations_array, device)


            if(epoch%10==0): 

                for prev_task_iterator in range(0, index + 1):
                    valid_loss, valid_kacc = process(model, dict_task_wise_valid_data_loaders[prev_task_iterator],
                                                     top_k, None)
                    log(f"VALID LOSS: {valid_loss:0.3f} " + "".join(
                        [f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

                    if(prev_task_iterator==index):
                        scheduler.step(valid_loss)
                        
                        if(valid_loss>=best_loss):
                            plateau_count += 1
                            if plateau_count % early_stopping == 0:
                                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                                continue_running=False
                                break
                            if plateau_count % patience == 0:
                                pass
                        else:
                            best_loss = valid_loss
                            plateau_count=0

                            

            if(epoch%3 ==0):
                model.save_state(os.path.join(running_dir, 'params_at_task{}_epoch{}.pkl'.format(index, epoch)))
                model.save_state(os.path.join(running_dir, 'checkpoint.pkl'))

                
            if(continue_running==False):
                break


        memory_input_logits = logits_to_memory(model, memory_data, top_k, None)

        reservoir_insert(reservoir_locations_array, memory_size_buffer, memory_input_logits, dict_memory_logits_kd,
                             index)


        sampled_reg_files = rng.choice(sampled_memory_files,args.lam_samples, replace=False)
        previous_data = Dataset(sampled_reg_files)
        
        previous_data_loader = torch.utils.data.DataLoader(previous_data, batch_size=batch_size,
                                                           shuffle=False, num_workers=num_workers,
                                                           collate_fn=load_batch)
        