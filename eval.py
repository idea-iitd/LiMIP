"""
File adapted from https://github.com/ds4dm/learn2branch
"""
import os
import sys
import importlib
import argparse
import csv
import math
import numpy as np
import time
import pickle
import pyscipopt as scip

# import tensorflow as tf
import torch
import utilities


class PolicyBranching(scip.Branchrule):

    def __init__(self, policy, device):
        super().__init__()

        self.policy_type = policy['type']
        self.policy_name = policy['name']
        self.device = device
        if self.policy_type == 'gcnn':
            model = policy['model']
            model.restore_state(policy['parameters'])
            model.to(device)
            model.eval()
            # model.dict_norm_task = policy['dict_norm_task']
            # print('dict_norm_task',model.dict_norm_task)
            self.policy = model.forward

        else:
            raise NotImplementedError

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):

        # SCIP internal branching rule
        if self.policy_type == 'internal':
            result = self.model.executeBranchRule(self.policy, allowaddcons)

        # custom policy branching
        else:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getIndex() for var in candidate_vars]

            state = utilities.extract_state(self.model, self.state_buffer)
            c,e,v = state
            state = (
                torch.as_tensor(c['values'], dtype=torch.float32),
                torch.as_tensor(e['indices'], dtype=torch.long),
                torch.as_tensor(e['values'], dtype=torch.float32),
                torch.as_tensor(v['values'], dtype=torch.float32),
                torch.as_tensor(c['values'].shape[0], dtype=torch.int32),
                torch.as_tensor(v['values'].shape[0], dtype=torch.int32),
            )

            state = map(lambda x:x.to(self.device), state)
            with torch.no_grad():
                _, var_logits = self.policy(state)
                var_logits = torch.squeeze(var_logits, 0).cpu().numpy()

            candidate_scores = var_logits[candidate_mask]
            best_var = candidate_vars[candidate_scores.argmax()]

            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset','faccost', 'tsp', 'setcover_densize','facdem','facdem_maxopen','indsetnewba'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=3,
    )
    parser.add_argument(
        '--model_name',
        help='searches for this model_name in respective trained_models folder',
        type=str,
        default='GAT_baseline_torch',
    )
    
    
    parser.add_argument(
        '-density', '--density',
        help='density',
        type=float,
        default=None,
    )
    
    parser.add_argument(
        '-affinity', '--affinity',
        help='affinity',
        type=int,
        default=None,
    )
 
    parser.add_argument(
        '-indnodes', '--indnodes',
        help='indnodes',
        type=int,
        default=None,
    )
    
    parser.add_argument(
        '-ratio', '--ratio',
        help='ratio',
        type=int,
        default=None,
    )
    
    parser.add_argument(
        '-add_item_prob', '--add_item_prob',
        help='add_item_prob',
        type=float,
        default=None,
    )
    

    parser.add_argument(
        '-path_load','--path_load',
        help='path_load',
        type=str,
        default=None
    )
    
    
    parser.add_argument(
        '-epoch_load','--epoch_load',
        help='epoch_load',
        type=str,
        default=None
    )
    
    
    parser.add_argument(
        '-faccost', '--faccost',
        help='faccost',
        type=int,
        default=None,
    )

    
    parser.add_argument(
        '-facdemlow', '--facdemlow',
        help='facdemlow',
        type=int,
        default=None,
    )


    parser.add_argument(
        '-facdemhigh', '--facdemhigh',
        help='facdemhigh',
        type=int,
        default=None,
    )
    

    parser.add_argument(
        '-facdemcaplow', '--facdemcaplow',
        help='facdemcaplow',
        type=int,
        default=None,
    )


    parser.add_argument(
        '-facdemcaphigh', '--facdemcaphigh',
        help='facdemcaphigh',
        type=int,
        default=None,
    )

    
    parser.add_argument(
        '-facmaxopen', '--facmaxopen',
        help='facmaxopen',
        type=int,
        default=None,
    )


    
    parser.add_argument(
        '-tsp_num_cities', '--tsp_num_cities',
        help='tsp_num_cities',
        type=int,
        default=-1,
    )

    parser.add_argument(
        '-tsp_num_modes', '--tsp_num_modes',
        help='tsp_num_modes',
        type=int,
        default=-1,
    )

    parser.add_argument(
        '-time_limit', '--time_limit',
        help='time_limit',
        type=int,
        default=2700,
    )
    
    args = parser.parse_args()

    instances = []
    seeds = [0,1,2,3,4]#, 1, 2]
    gcnn_models = [args.model_name]
    time_limit = args.time_limit #2700
    

    ## OUTPUT
    device = "CPU" if args.gpu == -1 else "GPU"
    result_file = f"GAT_{device}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    # eval_dir = f"eval_results/{args.problem}"
    eval_dir = f"eval/{args.problem}"
    
    os.makedirs(eval_dir, exist_ok=True)
    
    path_load = args.path_load
    
  
    epoch_load = args.epoch_load
    
    print('path_load', path_load)

    full_epoch_load_path = path_load + epoch_load
    print(" full_epoch_load_path ", full_epoch_load_path)


 
    if args.problem == 'indsetnewba':
        instances += [{'type': 'small', 'path': f"data/instances/indsetnewba_{args.affinity}/transfer_{args.indnodes}_{args.affinity}/instance_{i+1}.lp"} for i in range(20)]
        
  
    elif args.problem == 'facdem':
        facdemlow = args.facdemlow 
        facdemhigh = args.facdemhigh
        
        
        facdemcaplow = args.facdemcaplow
        facdemcaphigh = args.facdemcaphigh
        
        instances += [{'type': 'small', 'path': f"data/instances/facdem_{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}/transfer_100_100_5/instance_{i+1}.lp"} for i in range(20)]
    
    elif args.problem == 'facdem_maxopen':
        facdemlow = args.facdemlow
        facdemhigh = args.facdemhigh
        
        
        facdemcaplow = args.facdemcaplow
        facdemcaphigh = args.facdemcaphigh
        print('facdemlow', facdemlow) 
        print('facdemhigh', facdemhigh)
        
        instances += [{'type': 'small', 'path': f"data/instances/facdem_maxopen{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}_{args.facmaxopen}/transfer_100_100_5/instance_{i+1}.lp"} for i in range(20)]
      
  

    elif args.problem == 'setcover_densize':
        instances += [{'type': 'small', 'path': f"data/instances/setcover_densize_{args.density}/test_700r_800c_{args.density}d/instance_{i+1}.lp"} for i in range(20)]
       
           
    else:
        raise NotImplementedError

   
    result_file = str(str(instances[0]).split("/")[2]) + result_file 
    result_file = str(result_file) + '.csv'
    result_file = f"{eval_dir}/{result_file}"

    print('result_file ', result_file)
    
    
    
    branching_policies = []

    # GCNN models
    for model in gcnn_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'gcnn',
                'name': model,
                'seed': seed,
                'parameters': full_epoch_load_path, 
            })


    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")

    ### NUMPY / TORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    for policy in branching_policies:
        if policy['type'] == 'gcnn':
            if policy['name'] not in loaded_models:
                sys.path.insert(0, os.path.abspath(f"models/{policy['name']}"))
                import model
                importlib.reload(model)
                loaded_models[policy['name']] = model.GATPolicy()
                del sys.path[0]
            policy['model'] = loaded_models[policy['name']]


    print("running SCIP...")

    fieldnames = [
        'instance',
        
        'nnodes',
        'walltime',
        'seed',
        'model',
        
        'stime',
        'proctime',
        
        'device',
        'policy',
        
        'type',
        'problem',
        
        'nlps',
        'gap',
        'status',
        'ndomchgs',
        'ncutoffs',
        
    ]
    

    with open(result_file, 'a', newline='') as csvfile:
        
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                torch.manual_seed(policy['seed'])

                m = scip.Model()
                m.setIntParam('display/verblevel', 0)
                m.readProblem(f"{instance['path']}")
                utilities.init_scip_params(m, seed=policy['seed'])
                m.setIntParam('timing/clocktype', 1)  
                m.setRealParam('limits/time', time_limit)

                brancher = PolicyBranching(policy, device)
                m.includeBranchrule(
                    branchrule=brancher,
                    name=f"{policy['type']}:{policy['name']}",
                    desc=f"Custom MLPOpt branching policy.",
                    priority=666666, maxdepth=-1, maxbounddist=1)

                walltime = time.perf_counter()
                proctime = time.process_time()

                m.optimize()

                walltime = time.perf_counter() - walltime
                proctime = time.process_time() - proctime

                stime = m.getSolvingTime()
                nnodes = m.getNNodes()
                nlps = m.getNLPs()
                gap = m.getGap()
                status = m.getStatus()
                ndomchgs = brancher.ndomchgs
                ncutoffs = brancher.ncutoffs

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'gap': gap,
                    'status': status,
                    'ndomchgs': ndomchgs,
                    'ncutoffs': ncutoffs,
                    'walltime': walltime,
                    'proctime': proctime,
                    'problem':args.problem,
                    'device': "CPU" if args.gpu == -1 else "GPU",
                    'model':full_epoch_load_path,
                })
 
                csvfile.flush()
                m.freeProb() 

                print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} ({nnodes+2*(ndomchgs+ncutoffs)}) nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")
