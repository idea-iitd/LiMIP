# LiMIP: Lifelong Learning to Solve Mixed Integer Programs

## Authors: Sahil Manchanda and Sayan Ranu

### Conference: 37th  Conference on Artificial Intelligence : AAAI 2023

## Dependencies
To install SCIP and other dependencies, follow the instructions of https://github.com/ds4dm/learn2branch/blob/master/INSTALL.md

 We use the following python packages for python version 3.6.13.
```sh
torch==1.9.0+cu111
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torchaudio==0.9.0
torchvision==0.10.0+cu111
scipy==1.5.2
numpy==1.18.1
networkx==2.4
Cython==0.29.13
PySCIPOpt==2.1.5
scikit-learn==0.20.2
```

## Generating training instances
* To generate dataset for Set Cover(SC) problem with edge density 0.05 (SC_0.05) run:
* ` python scripts/Cont_generate_instances.py setcover_densize --density 0.05`

Similarly can be done for other set cover datasets
* ` python scripts/Cont_generate_instances.py setcover_densize --density 0.075`
* ` python scripts/Cont_generate_instances.py setcover_densize --density 0.1`
* ` python scripts/Cont_generate_instances.py setcover_densize --density 0.125`
* ` python scripts/Cont_generate_instances.py setcover_densize --density 0.15`
* ` python scripts/Cont_generate_instances.py setcover_densize --density 0.2`



For Indset(IS) with affinity 4 and number of nodes 750 use
* `python scripts/Cont_generate_instances.py indsetnewba --affinity 4 --indnodes 750` 

For other affinities and size for IndSet:
* `python scripts/Cont_generate_instances.py indsetnewba --affinity 4 --indnodes 500` 
* `python scripts/Cont_generate_instances.py indsetnewba --affinity 4 --indnodes 450`

* `python scripts/Cont_generate_instances.py indsetnewba --affinity 5 --indnodes 450` 
* `python scripts/Cont_generate_instances.py indsetnewba --affinity 5 --indnodes 400` 
* `python scripts/Cont_generate_instances.py indsetnewba --affinity 5 --indnodes 350` 




## Generating training samples

To generate training sample for setcover problem with edge-density 0.05 use

* ` python Cont_generate_dataset.py setcover_densize --density 0.05 -j 20`

Similarly for other instances in set cover( with different densities)
* ` python Cont_generate_dataset.py setcover_densize --density 0.05 -j 20`
* ` python Cont_generate_dataset.py setcover_densize --density 0.075 -j 20`
* ` python Cont_generate_dataset.py setcover_densize --density 0.1 -j 20`
* ` python Cont_generate_dataset.py setcover_densize --density 0.125 -j 20`
* ` python Cont_generate_dataset.py setcover_densize --density 0.15 -j 20`
* ` python Cont_generate_dataset.py setcover_densize --density 0.2 -j 20`


For other problems such as independent set(with different affinities and number of nodes), use below commands:
* ` python Cont_generate_dataset.py indsetnewba --affinity 4  --indnodes 750 -j 20`
* ` python Cont_generate_dataset.py indsetnewba --affinity 4  --indnodes 500 -j 20`
* ` python Cont_generate_dataset.py indsetnewba --affinity 4  --indnodes 450 -j 20`
* ` python Cont_generate_dataset.py indsetnewba --affinity 5  --indnodes 450 -j 20`
* ` python Cont_generate_dataset.py indsetnewba --affinity 5  --indnodes 400 -j 20`
* ` python Cont_generate_dataset.py indsetnewba --affinity 5  --indnodes 350 -j 20`



## Training
To train for sequence [SetCover_0.05,SetCover_0.075, SetCover_0.1,   SetCover_0.125, SetCover_0.15, SetCover_0.2]


` python -u train.py --g 1  --prob_seq setcover_densize_0.05-setcover_densize_0.075-setcover_densize_0.1-setcover_densize_0.125-setcover_densize_0.15-setcover_densize_0.2 `



To train for Indset [indsetnewba_4_750, indsetnewba_4_500,indsetnewba_4_450, indsetnewba_5_450, indsetnewba_5_400, indsetnewba_5_350]

` python -u train.py --g 1  --prob_seq indsetnewba_4_750-indsetnewba_4_500-indsetnewba_4_450-indsetnewba_5_450-indsetnewba_5_400-indsetnewba_5_350 `

The prob_seq consists of different tasks separated by `-`

# Evaluation

To evaluate the above continually trained model on test instances of setcover with density 0.05, run the following

* ` python -u eval.py setcover_densize --g 1  --path_load trained_models/MODEL_setcover_densize_0.05_setcover_densize_0.1_setcover_densize_0.15_setcover_densize_0.2/GAT_baseline_torch/0/ --density 0.05 --epoch_load checkpoint.pkl `



To evaluate the above continually trained model on test instances of indset with affinity 4 and num nodes 750, run the following

* ` python -u eval.py indsetnewba --g 1  --path_load trained_models/MODEL_indsetnewba_4_750-indsetnewba_4_500-indsetnewba_4_450-indsetnewba_5_450-indsetnewba_5_400-indsetnewba_5_350/GAT_baseline_torch/0/ --affinity 4 --indnodes 750 --epoch_load checkpoint.pkl `
