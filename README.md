# UL-NET

## This is the CS249 UL-Net Project Repo
Author: Zijie Huang, Roshni Iyer, Alex Wang, Derek Xu

### Dataset (CN15k)
Uncertain Knowledge Graph from ConceptNet, containing 15,000 entities, 241,158 uncertain facts, 32 relations.
Splitted into train.tsv, val.tsv, and test.tsv

The original data can be found here at: https://drive.google.com/file/d/1UJQ8hnqPGv1O9pYglfNF5lY_sgDQkleS/view?usp=sharing
* Coutesy to UKGE authors

## Run the Experiment
    python main.py --preprocess
   
The above command will preprocess the dataset and start the training. It will also evaluate the model at the end.
You only have to run with --preprocess flag once. The preprocessed data will be saved in ```save``` folder. Afterward, 
you can skip the preprocessing step by calling without the flag: (various flags can be looked up in main.py)

    python main.py
   
During the experiment, there will be folders generated, namely the ```save``` and ```record``` folder. The ```save``` folder contains the preprocessed data for later runs, and the ```record``` folder contains the experiment results of the model, timestamped. Evaluation results are saved in ```.json``` format in the timestamped folder.


