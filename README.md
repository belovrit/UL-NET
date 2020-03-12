# UL-NET

## This is the CS249 Probabilistic Graphical Model Project Repo
Author: Zijie Huang, Roshni Iyer, Alex Wang, Derek Xu

## Run the Experiment
    python main.py --preprocess
   
The above command will preprocess the dataset and start the training. It will also evaluate the model at the end.
You only have to run with --preprocess flag once. The preprocessed data will be saved in save folder. Afterward, 
you can skipp the preprocessing step by calling without the flag: (various flags can be looked up in main.py)

    python main.py
   
During the experiment, there will be folders generated, namely the save and record folder. The save folder contains the preprocessed data for later runs, and the record folder contains the experiment results of the model, timestamped. Evaluation result is saved in .json format in the timestamped folder.
