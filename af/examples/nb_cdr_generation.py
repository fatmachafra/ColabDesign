import argparse
parser = argparse.ArgumentParser(description='Running AF-Design for CDR diversification/optimization')
parser.add_argument("-f", '--file', default='/usr/users/fatma.chafra01/ColabDesign/8EE2_antigen_4A_contact.pdb')
parser.add_argument("-o", '--optimizer', help='optimizer type for starting the backpropagation from logit, softmax, one-hot etc.')
parser.add_argument("-g", '--gd', help='optimizer type for the backpropagation (GD_method)')
parser.add_argument("-l", '--logit', default='0')
parser.add_argument("-s", '--soft')
parser.add_argument("-d", '--hard')
parser.add_argument("-r", '--learning_rate', default=0.01)
parser.add_argument("-n", '--num_models', default=1)
parser.add_argument("-w", '--weights', help='text of comma separated weights in the order of con, dgram_cce, exp_res, fape, helix, i_con, i_pae, pae, plddt, rmsd, seq_ent')
args = parser.parse_args()

with open(args.weights, 'r') as file:
  weights_parameters = file.readline().strip().split(',')

print('weights parameters', weights_parameters)

import sys
pdb = (args.file).split('/')[5].split('_')[0]
filename = f"{pdb}_antigen_4A_contact_{args.optimizer}_{args.gd}_{args.logit}_{args.soft}_{args.hard}_{args.learning_rate}_{args.num_models}_weights_{'_'.join(weights_parameters)}_c62.pdb"
# Open a file to write logs
#with open(filename, 'w') as f:
    # Redirect stdout to the file
#    sys.stdout = f


import os
# set working directory
try:
    os.chdir('/usr/users/fatma.chafra01/ColabDesign/af')
    print(os.getcwd())
except Exception as e:
    print(f"Error: {e}")


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import re

from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict
from colabdesign.af.alphafold.common import residue_constants

from IPython.display import HTML
import numpy as np

import time
import tracemalloc

tracemalloc.start()

os.chdir('/usr/users/fatma.chafra01/ColabDesign/af/examples')
print(os.getcwd())

# prep inputs
target_chain = "A" #@param {type:"string"}
target_hotspot = "" #@param {type:"string"}
if target_hotspot == "": target_hotspot = None
# specifies positions in binder (chain B) that should remain fixed during
# redesign

pos = "" #@param {type:"string"}
pos = re.sub("[^0-9,]", "", pos)
if pos == "": pos = None
#@markdown - restrict loss to predefined positions on target (eg. "1-10,12,15")
target_flexible = False #@param {type:"boolean"}
#@markdown - allow backbone of target structure to be flexible

#@markdown ---
#@markdown **binder info**
binder_len = 102 #@param {type:"integer"}
#@markdown - length of binder to hallucination
binder_seq = "" #@param {type:"string"}
binder_seq = re.sub("[^A-Z]", "", binder_seq.upper())
if len(binder_seq) > 0:
  binder_len = len(binder_seq)
else:
  binder_seq = None
#@markdown - if defined, will initialize design with this sequence

binder_chain = "C" #@param {type:"string"}
if binder_chain == "": binder_chain = None
#@markdown - if defined, supervised loss is used (binder_len is ignored)

fix_seq = True #@param {type:"boolean"}
#@markdown - When set to True, it maintains the original sequence at the fixed positions specified by fix_pos.
#@markdown ---
#@markdown **model config**
use_multimer = True #@param {type:"boolean"}
#@markdown - use alphafold-multimer for design
num_recycles = 0 #@param ["0", "1", "3", "6"] {type:"raw"}
num_models = str(args.num_models) #@param ["1", "2", "3", "4", "5", "all"]
num_models = 5 if num_models == "all" else int(num_models)
#@markdown - number of trained models to use during optimization


x = {"pdb_filename":pdb,
     "chain":target_chain,
     "binder_len":binder_len,
     "binder_chain":binder_chain,
     "hotspot":target_hotspot,
     "use_multimer":use_multimer,
     "rm_target_seq":target_flexible,
     "fix_seq": fix_seq}

# x["pdb_filename"] = get_pdb(x["pdb_filename"])
# instead of using the get_pdb function, using the modified pdb file
x["pdb_filename"] = args.file

#if "x_prev" not in dir() or x != x_prev:
clear_mem()
model = mk_afdesign_model(protocol="binder",
                          use_multimer=x["use_multimer"],
                          num_recycles=num_recycles,
                          recycle_mode="sample")
model.prep_inputs(**x,
                  ignore_missing=False)
x_prev = copy_dict(x)
print("target length:", model._target_len)
print("binder length:", model._binder_len)
binder_len = model._binder_len

# print out default model parameters
# model weights
print('default weights', model.opt["weights"].keys())
for keys in model.opt["weights"].keys():
    print(f'{keys}', model.opt["weights"][keys])

# forming the bias matrix to fix the nb core positions
# after modiying the pdb file, the aa seq got truncated so had to change the intervals from (1, 27), (35,53), (59,99), (112,121) to:
fix_pos = [(19, 27), (35,53), (59,99), (112,120)]
seq = 'MAEVQLVESGGGLVQPGGSLRLSCTTSTSLFSITTMGWYRQAPGKQRELVASIKRGGGTNYADSMKGRFTISRDNARNTVFLEMNNLTTEDTAVYYCNAAILAYTGEVTNYWGQGTQVTV'
# make sure to also delete the last part that is not in the structure from the original sequence because it will not be present in the later generated sequence
print('seq length', len(seq))
# instead trying to make a bias matrix as suggested here: https://github.com/sokrypton/ColabDesign/issues/107
print(model._binder_len)
bias = np.zeros((model._binder_len,20))
for item in fix_pos:
  start = item[0] -1
  end = item[1] -1
  print(start, end)
  while start <= end:
    aa = seq[start]
    print(start, aa)
    start += 1
    # because the index changed once the pdb file was truncated
    bias[start-19,residue_constants.restype_order[str(aa)]] = 1e8
    print(f'bias added to:{start-19} as {aa}')
print('bias matrix', bias)

# check whether the bias matrix makes sense by randomly printing out a row that has to contain a serine (pos 19 according to prev numbering, now pos 19 - 19 so 0)
# I don't know the order of the one hot but it is not exactly the alphabetically ordered classical one hot
def find_rows_without_value(arr, value):
    # Check each row for the presence of the value
    rows_with_value = np.any(np.isclose(arr, value), axis=1)

    # Get the indices of rows that don't have the value
    rows_without_value = np.where(~rows_with_value)[0]

    return rows_without_value

print("Rows without 1.0e+08 in bias matrix:", find_rows_without_value(bias, 1.0e+08))

# restart model before running the optimizer because without it will be using the modified weights from the previous round (as the model is not created again)
model.restart(seq=binder_seq)
model.set_seq(bias=bias)
# i_pae and i_con 2.0, rest 1.0
model.opt["weights"].update({"i_con": 2.0, "i_pae": 2.0, "con": 0.5, "dgram_cce": 10.0, "exp_res": 1.0, "fape": 0.1, "pae": 10.0, "plddt": 10.0, "rmsd": 20.0, "seq_ent": 1.0})
print('----------')
weights_list = []
for keys in model.opt["weights"].keys():
  print(f'{keys}', model.opt["weights"][keys])
  weights_list.append(keys)
print('----------')
print(model.opt.keys())
# bias is not a key of model.opt so don't know how to access it
print('bias matrix added to model', model._inputs["bias"])
# has the bias matrix associated with the model!

#@title **run AfDesign**
from scipy.special import softmax

optimizer = args.optimizer #@param ["pssm_semigreedy", "3stage", "semigreedy", "pssm", "logits", "soft", "hard"]
#@markdown - `pssm_semigreedy` - uses the designed PSSM to bias semigreedy opt. (Recommended)
#@markdown - `3stage` - gradient based optimization (GD) (logits → soft → hard)
#@markdown - `pssm` - GD optimize (logits → soft) to get a sequence profile (PSSM).
#@markdown - `semigreedy` - tries X random mutations, accepts those that decrease loss
#@markdown - `logits` - GD optimize logits inputs (continious)
#@markdown - `soft` - GD optimize softmax(logits) inputs (probabilities)
#@markdown - `hard` - GD optimize one_hot(logits) inputs (discrete)

#@markdown WARNING: The output sequence from `pssm`,`logits`,`soft` is not one_hot. To get a valid sequence use the other optimizers, or redesign the output backbone with another protocol like ProteinMPNN.

#@markdown ----
#@markdown #### advanced GD settings
GD_method = args.gd #@param ["adabelief", "adafactor", "adagrad", "adam", "adamw", "fromage", "lamb", "lars", "noisy_sgd", "dpsgd", "radam", "rmsprop", "sgd", "sm3", "yogi"]
learning_rate = float(args.learning_rate) #@param {type:"raw"}
norm_seq_grad = True #@param {type:"boolean"}
dropout = True #@param {type:"boolean"}

# optimizer hard - soft iteration numbers
logit = int(args.logit) #only used for the pssm and 3 stage optimizers
soft = int(args.soft)
hard = int(args.hard)

# initializing the backpropagation optimizer
model.set_optimizer(optimizer=GD_method,
                    learning_rate=learning_rate,
                    norm_seq_grad=norm_seq_grad)
models = model._model_names[:num_models]

flags = {"num_recycles":num_recycles,
         "models":models,
         "dropout":dropout}

# calling the respective function to specify from which type of output the backpropagation is done on (logits, softmax, PSSM, hard/one-hot etc.)
print('running optimizer method:', optimizer)
start_time = time.time()
print('start', start_time)
if optimizer == "3stage":
  model.design_3stage(120, 60, 10, **flags)
  pssm = softmax(model._tmp["seq_logits"],-1)

if optimizer == "pssm_semigreedy":
  #model.design_pssm_semigreedy(120, 32, **flags)
  model.design_pssm_semigreedy(soft, hard, verbose=1, **flags)
  pssm = softmax(model._tmp["seq_logits"],1)

if optimizer == "semigreedy":
  model.design_pssm_semigreedy(0, hard, **flags)
  pssm = None

if optimizer == "pssm":
  model.design_logits(logit, e_soft=1.0, num_models=1, ramp_recycles=True, **flags)
  model.design_soft(soft, num_models=1, **flags)
  flags.update({"dropout":False,"save_best":True})
  model.design_soft(10, num_models=num_models, **flags)
  pssm = softmax(model.aux["seq"]["logits"],-1)

O = {"logits":model.design_logits,
     "soft":model.design_soft,
     "hard":model.design_hard}

if optimizer in O:
  O[optimizer](120, num_models=1, ramp_recycles=True, **flags)
  flags.update({"dropout":False,"save_best":True})
  O[optimizer](10, num_models=num_models, **flags)
  pssm = softmax(model.aux["seq"]["logits"],-1)

end_time = time.time()
print('end', end_time)
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.6f} seconds")

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 10**6:.2f} MB")
print(f"Peak memory usage: {peak / 10**6:.2f} MB")
tracemalloc.stop()

print('optimizer done')
# output_file = f'{filename.split("{{.}}")[0]}.pdb'
model.save_pdb(filename)
print('model saved in file:', filename)

# html_output = HTML(model.animate(dpi=100))
# with open('filename.html', 'w') as file:
    # file.write(html_output.data)