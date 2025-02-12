import os
import sys

# Add the path to your colabdesign environment's site-packages
sys.path.append('/usr/users/fatma.chafra01/miniforge3/envs/colabdesign/lib/python3.10/site-packages')
print(sys.executable)
print('----------')
import argparse
parser = argparse.ArgumentParser(description='Running AF-Design for CDR diversification/optimization')
parser.add_argument("-f", '--file')
parser.add_argument("-id", '--pdb_id', help='pdb_id string in lowercase')
parser.add_argument("-dir", "--output_dir", default = '/usr/users/fatma.chafra01/ColabDesign/af/examples', help="absolute path that would contain the pdb_id/pdb_id_.pdb file")
parser.add_argument("-ad", '--additional_description', help='description like 4A_contact with _ instead of space')
parser.add_argument("-o", '--optimizer', help='optimizer type for starting the backpropagation from ["pssm_semigreedy", "3stage", "semigreedy", "pssm", "logits", "soft", "hard"]')
parser.add_argument("-g", '--gd', help='optimizer type for the backpropagation (GD_method)')
parser.add_argument("-l", '--logit', default='0')
parser.add_argument("-s", '--soft')
parser.add_argument("-d", '--hard')
parser.add_argument("-r", '--learning_rate', default=0.01)
parser.add_argument("-n", '--num_models', default=1)
parser.add_argument("-w", '--weights', help='text of comma separated weights in the order of con, dgram_cce, exp_res, fape, helix, i_con, i_pae, pae, plddt, rmsd, seq_ent')
parser.add_argument("-c", '--recycles', default=0)
parser.add_argument("-e", '--cores', default=38)
parser.add_argument("-t", '--use_templates', default=True)
parser.add_argument("-x", '--rm_template_ic', default=False)
parser.add_argument("-b", '--bias', default=True)
parser.add_argument("-m", '--bias_matrix', help='bias matrix to be used if bias=True. Naming convention: <abspath>/bias_matrix/bias_matrix_pdb.csv')
parser.add_argument("-a", '--antigen_target', help='chain name of the antigen target, capitalized')
parser.add_argument("-nb", '--nb_binder', help='chain name of the nb binder, capitalized')
parser.add_argument("-q", '--nb_binder_seq', help='string of the nb sequence that is associated with a structure in the file')
args = parser.parse_args()
print(args)
print('----------')
order_weights = ['con', 'dgram_cce', 'exp_res', 'fape', 'helix', 'i_con', 'i_pae', 'pae', 'plddt', 'rmsd', 'seq_ent']
with open(args.weights, 'r') as file:
  weights_list = file.readline().strip().split(',')

print('weights list:', weights_list)
weights_dict = {k : float(val) for k, val in zip(order_weights, weights_list)}
print('----------')
pdb = args.pdb_id
weights_rounded = [str(round(float(weight),3)) for weight in weights_list]
output_dir = args.output_dir
bias_matrix_name = args.bias_matrix.split('bias_matrix/')[-1].split('.')[0]
filename = f"{output_dir}/{pdb}/{pdb}_{args.additional_description}_{args.optimizer}_{args.gd}_{args.logit}_{args.soft}_{args.hard}_{args.learning_rate}_models_{args.num_models}_weights_{'_'.join(weights_rounded)}_c{args.cores}_use_templates_{args.use_templates}_rm_template_ic_{args.rm_template_ic}_bias_{args.bias}_bias_matrix_{bias_matrix_name}_num_recycles_{args.recycles}.pdb"
print('PDB filename of final output:', filename)
if not os.path.isdir(f"{output_dir}/{pdb}"):
    raise FileNotFoundError(f"The directory '{output_dir}/{pdb}' does not exist.")
print('----------')
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
#from colabdesign.af.alphafold.common import residue_constants

#from IPython.display import HTML
import numpy as np

import time
import tracemalloc

tracemalloc.start()

os.chdir('/usr/users/fatma.chafra01/ColabDesign/af/examples')
print(os.getcwd())

# prep inputs
# target_chain = "A" #@param {type:"string"}
target_chain = args.antigen_target #@param {type:"string"}
target_hotspot = "" #@param {type:"string"}
if target_hotspot == "": target_hotspot = None
# specifies positions in binder (chain B) that should remain fixed during redesign

pos = "" #@param {type:"string"}
pos = re.sub("[^0-9,]", "", pos)
if pos == "": pos = None
#@markdown - restrict loss to predefined positions on target (eg. "1-10,12,15")
target_flexible = False #@param {type:"boolean"}
#@markdown - allow backbone of target structure to be flexible

#@markdown ---
#@markdown **binder info**
binder_len = 102 #@param {type:"integer"} # ignored if binder_chain is defined (in ColabDesign/colabdesign/af/prep.py)
#@markdown - length of binder to hallucination
binder_seq = args.nb_binder_seq #@param {type:"string"}
binder_seq = re.sub("[^A-Z]", "", binder_seq.upper())
if len(binder_seq) > 0:
  binder_len = len(binder_seq)
else:
  binder_seq = None
#@markdown - if defined, will initialize design with this sequence

# binder_chain = "C" #@param {type:"string"}
binder_chain = args.nb_binder #@param {type:"string"}
if binder_chain == "": binder_chain = None
#@markdown - if defined, supervised loss is used (binder_len is ignored)

fix_seq = True #@param {type:"boolean"}
#@markdown - When set to True, it maintains the original sequence at the fixed positions specified by fix_pos.
#@markdown ---
#@markdown **model config**
use_multimer = True #@param {type:"boolean"}
#@markdown - use alphafold-multimer for design
num_recycles = int(args.recycles) #@param ["0", "1", "3", "6"] {type:"raw"}
num_models = int(args.num_models) #@param ["1", "2", "3", "4", "5", "all"]
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
use_templates = args.use_templates == 'True'
print('use_templates:', use_templates)
rm_template_ic = args.rm_template_ic == 'True'
print('rm_template_ic:', rm_template_ic)
print('----------')

clear_mem()
model = mk_afdesign_model(protocol="binder",
                          use_multimer=x["use_multimer"],
                          num_recycles=num_recycles,
                          recycle_mode="sample", use_templates=use_templates)
model.prep_inputs(**x,
                  ignore_missing=False, rm_template_ic=rm_template_ic, use_binder_template=True)
x_prev = copy_dict(x)
print("target length:", model._target_len)
print("binder length:", model._binder_len)
binder_len = model._binder_len
print('----------')
# print out default model parameters
# model weights
print('default weights', model.opt["weights"].keys())
for keys in model.opt["weights"].keys():
    print(f'{keys}', model.opt["weights"][keys])
print('----------')

# this is not supposed to work but anyways
print('Former seed/key:', model.key())
model.set_seed(123324)
print('Set seed/key:', model.key())
print('----------')

# restart model before running the optimizer because without it will be using the modified weights from the previous round (reset_opt = False)
print('setting seed with model.restart(seed=123324). the previous setting of seed usually does not work!')
model.restart(seq=binder_seq, seed=123324, reset_opt=False)
print('----------')
print('type args.bias:', type(args.bias))
print('args.bias:', args.bias)
print('args.bias == True', args.bias == True)
print('args.bias == True as string', args.bias == 'True')
print('----------')
if args.bias == 'True': # because the system arguments are considered as strings so saying args.bias == True leads to the evaluation of else all the time!! 
  print('bias = True continuing with a bias matrix')
  try:
    bias = np.loadtxt(args.bias_matrix, delimiter=',')
    print("Shape of the bias matrix:", bias.shape)
    print('The bias matrix:\n', bias)
    if bias.shape == (model._binder_len,20):
      print('Bias matrix shape is appropriate')
      model.set_seq(bias=bias)
    else:
      raise ValueError(f"Bias matrix shape is incorrect. Expected ({model._binder_len}, 20), but got {bias.shape}")
  except:
    print('Bias matrix not found')
else:
  print('bias = False continuing without bias matrix')
print('----------')
# set weights
# in the order of con, dgram_cce, exp_res, fape, helix, i_con, i_pae, pae, plddt, rmsd, seq_ent
model.opt["weights"].update(weights_dict)
print('----------')
print('New weights:')
for keys in model.opt["weights"].keys():
  print(f'{keys}', model.opt["weights"][keys])
print('----------')
print(model.opt.keys())

# bias is not a key of model.opt so access it from model._inputs
print('bias matrix of model', model._inputs["bias"])
# has the bias matrix associated with the model!
print('----------')

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
  #model.design_3stage(120, 60, 10, **flags)
  model.design_3stage(logit, soft, hard, **flags)
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