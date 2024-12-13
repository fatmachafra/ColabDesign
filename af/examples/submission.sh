#!/bin/bash

#SBATCH --output=cdr_gen_c1_1_1.out
#SBATCH --error=cdr_gen_c1_1_1.err
#SBATCH -c 1
#SBATCH -p soeding
#SBATCH --mem=32GB
#SBATCH -t 00-01:00:00

python3 -u /usr/users/fatma.chafra01/ColabDesign/af/examples/nb_cdr_generation.py -o pssm_semigreedy -g sgd -s 1 -d 1 -w weights_test.csv
