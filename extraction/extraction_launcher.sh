#!/bin/bash

unset LD_PRELOAD
source activate $CDE_ENV
NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=$NODES*2

if [ $CDE_USE_WANDB = 1 ]; then
    echo "Logging in to wandb..."
    wandb login $WANDB_API_KEY
fi

mpirun -f ${COBALT_NODEFILE} -n ${PROCS} -genvall bash e2e_workflow/extraction/single_mpi_thread.sh


# Parameters to pass in:
# BERT_MODEL_NAME
# DOCUMENT_DIR
# OUTPUT_DIR
# CDE_ENV
# EXTRACTION_PYTHON_SCRIPT
# HF_HOME if using huggingface
# WANDB_API_KEY WandB API key
# CDE_USE_WANDB bool
# WANDB_PROJECT
