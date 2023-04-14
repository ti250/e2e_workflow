unset LD_PRELOAD
source activate $CDE_ENV

export USE_MPI=1

python3 $EXTRACTION_PYTHON_SCRIPT
