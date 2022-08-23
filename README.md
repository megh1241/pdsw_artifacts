# Artifact for PDSW submission: Towards Scalable Learning Model Repositories with Fine-Grain Access and Persistency Support


## Structure
### experiments/:
 Experiments and scripts (including launch .qsub scripts)

### experiments/cpp-store/:
 Source code(C++) for model repository. 

### tmci/
Fork of https://xgitlab.cels.anl.gov/sds/tmci to capturing pointers to tensorflow tensors directly using a custom tensorflow C++ operator. This enables zero-copy transfer of tensors. This needs to be installed separately.

### plot_scripts/
Scripts to generate plots in the paper

## Installation

### tmci
```
cd tmci
python setup.py install --user
```


### cpp-store
```bash
cd pdsw_artifacts/experiments/cpp-store
cmake .
make
```

## Running Benchmarks


## Generating Plots
