# 고영최적화
고영 Self-Alignment Mounter Feedforward Model 최적화 과제 (5차년도)

### Data
![data](doc/data_demo.png)

### Variables
![variables](doc/data_var.png)

### Optimization pipeline
![optimization](doc/pipeline.png)

## Bayesian Optimization

#### Surrogate Model
- BoTorch (GPyTorch)-based Bayesian Optimization GPR surrogate model
- Neural Process surrogate model

#### Acquisition Function
- UpperConfidenceBound
- Expected Improvement
- q-EI

## Procedure
> run Makefile:
> ```bash
> make
> ```
1. *imputation.py* handles imputation for three missing rows
2. *multirf.py* prepares models for reflow oven simulation (under *reflow_oven*)
3. *main.py* runs the process

> Other
> *config.yml" contains configuration parameters
> */ckpts* contains checkpoints for training SingleTaskGP model
```python
python main.py # run with MOM4 data
python main.py --toy # run with toy data
python main.py --toy --load PATH # load model and optimizer from checkpoints
```

<!-- ## Contribution
## License -->