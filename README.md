# 고영최적화

고영 Self-Alignment Mounter Feedforward Model 최적화 과제 (5차년도)

## Bayesian Optimization

#### Surrogate Model
- BoTorch (GPyTorch)-based Bayesian Optimization GPR surrogate model
- Neural Process surrogate model

#### Acquisition Function
- UpperConfidenceBound
- Expected Improvement
- q-EI
- Probability of Improvement

## How to run
> for configurations, refer to *config.yml*
> for loading models, refer to */ckpts* directory
```python
python main.py # run with MOM4 data
python main.py --toy # run with toy data
python main.py --toy --load PATH # load model and optimizer from checkpoints
```

<!-- ## Contribution
## License -->