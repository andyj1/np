# 고영최적화

고영 Self-Alignment 최적화 과제 (5차년도)

## Bayesian Optimization

#### Surrogate Model
- BoTorch-based Bayesian Optimization GPR surrogate model
- Neural Process-based surrogate model

#### Acquisition Function
- UpperConfidenceBound
- Expected Improvement
- ...

## How to run
```python
python main.py
```

## Configuration
> - num_iter: 100 // number of iterations of optimization to do
> - num_samples: 50 // number of samples for input data
> - mu: 0 // input data normal distribution parameter
> - sigma: 5 // input data normal distribution parameter
> - dist_mu: 10 // distance offset normal distribution parameter
> - dist_sigma: 10 // distance offset normal distribution parameter
> - angle_mu: 0 // angle offset normal distribution parameter
> - angle_sigma: 40 // angle offset normal distribution parameter
> - num_context: 10 // number of context points for NP model
> - hidden_dim: 10 // number of hidden dimensions for NP model
> - decoder_dim: 15 // number of decoder dimension for NP model
> - z_samples: 20 // number of latent variable samples for NP Model


<!-- ## Contribution
## License -->