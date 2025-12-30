# European Call Option Pricing Framework

A Python-based framework for pricing **European call options** using **risk-neutral valuation** and **Monte Carlo simulations**. This project demonstrates concepts from **quantitative finance**, **stochastic calculus**, and **numerical simulation** in a practical, object-oriented Python implementation.

## Features

- **Random Variable and Probability Measure abstractions**: Flexible classes to model stochastic processes and probability distributions.  
- **Normally distributed random variables**: Generate random variables with specified mean and variance for simulation.  
- **Risk-neutral probability measure**: Implements Girsanov’s theorem to adjust probabilities for risk-neutral valuation.  
- **Option pricing**: Compute European call option prices as the **discounted expectation of payoffs** under the risk-neutral measure.  
- **Validation against Black–Scholes formula**: Ensures Monte Carlo simulations match analytical solutions.  
- **Unit tests**: Comprehensive tests for correctness, reproducibility, and robustness of the framework.  


Usage
```
from pricing_framework import european_call_payoff_rv, price_option

# Parameters
S0 = 100.0    # Initial stock price
K = 100.0     # Strike price
mu = 0.05     # Drift
sigma = 0.2   # Volatility
r = 0.05      # Risk-free rate

# Create payoff random variable
payoff_rv = european_call_payoff_rv(S0, mu, sigma, K)

# Price the option using Monte Carlo simulation
option_price = price_option(payoff_rv, mu, sigma, r)
print(f"European call option price: {option_price:.2f}")
```
