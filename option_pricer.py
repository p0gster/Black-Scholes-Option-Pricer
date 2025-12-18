# Sample space is the set of integers from 0 to sample_space_size - 1

# Define the size of the sample space
sample_space_size = 10  # You can change this number
terminal_time=10


from typing import Callable, List
import unittest
import math


# Random Variable Abstract Data Type
class RandomVariable:
    def __init__(self, func: Callable[[int], float]):
        """Construct a random variable from a function from the sample space to real numbers."""
        if not callable(func):
            raise TypeError("Function must be callable")
        self.func = func

    def evaluate(self, outcome: int) -> float:
        return self.func(outcome)

    def values(self, space: List[int]) -> List[float]:
        return [self.func(o) for o in space]

# Procedure: normally distributed random variable
import numpy as np

def normally_distributed_random_variable(mean: float, variance: float) -> RandomVariable:
    """Return a RandomVariable representing a normally distributed real value.
    The RNG is seeded so the value is reproducible for each outcome.
    Each outcome gets its own sampled value.
    """
    def seed_to_normal_rv_value(seed):
      rng = np.random.default_rng(seed)
      std = variance ** 0.5
      # Precompute values for the discrete sample space
      return rng.normal(mean, std)

    return RandomVariable(lambda sample: seed_to_normal_rv_value(sample))

# Probability Measure Abstract Data Type
class ProbabilityMeasure:
    def __init__(self, func: Callable[[int], float]):
        """Construct a probability measure from the sample space to real numbers (probabilities). 
        Note: sum of probabilities must be one, this code does not check this."""
        if not callable(func):
            raise TypeError("Function must be callable")
        self.func = func

    def probability(self, outcome: int) -> float:
        p = self.func(outcome)
        if not (0 <= p <= 1):
            raise ValueError(f"Probability must be between 0 and 1, got {p}")
        return p

    def probabilities(self, space: List[int]) -> List[float]:
        return [self.probability(o) for o in space]

def create_risk_neutral_measure(mu: float, sigma: float, r: float) -> ProbabilityMeasure:
    """
    Create a risk-neutral probability measure using Girsanov's theorem.

    Inputs:
    - mu: drift of the underlying
    - sigma: volatility
    - r: risk-free interest rate

    Returns:
    - ProbabilityMeasure instance representing the risk-neutral measure
    """
    global terminal_time, sample_space_size

    # Create Wiener random variable with mean 0 and variance terminal_time
    wiener_rv = normally_distributed_random_variable(0.0, terminal_time)

    # Market price of risk
    theta = (mu - r) / sigma

    # Define the risk-neutral probability function
    def p_star(omega: int) -> float:
        return (1 / sample_space_size) * math.exp(
            -theta * wiener_rv.evaluate(omega)
            - 0.5 * theta**2 * terminal_time
        )

    return ProbabilityMeasure(p_star)




# Expectation function
def expectation(rv: RandomVariable, pm: ProbabilityMeasure) -> float:
    """Return the expectation of the random variable with respect to the probability measure.
    Computed as the sum over the sample space of rv(outcome) * pm(outcome).
    """
    return sum(rv.evaluate(o) * pm.probability(o) for o in range(0,sample_space_size))



# Unit tests for RandomVariable class
class TestRandomVariable(unittest.TestCase):
    def test_valid_function(self):
        rv = RandomVariable(lambda x: x + 1)
        self.assertEqual(rv.evaluate(2), 3)
        self.assertEqual(rv.values([1,2,3]), [2,3,4])

    def test_non_callable_raises(self):
        with self.assertRaises(TypeError):
            RandomVariable(42)  # Not a function

# Unit tests for ProbabilityMeasure class
class TestProbabilityMeasure(unittest.TestCase):
    def test_valid_probability(self):
        pm = ProbabilityMeasure(lambda x: 0.1 * x)
        self.assertAlmostEqual(pm.probability(2), 0.2)

    def test_probability_out_of_bounds(self):
        pm = ProbabilityMeasure(lambda x: x)  # Can exceed 1
        with self.assertRaises(ValueError):
            pm.probability(2)  # 2 is not a valid probability

    def test_non_callable_raises(self):
        with self.assertRaises(TypeError):
            ProbabilityMeasure(42)  # Not a function

# Unit tests for Normally Distributed Random Variable
class TestNormalRandomVariable(unittest.TestCase):
    def test_normal_rv_mean_std(self):
        mean = 0.0
        variance = 4.0
        rv = normally_distributed_random_variable(mean, variance)
        values = rv.values(list(range(1000)))

        # Sample mean and std from discrete sample
        sample_mean = sum(values) / len(values)
        sample_variance = sum((x - sample_mean)**2 for x in values) / len(values)
        sample_std = sample_variance ** 0.5

        # Check approximate mean and std
        self.assertAlmostEqual(sample_mean, mean, delta=0.1)  # loose bound due to small sample size
        self.assertAlmostEqual(sample_std, variance**0.5, delta=0.1)

# Unit tests for Expectation function
class TestExpectation(unittest.TestCase):
    def test_expectation_uniform(self):
        # Uniform probability measure on sample space
        pm = ProbabilityMeasure(lambda x: 1 / sample_space_size)
        rv = RandomVariable(lambda x: x)
        expected = sum(x * (1 / sample_space_size) for x in range(sample_space_size))
        self.assertAlmostEqual(expectation(rv, pm), expected)


class TestRiskNeutralMeasure(unittest.TestCase):
    def test_probabilities_non_negative(self):
        """All risk-neutral probabilities should be non-negative."""
        mu = 0.1
        sigma = 0.2
        r = 0.05

        pm_star = create_risk_neutral_measure(mu, sigma, r)
        probs = pm_star.probabilities(list(range(sample_space_size)))

        for p in probs:
            self.assertGreaterEqual(p, 0.0)

    def test_approximate_normalization(self):
        """Sum of probabilities should be approximately 1 (discrete approximation)."""
        mu = 0.1
        sigma = 0.3
        r = 0.05

        pm_star = create_risk_neutral_measure(mu, sigma, r)
        total_mass = sum(pm_star.probabilities(list(range(sample_space_size))))

        # In a finite discrete approximation, allow a loose tolerance
        self.assertAlmostEqual(total_mass, 1.0, delta=0.5)

    def test_zero_market_price_of_risk(self):
        """If mu == r, the risk-neutral measure should reduce to the uniform measure."""
        mu = 0.05
        r = 0.05
        sigma = 0.2

        pm_star = create_risk_neutral_measure(mu, sigma, r)
        uniform = 1 / sample_space_size

        probs = pm_star.probabilities(list(range(sample_space_size)))
        for p in probs:
            self.assertAlmostEqual(p, uniform, delta=1e-6)


# To ensure unittest discovers the tests in interactive/run environments
def run_tests():
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestRandomVariable)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestProbabilityMeasure)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestExpectation)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestNormalRandomVariable)
    suite5 = unittest.TestLoader().loadTestsFromTestCase(TestRiskNeutralMeasure)

    all_tests = unittest.TestSuite([suite1,suite2,suite3,suite4,suite5])
    unittest.TextTestRunner(verbosity=2).run(all_tests)

# Run tests only if script is executed directly
run_tests()
