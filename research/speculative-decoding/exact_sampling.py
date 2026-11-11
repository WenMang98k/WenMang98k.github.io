"""Original educational implementation by WenMang; no upstream source is vendored.
Finite-vocabulary, one-position speculative sampling. No model or GPU required.
Run: python3 exact_sampling.py
"""
import math
import random


def validate_distribution(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Expected a finite, nonnegative distribution')
    if not math.isclose(sum(values), 1.0, abs_tol=1e-10):
        raise ValueError('Distribution must sum to one')


def sample_index(probabilities, rng):
    u = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probabilities):
        cumulative += p
        if u < cumulative:
            return i
    return len(probabilities) - 1


def speculative_sample_one(target, proposal, rng):
    """Return a token index with the target distribution; q is the ACTUAL proposal."""
    validate_distribution(target)
    validate_distribution(proposal)
    if len(target) != len(proposal):
        raise ValueError('Vocabulary sizes differ')
    y = sample_index(proposal, rng)
    # A q-zero event cannot be drawn by sample_index, including at rng.random()==0.
    if proposal[y] <= 0:
        raise ArithmeticError('Sampled an impossible proposal token')
    alpha = min(1.0, target[y] / proposal[y])
    if rng.random() < alpha:
        return y
    residual = [max(0.0, p - q) for p, q in zip(target, proposal)]
    mass = sum(residual)
    if mass <= 0:
        raise ArithmeticError('Rejected with no residual probability mass')
    return sample_index([p / mass for p in residual], rng)


if __name__ == '__main__':
    target, proposal = [0.7, 0.3], [0.4, 0.6]
    rng = random.Random(20261111)
    counts = [0, 0]
    for _ in range(100000):
        counts[speculative_sample_one(target, proposal, rng)] += 1
    observed = [c / sum(counts) for c in counts]
    assert max(abs(a - b) for a, b in zip(observed, target)) < .01
    for p, q in [([1., 0.], [0., 1.]), ([.5, .5], [.5, .5])]:
        for _ in range(1000):
            token = speculative_sample_one(p, q, rng)
            assert p[token] > 0
    print('Target:', target, 'Observed:', observed)
    print('PASS: Monte Carlo example and zero-support/equal-distribution edge cases')
