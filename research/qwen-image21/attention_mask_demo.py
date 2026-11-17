"""CPU-only teaching example: block-causal visibility and prefix independence.

Uses Python standard library only. This is not Qwen inference or a model
quality/performance test; no checkpoint, torch, GPU, RoPE or learned modulation.
"""
import math

LABELS = ['T0', 'T1', 'R0a', 'R0b', 'T2', 'Xa', 'Xb']
IMAGE_IDS = [-1, -1, 0, 0, -1, 1, 1]
PREFIX = 5


def allowed(q, k):
    same_image = IMAGE_IDS[q] >= 0 and IMAGE_IDS[q] == IMAGE_IDS[k]
    return q >= k or same_image


def attention_layer(states, use_mask):
    # Toy identity Q/K/V projections with a tokenwise residual.
    result = []
    for q, query in enumerate(states):
        indices = [k for k in range(len(states)) if not use_mask or allowed(q, k)]
        scores = [sum(a * b for a, b in zip(query, states[k])) / math.sqrt(2) for k in indices]
        peak = max(scores)
        weights = [math.exp(s - peak) for s in scores]
        total = sum(weights)
        result.append([query[d] + sum(w * states[k][d] for w, k in zip(weights, indices)) / total for d in range(2)])
    return result


def forward(target, use_mask):
    h = [[0.1, 0.2], [0.3, -0.1], [0.6, 0.4], [-0.2, 0.5], [0.4, 0.3]] + target
    for _ in range(2):
        h = attention_layer(h, use_mask)
    return h


def main():
    print('Rows = query; columns = key; 1 = allowed')
    print('     ' + ' '.join(f'{x:>3}' for x in LABELS))
    for q, label in enumerate(LABELS):
        print(f'{label:>4} ' + ' '.join(f'{int(allowed(q, k)):>3}' for k in range(len(LABELS))))
    assert not any(allowed(q, k) for q in range(PREFIX) for k in range(PREFIX, len(LABELS)))
    assert allowed(2, 3)  # Same reference-image block is bidirectional.
    assert not allowed(0, 1)  # Text cannot read future text.
    assert all(allowed(q, k) for q in range(PREFIX, len(LABELS)) for k in range(len(LABELS)))
    target_a, target_b = [[0.2, 0.4], [0.5, -0.3]], [[2.0, -1.0], [-1.0, 3.0]]
    for masked in [True, False]:
        a, b = forward(target_a, masked), forward(target_b, masked)
        delta = max(abs(x - y) for ra, rb in zip(a[:PREFIX], b[:PREFIX]) for x, y in zip(ra, rb))
        print(f'Two-layer prefix change, masked={masked}: {delta:.8f}')
        assert delta == 0 if masked else delta > 1e-6
    print('PASS: toy dependency checks only; not GPU inference validation.')


if __name__ == '__main__':
    main()
