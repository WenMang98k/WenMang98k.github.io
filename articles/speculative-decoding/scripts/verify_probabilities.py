"""Exact finite-vocabulary checks; does not benchmark LLM inference."""
import random
random.seed(20261111)
def normalized(xs):
    total=sum(xs)
    return [x/total for x in xs]
for n in [2,3,7,31]:
    for _ in range(250):
        p=normalized([random.random() for _ in range(n)])
        q=normalized([random.random() for _ in range(n)])
        accepted=[min(x,y) for x,y in zip(p,q)]
        rejection=1-sum(accepted)
        residual=[max(0,x-y) for x,y in zip(p,q)]
        assert abs(sum(residual)-rejection)<1e-12
        if rejection>1e-12:
            result=[a+rejection*r/sum(residual) for a,r in zip(accepted,residual)]
            assert max(abs(a-b) for a,b in zip(result,p))<1e-12
        assert abs(sum(accepted)-(1-.5*sum(abs(a-b) for a,b in zip(p,q))))<1e-12
for p,q in [([1.,0.],[0.,1.]),([.5,.5],[.5,.5]),([.7,.3],[.4,.6])]:
    assert all(abs(a+r-t)<1e-12 for a,r,t in zip([min(x,y) for x,y in zip(p,q)],[max(0,x-y) for x,y in zip(p,q)],p))
print('PASS: 1000 random distributions and 3 edge/example cases; acceptance + residual = target; overlap = 1 - TV')
