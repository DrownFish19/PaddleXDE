from paddlexde.utils.brownian import BrownianInterval

bm = BrownianInterval(t0=-1.0, t1=1.0, size=(4, 1))

print(bm(0.0, 0.5))
