[![Build Status](https://travis-ci.org/baggepinnen/SpectralDistances.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/SpectralDistances.jl)
[![codecov](https://codecov.io/gh/baggepinnen/SpectralDistances.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/SpectralDistances.jl)


## Examples
```julia
loss = ModelDistance(LS(na=order), EuclideanCoefficientDistance())
loss = ModelDistance(LS(na=order), InnerProductCoefficientDistance())
loss = ModelDistance(LS(na=order), KernelWassersteinRootDistance(domain=Discrete(), λ=10.))
loss = ModelDistance(LS(na=order), KernelWassersteinRootDistance(domain=Continuous(), λ=10.))
loss = ModelDistance(LS(na=order), KernelWassersteinRootDistance(domain=Continuous(), λ=0.1, transform=logmag))

lossobj = ModelDistance(LS(na=10), InnerProductCoefficientDistance()) + EnergyDistance()
loss = X -> evaluate(lossobj, X,X0)

loss = (Xp) -> batch_loss(batchsize, lossobj, X, Xp)
gradfun = Xp->Flux.gradient(loss, Xp)[1] |> Flux.data
```

# List of distances
```julia
EuclideanCoefficientDistance
InnerProductCoefficientDistance
ModelDistance
EuclideanRootDistance
ManhattanRootDistance
HungarianRootDistance
KernelWassersteinRootDistance
OptimalTransportModelDistance
OptimalTransportSpectralDistance
ClosedFormSpectralDistance
EnergyDistance
```
# List of models and fitmethods
```julia
AR: model
ARMA: model
PLR: fitmethod for ARMA
LS: fitmethod for AR
```

# Other types and functions
```julia
Continuous
Discrete
SortAssignement
HungarianAssignement
domain_transform # transform roots or model between continuous and discrete domains
```
