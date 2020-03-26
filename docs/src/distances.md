# Distances

## Overview

The following is a reference on all the distances defined in this package. Once a distance is defined, it can be evaluated in one of two ways, defined by the  [Distances.jl](https://github.com/JuliaStats/Distances.jl) interface
```julia
dist = DistanceType(options)
d = evaluate(d, x1, x2; kwargs...) # keyword arguments are used to control the solvers for some transport-based distances
d = dist(x1, x2) # A shorter syntax for calling the distance
```
Before we proceed, the following distances are available
```@index
Pages = ["distances.md"]
Order   = [:type]
```
```@setup dist
using SpectralDistances, InteractiveUtils
```

Some of these distances operate directly on signals, these are
```@example dist
foreach(println, subtypes(SpectralDistances.AbstractSignalDistance)) # hide
```
Of these, [`ModelDistance`](@ref) is a bit special, works like this
```@docs
ModelDistance
```
The inner distance in [`ModelDistance`](@ref) can be any [`AbstractRationalDistance`](@ref). The options are
```@example dist
foreach(println, subtypes(SpectralDistances.AbstractRationalDistance)) # hide
```
These distances operate on LTI models. Some operate on the coefficients of the models
```@example dist
foreach(println, subtypes(SpectralDistances.AbstractCoefficientDistance)) # hide
```
and some operate on the roots of the models
```@example dist
foreach(println, subtypes(SpectralDistances.AbstractRootDistance)) # hide
```

## A full example
To use the [`SinkhornRootDistance`](@ref) and let it operate on signals, we may construct our distance object as follows
```@repl dist
innerdistance = SinkhornRootDistance(domain=Continuous(), β=0.005, p=2)
dist = ModelDistance(TLS(na=30), innerdistance)
X1, X2 = randn(1000), randn(1000);
dist(X1,X2)

dist = ModelDistance(LS(na=2), innerdistance);
t = 0:0.01:10;
X1, X2 = sin.(2π*1 .*t), sin.(2π*1.1 .*t); # Two signals that are close in frequency
dist(X1,X2)
X1, X2 = sin.(2π*1 .*t), sin.(2π*2 .*t);   # Two signals that are further apart in frequency
dist(X1,X2)
```

## Using Welch periodograms
We can calculate the Wasserstein distance between spectra estimated using the Welch method like so
```@repl dist
dist = WelchOptimalTransportDistance(p=2)
X1, X2 = randn(1000), randn(1000);
dist(X1,X2)
t = 0:0.01:10;
X1, X2 = sin.(2π*1 .*t), sin.(2π*1.1 .*t); # Two signals that are close in frequency
dist(X1,X2)
X1, X2 = sin.(2π*1 .*t), sin.(2π*2 .*t);   # Two signals that are further apart in frequency
dist(X1,X2)
```


## Function reference

```@index
Pages = ["distances.md"]
Order   = [:function, :macro, :constant]
```

## Docstrings
```@autodocs
Modules = [SpectralDistances]
Private = false
Pages   = ["losses.jl", "sinkhorn.jl", "jump.jl"]
```

## Details
Transport-based distances may require some tuning parameters to be set for the solvers. The available solvers are
- [`sinkhorn`](@ref): not recommended due to numerical issues, but this is the standard algorithm.
- [`sinkhorn_log`](@ref): better numerical stability than the standard.
- [`sinkhorn_log!`](@ref): in-place version that is faster, but some AD libraries might not like it.
- [`IPOT`](@ref) Finds exact solution (without entropy regularization), requires β around 0.1-1.
- [`ot_jump`](@ref): exact solution using JuMP, requires `using JuMP, GLPK` before it becomes available.

### Providing solver and options
```julia
options = (solver=sinkhorn_log!, tol=1e-6, iters=100_000)
distance = SinkhornRootDistance(domain=Continuous(), p=1, β=0.001)
SpectralDistances.evaluate(distance, model1, model2; options...)
```
