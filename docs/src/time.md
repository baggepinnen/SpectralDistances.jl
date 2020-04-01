```@setup time
using SpectralDistances
```
# Time-Frequency distances
For non-stationary signals, it is important to consider how the spectrum changes with time. This package has some, so far, basic support for time-frequency representations of non-stationary signals.
## Overview

We define a custom fit method for fitting time varying spectra, [`TimeWindow`](@ref). It takes as arguments an inner fitmethod, the number of points that form a time window, and the number of points that overlap between two consecutive time windows:
```@repl time
fitmethod = TimeWindow(TLS(na=2), 1000, 500)
y = sin.(0:0.1:100);
model = fitmethod(y)
```
This produces a custom model type, [`TimeVaryingRoots`](@ref) that internally stores a vector of [`ContinuousRoots`](@ref).

Accompanying this time-varying model is a time-aware distance, [`TimeDistance`](@ref). It contains an inner distance (currently only [`OptimalTransportRootDistance`](@ref) supported), and some parameters that are specific to the time dimension, example:
```@repl time
dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=2, weight=s1∘residueweight), tp=2, c=0.1)
```
`tp` is the same as `p` but for the time dimension, and `c` trades off the distance along the time axis with the distance along the frequency axis. A smaller `c` makes it cheaper to transport mass across time. The frequency axis spans `[-π,π]` and the time axis is the non-negative integers, which should give you an idea for how to make this trade-off.

Below, we construct a signal that changes frequency after half the time, and measure the distance between two different such signals. If the time penalty is large, it is cheaper to transport mass along the frequency axis, but if we make `c` smaller, after a while the transport is cheaper along the time dimension
```@example time
# Construct a signal that changes freq after half
signal(f1,f2) = [sin.((0:0.1:49.9).*f1);sin.((50:0.1:99.9).*f2)]
fm = TimeWindow(TLS(na=2), 500, 0)
m = signal(1,2)  |> fm # Signal 1
m2 = signal(2,1) |> fm # Signal 2 has the reverse frequencies

# First it is expensive to transport along time
dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=1, weight=s1∘residueweight), tp=1, c=1.0) # c is large
evaluate(dist, m, m2)
```

```@example time
# Then we make it cheaper
dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=1, weight=s1∘residueweight), tp=1, c=0.01) # c is small
evaluate(dist, m, m2)
```

## Function reference

```@index
Pages = ["time.md"]
Order   = [:function, :macro, :constant]
```

## Docstrings
```@autodocs
Modules = [SpectralDistances]
Private = false
Pages   = ["time.jl"]
```
