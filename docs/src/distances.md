# Distances

## Overview

The following is a reference for all the distances defined in this package. Once a distance object is defined, it can be evaluated in one of two ways, defined by the  [Distances.jl](https://github.com/JuliaStats/Distances.jl) interface
```julia
dist = DistanceType(options)
d = evaluate(d, x1, x2; kwargs...) # keyword arguments are used to control the solvers for some transport-based distances
d = dist(x1, x2) # A shorter syntax for calling the distance
```
**Note:** All distances return the distance raised to the power `p`, thus
`RationalOptimalTransportDistance(p=2)(x1,x2) == W₂(x1,x2)^2` where `W₂` denotes the Wasserstein distance of order 2.

Before we proceed, we list a number of classes of distances that are available
```@index
Pages = ["distances.md"]
Order   = [:type]
```
```@setup dist
using SpectralDistances, InteractiveUtils, DSP
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
To use the [`OptimalTransportRootDistance`](@ref) and let it operate on signals, we may construct our distance object as follows
```@repl dist
innerdistance = OptimalTransportRootDistance(domain=Continuous(), β=0.005, p=2)
dist = ModelDistance(TLS(na=10), innerdistance)
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

## Gradients
Some distances will allow you to propagate gradients
through them. Below is an example using Zygote and
the [`OptimalTransportRootDistance`](@ref)
```@example dist
using Zygote
Zygote.@nograd rand # Currently needed woraround
x1 = SpectralDistances.bp_filter(randn(100000), (0.1,0.3))  # Create two signals
x2 = SpectralDistances.bp_filter(randn(100000), (0.1,0.2))
fm = LS(na=10)      # LS is the best supported fitmethod for gradients

dist = ModelDistance(fm,OptimalTransportRootDistance(domain = Continuous()))      # Since we're measureing distance between signals, we wrap the distance in a ModelDistance
```
```@repl dist
dist(x1,x2)
∇x1 = Zygote.gradient(x->real(evaluate(dist,x,x2)), x1)[1] # The call to real is a workaround for a Zygote bug
```

The differentiation takes some time, but it should be fast enough to be generally useful for gradient-based learning of autoencoders etc. The following is a benchmark performed on an old laptop without GPU (the distances are not yet tested on GPUs)
```julia
@btime Zygote.gradient(x->real(evaluate($dist,x,$x2)), $x1);
#  134.965 ms (107566 allocations: 134.77 MiB)
```
with a precomputed reference model, it goes even faster
```julia
m2 = fm(x2)
m2 = change_precision(Float64, m2) # Tihs step is important for performance
@btime Zygote.gradient(x->real(evaluate($dist,x,$m2)), $x1);
#  80.200 ms (103437 allocations: 69.62 MiB)
```

The same benchmarks performed on a 2019 desktop computer yields the following timings
```julia
julia> @btime Zygote.gradient(x->real(evaluate($dist,x,$x2)), $x1);
  45.926 ms (107748 allocations: 136.18 MiB)

julia> @btime Zygote.gradient(x->real(evaluate($dist,x,$m2)), $x1);
  25.120 ms (103619 allocations: 71.03 MiB)
```


## Unbalanced transport
There are situations in which one would like to avoid fully transporting all mass between two measures. A few such cases are
- The two measures do not have the same mass. In this case, the standard, balanced, optimal-transport problem is unfeasible.
- Energy is somehow lost or added to one spectra in a way that should not be accounted for by transport. This would be the case if
  - Spectral energy is absorbed by a channel through which a signal is propagated. In this case it would not make sense to try to transport mass from the other spectrum away from the absorbed (dampended) frequency.
  - Spectral energy is added by a noise source. This energy should ideally not be considered for transport and should rather be destroyed.

For situations like this, an `AbstractDivergence` can be supplied to the `OptimalTransportRootDistance`. This divergence specifies how expensive it is to create or destroy mass in the spectra. The available divergences are listed in [the docs of UnbalancedOptimalTransport.jl](https://ericphanson.github.io/UnbalancedOptimalTransport.jl/stable/public_api/#Divergences-1), to which we outsource the solving of the unbalanced problem. For convenience, the wrapper [`sinkhorn_unbalanced`](@ref) is available to interface the unbalanced solver in the same way as the solvers from this package are interfaced.

```@repl dist
using DSP
fm = LS(na = 10)
m1 = fm(filtfilt(ones(10), [10], randn(1000)))
m2 = fm(filtfilt(ones(5), [5], randn(1000)))
dist = OptimalTransportRootDistance(domain = Continuous(), p=1, divergence=Balanced())
d1 = evaluate(dist,m1,m2)
dist = OptimalTransportRootDistance(domain = Continuous(), p=1, divergence=KL(1.0))
d2 = evaluate(dist,m1,m2)
dist = OptimalTransportRootDistance(domain = Continuous(), p=1, divergence=KL(10.0))
d3 = evaluate(dist,m1,m2)
dist = OptimalTransportRootDistance(domain = Continuous(), p=1, divergence=KL(0.01))
d4 = evaluate(dist,m1,m2)
d1 > d3 > d2 > d4
```
When the distance is evaluated the second time, unbalanced transport is used. The `d2` should be equal to or smaller than `d1`. If we make the `KL` term larger, the distance approaches the balanced cost, and if we make it smaller, it becomes very cheap to create/destroy mass and less mass is transported.

The first case, where `divergence=Balanced()` was supplied, should be equivalent to not providing any divergence at all. In pratice results might differ slightly since a different solver implementation is used.

Currently, only distance calculations using unbalanced transport is supported. Barycenter calculations will currently ignore the divergence.

Below is an example in which the unbalanced transport between two systems is computed. The two systems do not have the same number of poles, and if destruction of mass is made cheap, not all mass is transported. The thickness of the lines indicate mass flow.
```@example dist
using Plots
m1 = AR(Continuous(), [1, 0.1, 1.3])                        |> change_precision(Float64)
m2 = AR(Continuous(), polyconv([1, 0.1, 1], [1, 0.1, 1.2])) |> change_precision(Float64)
D  = SpectralDistances.distmat_euclidean(m1.pc, m2.pc)
w1, w2 = unitweight.((m1, m2))
figs = map([0.001, 0.01, 0.1]) do tv
    divergence = TV(tv)
    Γ, a, b = sinkhorn_unbalanced(D, w1, w2, divergence, β = 0.01)
    lineS   = 20Γ
    lineS[lineS.<0.1] .= 0
    alphaS  = lineS ./ maximum(lineS)
    f       = scatter(m1.pc, legend = false, ms = 10, title = "TV=$tv")
    scatter!(m2.pc, ms = 10)
    for (i, p1) in enumerate(m1.pc), (j, p2) in enumerate(m2.pc)
        coords = [p1, p2]
        plot!(
            real(coords),
            imag(coords),
            linewidth = lineS[i, j],
            alpha     = alphaS[i, j],
            color     = :black,
        )
    end
    f
end
plot(figs..., layout = (1, 3), ylims = (0.9, 1.2))
savefig("unbalanced_poles.html"); nothing # hide
```

```@raw html
<object type="text/html" data="../unbalanced_poles.html" style="width:100%;height:450px;"></object>
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
- [`sinkhorn`](@ref): not recommended due to numerical issues, but this is the most commonly cited algorithm.
- [`sinkhorn_log`](@ref): better numerical stability than the standard.
- [`sinkhorn_log!`](@ref): in-place version that is faster, but some AD libraries might not like it (often the default if no solver is provided).
- [`IPOT`](@ref) Finds exact solution (without entropy regularization), requires β around 0.1-1.
- [`ot_jump`](@ref): exact solution using JuMP, requires `using JuMP, GLPK` before it becomes available.

### Providing solver and options
```julia
options = (solver=sinkhorn_log!, tol=1e-6, iters=100_000)
distance = OptimalTransportRootDistance(domain=Continuous(), p=1, β=0.001)
SpectralDistances.evaluate(distance, model1, model2; options...)
```
