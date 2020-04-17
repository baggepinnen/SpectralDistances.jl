# Models and root manipulations
```@setup lti
using SpectralDistances, InteractiveUtils, Plots, ControlSystems
```

## Overview

Most distances available in this package operate on linear models estimated from a time-domain signal.
This package supports two kind of LTI models, [`AR`](@ref) and [`ARMA`](@ref).
[`AR`](@ref) represents a model with only poles whereas [`ARMA`](@ref) has zeros as well.
These types are subtypes of `ControlSystems.LTISystem`, so many of the functions from the [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl) toolbox work on these models as well. When acting like a `ControlSystems.LTISystem`, the default is to use the continuous-time representation of the model. The discrete-time representation can be obtained by `tf(m, 1)` where `1` is the sample time. More on the time-domain representation below.

!!! note "Note"
    This package makes the assumption that the sample time is 1 everywhere. When an `AbstractModel` is constructed, one must thus take care to rescale the frequency axis accordingly if this does not hold. If the discrete-time representation is never used, this is of no concern.

To fit a model to data, one first has to specify a [`FitMethod`](@ref), the options are
```@example lti
foreach(println, subtypes(SpectralDistances.FitMethod)) # hide
```

For example, to estimate an [`AR`](@ref) model of order 6 using least-squares, we can do the following
```@repl lti
data = randn(1000);
fitmethod = LS(na=6)
model = fitmethod(data)
change_precision(Float32, model) # Can be useful to reduce the computational cost of some distances
pzmap(model)
savefig("pzmap_models.html"); nothing # hide
```

```@raw html
<object type="text/html" data="../pzmap_models.html" style="width:100%;height:450px;"></object>
```


## Time domain
This package allows you to view a model through two different lenses: as a continuous-time model that models the *differential* properties of the signal, or as a discrete-time model that models the *difference* properties of the signal. Signals are inevetably sampled before the computer interacts with them, and are thus natively in the discrete domain. Theory, on the other hand, is slightly more intuitive in the continuous time domain. The two domains are related by the *conformal mapping*
$$p_c = \log(p_d)$$
where $p$ denotes a pole of a transfer function and subscripts $c,d$ denote the continuous and discrete domains respectively. When creating a distance, the default domain is [`Continuous`](@ref). Some functions require you to be explicit regarding which domain you have in mind, such as when creating models from vectors or when asking for the roots/poles of a model.

Sometimes you may get a message saying "Roots on the negative real axis, no continuous time representation exist." when estimating a model from a signal. This means that one of the poles in the discrete time model, which is what is being estimated from data, landed on the legative real axis. No continuous-time system can ever create such a discrete-time model through sampling, and the some features of this package will work slightly worse if such a model is used, notably the [`EuclideanRootDistance`](@ref) and [`embedding`](@ref). Optimal-transport based distances will not have a problem with this scenario.

To reduce the likelihood of this occurring, you may try to bandpass filter the signal before estimating the model, reduce the regularization factor if regularization was used, change the model order, or consider using the [`TLS`](@ref) fit method.

The difference between the pole locations in continuous and discrete time is vizualized in the pole diagrams below
```@example lti
pzmap(model, layout=2, sp=1, xlabel="Re", ylabel="Im", title="Continuous")
vline!([0], primary=false, l=(:black, :dash), sp=1)
pzmap!(tf(model,1),    sp=2, xlabel="Re", ylabel="Im", title="Discrete")
savefig("pzmap_models2.html"); nothing # hide
```

```@raw html
<object type="text/html" data="../pzmap_models2.html" style="width:100%;height:450px;"></object>
```


## Type reference
```@index
Pages = ["ltimodels.md"]
Order   = [:type]
```

## Function reference

```@index
Pages = ["ltimodels.md"]
Order   = [:function]
```

## Docstrings
```@autodocs
Modules = [SpectralDistances]
Private = false
Pages   = ["ltimodels.jl","eigenvalue_manipulations.jl"]
```

```@docs
ControlSystems.tf(m::AR, ts)
ControlSystems.tf(m::AR)
ControlSystems.denvec(::Discrete, m::SpectralDistances.AbstractModel)
```
