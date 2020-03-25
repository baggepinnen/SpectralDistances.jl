# Models and root manipulations
```@setup lti
using SpectralDistances, InteractiveUtils
```

## Overview

This package supports two kind of LTI models, [`AR`](@ref) and [`ARMA`](@ref).
[`AR`](@ref) represents a model with only poles whereas [`ARMA`](@ref) has zeros as well.
These types are subtypes of `ControlSystems.LTISystem`, so many of the functions from the [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl) toolbox work on these models as well. When acting like a `ControlSystems.LTISystem`, the default is to use the continuous-time representation of the model. The discrete-time representation can be obtained by `tf(m, 1)` where `1` is the sample time.

!!! note "Note"
    This package makes the assumption that the sample time is 1 everywhere. When an `AbstractModel` is constructed, one must thus take care to rescale the frequency axis accordingly if this does not hold. If the discrete-time representation is never used, this is of no concern.

To fit a model to data, one first has to specify a [`FitMethod`](@ref), the options are
```@example lti
foreach(println, subtypes(SpectralDistances.FitMethod)) # hide
```

For example, to estimate an [`AR`](@ref) model of order 30 using least-squares, we can do the following
```@repl lti
data = randn(1000);
fitmethod = LS(na=30)
SpectralDistances.fitmodel(fitmethod, data)
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
