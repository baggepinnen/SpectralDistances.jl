# Models and root manipulations
```@setup lti
using SpectralDistances, InteractiveUtils
```

This package supports two kind of LTI models, [`AR`](@ref) and [`ARMA`](@ref).
[`AR`](@ref) represents a model with only poles whereas [`ARMA`](@ref) has zeros as well.

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
Order   = [:type, :function]
```

## Docstrings
```@autodocs
Modules = [SpectralDistances]
Private = false
Pages   = ["ltimodels.jl","eigenvalue_manipulations.jl"]
```
