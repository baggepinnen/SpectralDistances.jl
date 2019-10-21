# Interpolations

Whenever you define a distance, that distance implies the existence of a shortest path, a *geodesic*. An interpolation is essentially a datapoint on that shortest path. We provide some functionality to interpolate between different spectra and models.

```@index
Pages = ["interpolations.md"]
Order   = [:type, :function, :macro, :constant]
```
```@autodocs
Modules = [SpectralDistances]
Private = false
Pages   = ["interpolations.jl"]
```
