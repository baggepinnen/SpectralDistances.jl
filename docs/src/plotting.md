# Plotting

```@example plotting
using SpectralDistances, Plots; plotly() # hide
models = examplemodels(2)
assignmentplot(models[1], models[2], d=Discrete(), p=2)
savefig("assignment.html"); nothing # hide
```
```@raw html
<object type="text/html" data="../assignment.html" style="width:100%;height:450px;"></object>
```

```@example plotting
dist = OptimalTransportRootDistance(domain=Continuous())
flowplot(dist, models...)
savefig("flowplot.html"); nothing # hide
```
```@raw html
<object type="text/html" data="../flowplot.html" style="width:100%;height:450px;"></object>
```

Apart from the functions above, the plotting facilities from ControlSystemsBase.jl should also work on models from this package, e.g., `bodeplot, pzmap` etc.

```@index
Pages = ["plotting.md"]
Order   = [:type, :function, :macro, :constant]
```
```@autodocs
Modules = [SpectralDistances]
Private = false
Pages   = ["plotting.jl"]
```
