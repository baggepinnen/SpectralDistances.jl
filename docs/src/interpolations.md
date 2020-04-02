# Interpolations and Barycenters

Some distances distance define the existence of a shortest path, a *geodesic*. An interpolation is essentially a datapoint on that shortest path. We provide some functionality to interpolate between different spectra and models under transport-based metrics.

Below is an example usage of interpolations. We initially create two random systems, we then define the distance under which to interpolate and then calculate the frequency response for some different values of the interpolation parameter $t \in (0,1)$

```@example
using SpectralDistances, ControlSystems, Distances, Plots, Random
Random.seed!(0)

n = 4
r1 = complex.(-0.01 .+ 0.001randn(3), 2randn(3))
r1 = ContinuousRoots([r1; conj.(r1)])

r2 = complex.(-0.01 .+ 0.001randn(3), 2randn(3))
r2 = ContinuousRoots([r2; conj.(r2)])

r1,r2 = normalize_energy.((r1, r2))

A1 = AR(r1)
A2 = AR(r2)

##
fig1   = plot()
t      = 0.1
dist   = RationalOptimalTransportDistance(domain=Continuous(), p=2, interval=(0., exp10(1.01)))
interp = SpectralDistances.interpolator(dist, A1, A2)
w      = exp10.(LinRange(-1.5, 1, 300))
for t = LinRange(0, 1, 7)
    Φ = clamp.(interp(w,t), 1e-10, 100)
    plot!(w, sqrt.(Φ), xscale=:log10, yscale=:log10, line_z = t, lab="", xlabel="", title="W_2", ylims=(1e-3, 1e1), colorbar=false, l=(1,), c=:viridis)
end

rdist  = EuclideanRootDistance(domain                           =Continuous(), p=2)
interp = SpectralDistances.interpolator(rdist, A1, A2, normalize=false)
fig2   = plot()
for t = LinRange(0, 1, 7)
    Φ = interp(w,t)
    plot!(w, sqrt.(Φ), xscale=:log10, yscale=:log10, line_z = t, lab="", xlabel="", title="RD", ylims=(1e-3, 1e1), colorbar=false, l=(1,), c=:viridis)
end

fig3 = plot()
Φ1   = bode(tf(A1), w)[1][:]
Φ2   = bode(tf(A2), w)[1][:]
for t = LinRange(0, 1, 7)
    plot!(w, (1-t).*Φ1 .+ t.*Φ2, xscale=:log10, yscale=:log10, line_z = t, lab="", xlabel="Frequency", title="L_2", ylims=(1e-3, 1e1), colorbar=false, l=(1,), c=:viridis)
end

fig = plot(fig1, fig2, fig3, layout=(3,1))
savefig("interpolation.svg"); nothing # hide
```

![](interpolation.svg)

## Barycenters
A barycenter is a generalization the the arithmetic mean to metrics other than the Euclidean. A barycenter between models is calculated like this
```julia
bc = barycenter(distance, models)
```
It can be useful to provide some options to the solvers:
```julia
options = (solver=sinkhorn_log!, tol=1e-8, iters=1_000_000, γ=0.0, uniform=true, inneriters=500_000, innertol=1e-6)
distance = OptimalTransportRootDistance(domain=Continuous(), p=2, β=0.01, weight=s1∘residueweight)
bc = barycenter(distance, models; options...)
```
We can plot the barycenters:
```@example
using SpectralDistances, ControlSystems, Plots
models   = examplemodels(3)
distance = OptimalTransportRootDistance(domain=Continuous())
bc       = barycenter(distance, models)
w        = exp10.(LinRange(-0.5, 0.5, 350)) # Frequency vector
G        = tf.(models) # Convert models to transfer functions from ControlSystems.jl
plot()
bodeplot!.(G, Ref(w), plotphase=false, lab="Input models", linestyle=:auto)
bodeplot!(tf(bc), w, plotphase=false, lab="Barycenter", xscale=:identity, c=:green)
savefig("barycenter.svg"); nothing # hide
```
![](barycenter.svg)

## K-Barycenters
Below, we show an example of how one can run the K-barycenter algorithm on a collection of sound signals. `sounds` is expected to be of type `Vector{Vector{T}}`. The example further assumes that there is a vector of `labels::Vector{Int}` that contain the true classes of the datapoints, which you do not have in an unsupervised setting.
```julia
using SpectralDistances, ControlSystems
fitmethod = TLS(na=12)
models = SpectralDistances.fitmodel.(fitmethod, sounds)
G = tf.(models) # Convert to transfer functions for visualization etc.

##
using Clustering
dist = OptimalTransportRootDistance(domain=Continuous(), β=0.01, weight=s1∘residueweight)
@time clusterresult = SpectralDistances.kbarycenters(
    dist,
    models,
    n_classes, # number of clusters
    seed    = :rand,
    solver  = sinkhorn_log!,
    tol     = 2e-6,
    innertol = 2e-6,
    iters   = 100000,
    inneriters = 100000,
    verbose = true,
    output  = :best,
    uniform = true,
    kiters  = 10
)

bc,ass = clusterresult.barycenters, clusterresult.assignments

# Visualize results
using MLBase, Plots.PlotMeasures, AudioClustering
newass,perm = AudioClustering.associate_clusters(labels,ass)
classinds   = 1:n_classes
yt          = (classinds, [label_strings[findfirst(labels .== i)] for i in classinds])

@show mean(labels .== newass)
cm = confusmat(n_classes,labels,newass)
heatmap(cm./sum(cm,dims=2), xlabel="Cluster assignment",ylabel="Best matching class", color=:viridis)
anns = [(reverse(ci.I)..., text(val,12,:gray)) for (ci,val) in zip(CartesianIndices(cm)[:], vec(cm))]
annotate!(anns)
yticks!(yt)
xticks!(yt, xrotation=45)
current()
```
The figure should look like the last figure in [the paper](https://drive.google.com/file/d/1EPS_pyC_opKMLlnk02kIfHbpawWFl4W-/view).

```@index
Pages = ["interpolations.md"]
Order   = [:type, :function, :macro, :constant]
```
```@autodocs
Modules = [SpectralDistances]
Private = false
Pages   = ["interpolations.jl", "barycenter.jl", "kbarycenters.jl"]
```
