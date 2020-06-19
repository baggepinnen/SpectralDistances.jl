# Interpolations and Barycenters

Some distances distance define the existence of a shortest path, a *geodesic*. An interpolation is essentially a datapoint on that shortest path. We provide some functionality to interpolate between different spectra and models under transport-based metrics.

Below is an example usage of interpolations. We initially create two random systems, we then define the distance under which to interpolate and then calculate the frequency response for some different values of the interpolation parameter $t \in (0,1)$

```@example
using SpectralDistances, ControlSystems, Distances, Plots, Random
plotly()
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
savefig("interpolation.html"); nothing # hide
```

```@raw html
<object type="text/html" data="../interpolation.html" style="width:100%;height:450px;"></object>
```

## Barycenters
A barycenter is a generalization the the arithmetic mean to metrics other than the Euclidean. A barycenter between models is calculated like this
```julia
bc = barycenter(distance, models)
```
It can be useful to provide some options to the solvers:
```julia
options = (solver=sinkhorn_log!, tol=1e-8, iters=1_000_000, γ=0.0, uniform=true, inneriters=500_000, innertol=1e-6)
distance = OptimalTransportRootDistance(domain=Continuous(), p=2, β=0.01, weight=simplex_residueweight)
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
savefig("barycenter.html"); nothing # hide
```
```@raw html
<object type="text/html" data="../barycenter.html" style="width:100%;height:450px;"></object>
```

### Barycenters between spectrograms
We can also calculate a barycenter between spectrograms (or arbitrary matrices) using an efficient convolutional method. The most important parameter to tune in order to get a good result, apart from the regularization parameter `β`, is the `dynamic_floor`. This parameter determines where (in log space) the floor of the PSD is. This serves as a denoising, why the barycenter appears with a very dark background in the image below.
```@example barycenter_spectrograms
using SpectralDistances, DSP, Plots
N     = 24_000
t     = 1:N
f     = range(0.8, stop=1.2, length=N)
y1    = sin.(t .* f) .+ 0.1 .* randn.()
y2    = sin.(t .* reverse(f .+ 0.5)) .+ 0.1 .* randn.()
S1,S2 = spectrogram.((y1,y2), 1024)

A = [S1,S2]
β = 0.0001     # Regularization parameter (higher implies more smoothing and a faster, more stable solution)
λ = [0.5, 0.5] # Barycentric coordinates (must sum to 1)
B = barycenter_convolutional(A, β=β, tol=1e-6, iters=2000, ϵ=1e-100, dynamic_floor=-2)
plot(
    plot(S1, title="S1"),
    plot(B, title="Barycenter"),
    plot(S2, title="S2"),
    layout=(1,3),
    colorbar=false
)

savefig("barycenter_sg.html"); nothing # hide
```
```@raw html
<object type="text/html" data="../barycenter_sg.html" style="width:100%;height:450px;"></object>
```
Note that in order to calculate the barycenter, the sum of each input spectrogram is normalized.

This function works for any vector of matrices as long as all entries are positive and each matrix has an equal sum.

For a more thourogh example, see [whistle.jl](https://github.com/baggepinnen/SpectralDistances.jl/blob/master/examples/whistle.jl).

### Trade off between frequency and time
There is currently no way of having different costs between transport in time and transport along the frequency axis other than to change the resolution of the spectrogram.


## Barycentric coordiantes
The inverse problem to that of finding a barycenter is that of finding the barycentric coordinates λ of a query point $Q$, such that the resulting barycenter is as close as possible to the query point. Given a set of rational spectra $\left\{ G_i \right\}$, a nonlinear projection of a spectrum $Q$ onto this set can be obtained by solving the following nested optimization problem
```math
\begin{aligned}
  λ &= \argmin_{\bar{λ}} \, W\big(Q,  Q^*(\bar{λ})\big)\\
  Q^*(\bar{λ}) &= \argmin_{\bar{Q}} \sum_i \bar{λ}_i W(G_i, \bar{Q})
\end{aligned}
```
where $λ$ are the barycentric coordinates belonging to the probability simplex. Problems of this type are sometimes referred to as [histogram regression](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf).

A nonlinear projection onto a basis consisting of spectra can be useful for, e.g., spectral dictionary learning, basis pursuit, topic modelling, denoising and detection. The function [`barycentric_coordinates`](@ref) is available for select distances:
```@docs
barycentric_coordinates
```

Below is an example that assumes that you have access to a vector with a signal called `y` and the corresponding sample rate `fs` (in this example, we use the `y` constructed in the previous example. In the example, we simply use one of the input spectra as query point. This way, we know what barycentric coordinates we expect
```@example barycenter_spectrograms
using SpectralDistances, DSP, Plots
models = map(DSP.arraysplit(y1, 8*512, 0)) do y
    spectrogram(y, 512, fs = 24_000, window=hanning)
end

matrices = s1.(normalize_spectrogram.(models))

β      = 0.01
dist   = ConvOptimalTransportDistance(β=β)
Q      = matrices[2] # Use the second spectrogram as query
λ, res = barycentric_coordinates(dist, matrices, Q)
plot(
    bar(λ, title="Barycentric coordinates", lab=""),
    plot(spectrogram(y1, window=hanning), title="Signal"),
    heatmap(Q, title="Query point")
)

savefig("barycentric_coords_sg.html"); nothing # hide
```
```@raw html
<object type="text/html" data="../barycentric_coords_sg.html" style="width:100%;height:450px;"></object>
```

## K-Barycenters
Below, we show an example of how one can run the K-barycenter algorithm on a collection of sound signals. `sounds` is expected to be of type `Vector{Vector{T}}`. The example further assumes that there is a vector of `labels::Vector{Int}` that contain the true classes of the datapoints, which you do not have in an unsupervised setting.
```julia
using SpectralDistances, ControlSystems
fitmethod = TLS(na=12)
models = SpectralDistances.fitmodel.(fitmethod, sounds)
G = tf.(models) # Convert to transfer functions for visualization etc.

##
using Clustering
dist = OptimalTransportRootDistance(domain=Continuous(), β=0.01, weight=simplex_residueweight)
@time clusterresult = SpectralDistances.kbarycenters(
    dist,
    models,
    n_classes, # number of clusters
    seed       = :rand,
    solver     = sinkhorn_log!,
    tol        = 2e-6,
    innertol   = 2e-6,
    iters      = 100000,
    inneriters = 100000,
    verbose    = true,
    output     = :best,
    uniform    = true,
    kiters     = 10
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
The figure should look like the last figure in [the paper](http://arxiv.org/abs/2004.09152).

```@index
Pages = ["interpolations.md"]
Order   = [:type, :function, :macro, :constant]
```
```@autodocs
Modules = [SpectralDistances]
Private = false
Pages   = ["interpolations.jl", "barycenter.jl", "kbarycenters.jl"]
```
