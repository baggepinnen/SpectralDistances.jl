```@setup time
using SpectralDistances, Plots
plotly()
```
# Time-Frequency distances
For non-stationary signals, it is important to consider how the spectrum changes with time. This package has some, so far, basic support for time-frequency representations of non-stationary signals.
## Overview

We define a custom fit method for fitting time varying spectra, [`TimeWindow`](@ref). It takes as arguments an inner fitmethod, the number of points that form a time window, and the number of points that overlap between two consecutive time windows:
```@repl time
fitmethod = TimeWindow(LS(na=2), 1000, 500)
y = sin.(0:0.1:100);
model = fitmethod(y)
```
This produces a custom model type, [`TimeVaryingAR`](@ref) that internally stores a vector of [`ContinuousRoots`](@ref).
!!! note "Note"
    `TimeWindow` currently only supports fitmethod `LS`.

Accompanying this time-varying model is a time-aware distance, [`TimeDistance`](@ref). It contains an inner distance (currently only [`OptimalTransportRootDistance`](@ref) supported), and some parameters that are specific to the time dimension, example:
```@repl time
dist = TimeDistance(
    inner = OptimalTransportRootDistance(
        domain = Continuous(),
        p = 2,
        weight = simplex_residueweight,
    ),
    tp = 2,
    c = 0.1,
)
```
`tp` is the same as `p` but for the time dimension, and `c` trades off the distance along the time axis with the distance along the frequency axis. A smaller `c` makes it cheaper to transport mass across time. The frequency axis spans `[-π,π]` and the time axis is the non-negative integers, which should give you an idea for how to make this trade-off.

Below, we construct a signal that changes frequency after half the time, and measure the distance between two different such signals. If the time penalty is large, it is cheaper to transport mass along the frequency axis, but if we make `c` smaller, after a while the transport is cheaper along the time dimension
```@example time
# Construct a signal that changes freq after half
signal(f1,f2) = [sin.((0:0.1:49.9).*f1);sin.((50:0.1:99.9).*f2)]
fm = TimeWindow(LS(na=2), 500, 0)
m = signal(1,2)  |> fm # Signal 1
m2 = signal(2,1) |> fm # Signal 2 has the reverse frequencies

# First it is expensive to transport along time
dist = TimeDistance(
    inner = OptimalTransportRootDistance(
        domain = Continuous(),
        p      = 1,
        weight = simplex_residueweight,
    ),
    tp = 1,
    c  = 1.0, # c is large
)
evaluate(dist, m, m2)
```

Then we make it cheaper
```@example time
dist = TimeDistance(
    inner = OptimalTransportRootDistance(
        domain = Continuous(),
        p      = 1,
        weight = simplex_residueweight,
    ),
    tp = 1,
    c  = 0.01, # c is small
)
evaluate(dist, m, m2)
```


### Chirp example
Here we consider the estimation of the distance between two signals containing chirps, where the onset of the chirp differs. We start by creating some signals with different chirp onsets:
```@example time
fs = 100000
T = 3
t = 0:1/fs:T
N = length(t)
f = range(1000, stop=10_000, length=N)
chirp0 = sin.(f.*t)
function chirp(onset)
    y = 0.1sin.(20_000 .* t)
    inds = max(round(Int,fs*onset), 1):N
    y[inds] .+= chirp0[1:length(inds)]
    y
end
using DSP
plot(spectrogram(chirp(0), window=hanning), layout=2)
plot!(spectrogram(chirp(1), window=hanning), sp=2)
savefig("chirps.html"); nothing # hide
```

```@raw html
<object type="text/html" data="../chirps.html" style="width:100%;height:450px;"></object>
```

We then define the fit method and the distance, similar to previous examples
```@example time
fm     = TimeWindow(LS(na=4, λ=1e-4), 20000, 0)
m      = chirp(1) |> fm # This is the signal we'll measure the distance to
onsets = LinRange(0, 2, 21) # A vector of onset times
cv     = exp10.(LinRange(-3, -0.5, 6)); # a vector of `c` values for the time-transport cost
```

We now calculate the distance to the base signal for varying onsets and varying time-transport costs.
```@example time
dists = map(Iterators.product(cv, onsets)) do (c, onset)
    m2 = chirp(onset) |> fm
    dist = TimeDistance(
        inner = OptimalTransportRootDistance(
            domain = Continuous(),
            p      = 1,
            weight = simplex_residueweight,
        ),
        tp = 1,
        c  = c,
    )
    evaluate(
        dist,
        change_precision(Float64, m),  # we reduce the precision for faster computations
        change_precision(Float64, m2),
        iters = 10000,
        tol = 1e-2, # A coarse tolerance is okay for this example
    )
end

plot(onsets, dists',
    lab      = cv',
    line_z   = log10.(cv)',
    color    = :inferno,
    legend   = false,
    colorbar = true,
    xlabel   = "Onset [s]",
    title    = "Distance as function of onset and time cost"
)
savefig("chirp_dists.html"); nothing # hide
```
The results are shown below. The figure indicates the cost `log10(c)` using the color scale. We can see that the distance between the signals is smallest at `onset=1`, which was the onset for the base signal. We also see that for small values of `c`, it's cheap to transport along time. After increasing `c` for a while it stops getting more expensive, indicating that it's now cheaper to transport along the frequency axis instead.
```@raw html
<object type="text/html" data="../chirp_dists.html" style="width:100%;height:450px;"></object>
```

In this example, we chose the weight function `simplex_residueweight`, which ensures that each time step has the same amount of spectral mass. Individual poles will still have different masses within each timestep, as determined by the pole's residue.

## Dynamic Time Warping
This package interfaces with [DynamicAxisWarping.jl](https://github.com/baggepinnen/DynamicAxisWarping.jl) and provides optimized methods for [`DynamicAxisWarping.dtwnn`](@ref). Below is an example of how to search for a query pattern `Qm` in a much longer pattern `Ym`
```julia
searchresult = dtwnn(
    Qm,
    Ym,
    OptimalTransportRootDistance(p = 1, weight = simplex_residueweight),
    rad,
    saveall = true,
    tol = 1e-3,
)
```
Both `Qm` and `Ym` are expected to be of type [`TimeVaryingAR`](@ref).
```@docs
DynamicAxisWarping.dtwnn(q::TimeVaryingAR, y::TimeVaryingAR, dist, rad::Int)
```

For examples of the combination of DTW and OT, see the following notebooks
- [DTW-OT: Introduction](https://nbviewer.jupyter.org/github/baggepinnen/julia_examples/blob/master/frequency_warping.ipynb)
- [DTW-OT: Detection](https://nbviewer.jupyter.org/github/baggepinnen/julia_examples/blob/master/frequency_warping2.ipynb)

## Distance profile
A distance profile between a query pattern `Qm` and a much longer pattern `Ym` can be computed efficiently with (example)
```julia
dist = TimeDistance(
    inner = OptimalTransportRootDistance(p = 1, β = 0.5, weight = simplex_residueweight),
    tp    = 1,
    c     = 0.1,
)
res_tt = distance_profile(dist, Qm, Ym, tol=1e-3)
```
Both `Qm` and `Ym` are expected to be of type [`TimeVaryingAR`](@ref).
```@docs
SlidingDistancesBase.distance_profile(od::TimeDistance, q::TimeVaryingAR, y::TimeVaryingAR)
SlidingDistancesBase.distance_profile(od::ConvOptimalTransportDistance, q::DSP.Periodograms.TFR, y::DSP.Periodograms.TFR)
```

## Docstrings
```@autodocs
Modules = [SpectralDistances]
Private = false
Pages   = ["time.jl"]
```
