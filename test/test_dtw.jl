using SpectralDistances, DynamicAxisWarping, DSP

nfft = 32
T = 200
q = sin.(1:T)
y = [zeros(10nfft);q;zeros(10nfft)] .+ 0.0001 .* randn.()

Q = spectrogram(q,nfft)
Y = spectrogram(y,nfft)

n, m = size(Q.power)
rad = 5


## Discrete grid transport
res = dtwnn(
    log1p.(Q.power),
    log1p.(Y.power),
    DiscreteGridTransportDistance(Cityblock(), eltype(Q.power), n, n),
    rad,
    saveall = true,
)

@test res.cost < 0.01
@test res.loc == 21
@test all(isfinite, res.dists)
isinteractive() && plot(res.dists)

fm  = TimeWindow(inner = LS(na = 2, λ=1e-3), n = nfft, noverlap = nfft÷2)

Qm  = fm(q) |> change_precision(Float64)
Ym  = fm(y) |> change_precision(Float64);

res = dtwnn(
    Qm,
    Ym,
    EuclideanRootDistance(p = 1, weight = simplex_residueweight),
    rad,
    saveall = true,
)
@test res.cost < 0.01
@test  20 <= res.loc <= 22
@test all(isfinite, res.dists)
isinteractive() && plot(res.dists)
##
res_dtwotrd = dtwnn(
    Qm,
    Ym,
    OptimalTransportRootDistance(p = 1, weight = simplex_residueweight),
    rad,
    saveall = true,
    tol = 1e-3,
)
@test res.cost < 0.01
@test  20 <= res.loc <= 22
@test all(isfinite, res.dists)
isinteractive() && plot(res.dists)


dist = TimeDistance(
    inner = OptimalTransportRootDistance(p = 1, β = 0.5, weight = simplex_residueweight),
    tp    = 1,
    c     = 0.1,
)
res = distance_profile(dist, Qm, Ym, tol=1e-3, check_interval=10)
@test minimum(res) < 1
@test  20 <= argmin(res) <= 22
@test all(isfinite, res)
isinteractive() && plot(res)
