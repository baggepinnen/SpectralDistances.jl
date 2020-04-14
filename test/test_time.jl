# Test simple indexing
r = AR(ContinuousRoots(randn(2)))
m = TimeVaryingAR([r,r,r])

@test length(m) == 3*2
@test size(m) == (2,3)
mc = SpectralDistances.change_precision(Float32, m)
@test m ≈ mc
@test eltype(mc.models[1].pc) == Complex{Float32}

@test m[1] == m[1,1]
@test m[2] == m[2,1]
@test m[3] == m[1,2]
@test m[4] == m[2,2]

D = SpectralDistances.distmat_euclidean(m,m,2,2,1)
@test all(iszero, diag(D))
@test D[1,3] == 1
@test D[1,2] != 1
@test D[1,5] == 2^2

D = SpectralDistances.distmat_euclidean(m,m,1,1,1)
@test all(iszero, diag(D))
@test D[1,3] == 1
@test D[1,2] != 1
@test D[1,5] == 2^1


# test fitting
fm = TimeWindow(TLS(na=2), 1000, 500)
y = randn(10000)
m = fm(y)

y2 = randn(10000)
m2 = fm(y2)

@test m[1] == m[1,1]
@test m[2] == m[2,1]
@test m[3] == m[1,2]
@test m[4] == m[2,2]

D = SpectralDistances.distmat_euclidean(m,m,2,2,1)
@test all(iszero, diag(D))

# test that the distance increases as expected with varying frequencies, p=1
dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=1, weight=simplex_residueweight), tp=1, c=1.0)

@test evaluate(dist, m, m) < 1e-3
@test evaluate(dist, m, m2, iters=10000, tol=1e-3) > 0.1

fm = TimeWindow(TLS(na=2), 1000, 500)
y = sin.(0:0.1:100)
m = fm(y)

dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=1, weight=simplex_residueweight), tp=1, c=1.0)
dists = map([1,2,3,4,5]) do f
    y2 = sin.((0:0.1:100).*f)
    m2 = fm(y2)
    evaluate(dist, m, m2, iters=10000, tol=1e-3)
end

@test all(>(0), diff(dists))
@test dists ≈ 0.1*(0:4) atol=1e-3


# test that the distance increases as expected with varying frequencies, p=2
dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=2, weight=simplex_residueweight), tp=1, c=1.0)
dists = map([1,2,3,4,5]) do f
    y2 = sin.((0:0.1:100).*f)
    m2 = fm(y2)
    evaluate(dist, m, m2, iters=10000, tol=1e-3)
end

@test all(>(0), diff(dists))
@test dists ≈ (0.1*(0:4)).^2 atol=1e-3


# Construct a signal that changes freq after half
signal(f1,f2) = [sin.((0:0.1:49.9).*f1);sin.((50:0.1:99.9).*f2)]
fm = TimeWindow(TLS(na=2), 500, 0)
m = signal(1,2)  |> fm
m2 = signal(2,1) |> fm
# Mess with c such that it becomes cheap to transport in time
dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=1, weight=simplex_residueweight), tp=1, c=1.0)
d = evaluate(dist, m, m2, iters=10000, tol=1e-3)
@test d ≈ 0.1 rtol=5e-2

dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=1, weight=simplex_residueweight, β=0.001), tp=1, c=0.01)
d = evaluate(dist, m, m2, iters=10000, tol=1e-5)
@test d ≈ 0.01 rtol=5e-2


## Test chirps

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
isinteractive() && begin
    @eval using SpectralDistances, DSP, LPVSpectral, ThreadPools
    plot(spectrogram(chirp(0), window=hanning), layout=2)
    plot!(spectrogram(chirp(1), window=hanning), sp=2)
end
fm = TimeWindow(LS(na=4, λ=1e-4), 20000, 0)
m = chirp(1)  |> fm
# @show length(m)

onsets = LinRange(0, 2, 21); 1 ∈ onsets
cv = exp10.(LinRange(-3, -0.5, 6))

@time dists = map(Iterators.product(cv,onsets)) do (c,onset)
    m2 = chirp(onset) |> fm
    dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=1, weight=simplex_residueweight), tp=1, c=c)
    evaluate(dist, change_precision(Float64, m), change_precision(Float64, m2), iters=10000, tol=1e-2)
end

isinteractive() && plot(onsets, dists', lab=cv', line_z = log10.(cv)', color=:inferno, legend=false, colorbar=true, xlabel="Onset [s]", )
