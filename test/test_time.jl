# Test simple indexing
r = ContinuousRoots(randn(2))
m = TimeVaryingRoots([r,r,r])

@test m[1] == m[1,1]
@test m[2] == m[2,1]
@test m[3] == m[1,2]
@test m[4] == m[2,2]

D = SpectralDistances.distmat_euclidean(m,m,2)
@test all(iszero, diag(D))
@test D[1,3] == 1
@test D[1,2] != 1
@test D[1,5] == 2^2

D = SpectralDistances.distmat_euclidean(m,m,1)
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

D = SpectralDistances.distmat_euclidean(m,m)
@test all(iszero, diag(D))

# test that the distance increases as expected with varying frequencies, p=1
dist = OptimalTransportRootDistance(domain=Continuous(), p=1, weight=s1∘residueweight)

@test evaluate(dist, m, m, printerval=1) < 1e-3
evaluate(dist, m, m2, printerval=1000, iters=10000, tol=1e-3) > 0.1

fm = TimeWindow(TLS(na=2), 1000, 500)
y = sin.(0:0.1:100)
m = fm(y)

dists = map([1,2,3,4,5]) do f
    y2 = sin.((0:0.1:100).*f)
    m2 = fm(y2)
    evaluate(dist, m, m2, iters=10000, tol=1e-3)
end

@test all(>(0), diff(dists))
@test dists ≈ 0.1*(0:4) atol=1e-3

# test that the distance increases as expected with varying frequencies, p=2
dist = OptimalTransportRootDistance(domain=Continuous(), p=2, weight=s1∘residueweight)
dists = map([1,2,3,4,5]) do f
    y2 = sin.((0:0.1:100).*f)
    m2 = fm(y2)
    evaluate(dist, m, m2, iters=10000, tol=1e-3)
end

@test all(>(0), diff(dists))
@test dists ≈ (0.1*(0:4)).^2 atol=1e-3
