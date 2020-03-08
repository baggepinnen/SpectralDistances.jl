using Test, SpectralDistances
# const Continuous = SpectralDistances.Continuous

@testset "ISA" begin
    @info "Testing ISA"

    ## Extreme case 1- each measure is like a point, the bc should be close to the Euclidean mean of the center of each measure
    d = 2
    k = 4
    X0 = [1 1 2 2; 1 2 1 2]
    X = [X0 .+ 110rand(d) for _ in 1:4]
    S = ISA(X, iters=100, printerval=10)
    X̂ = [X[i][:,S[i]] for i in eachindex(X)]
    bc = mean(X̂)
    @test mean(bc) ≈ mean(mean, X)
    @test mean(bc, dims=2) ≈ mean(mean.(X, dims=2))
    # scatter(eachrow(reduce(hcat,X))...)
    # scatter!(eachrow(bc)...)

    ## Extreme case 2- variability between measures is small in comparison to the intra-measure variability. The bc should be close to the coordinate-wise Euclidean mean
    d = 2
    k = 4
    X0 = [1 1 2 2; 1 2 1 2]
    X = [X0 .+ 0.1rand(d) for _ in 1:4]
    S = ISA(X, iters=100, printerval=10)
    X̂ = [X[i][:,S[i]] for i in eachindex(X)]
    bc = mean(X̂)
    @test mean(bc) ≈ mean(mean, X)
    @test mean(bc, dims=2) ≈ mean(mean.(X, dims=2))
    # scatter(eachrow(reduce(hcat,X))...)
    # scatter!(eachrow(bc)...)

    # Extreme case 3, with weights where one weight is extremely large
    d = 2
    k = 4
    X0 = [1 1 2 2; 1 2 1 2]
    X = [X0 .+ 110rand(d) for _ in 1:4]
    w = s1([10000;ones(length(X)-1)])
    S = ISA(X, w, iters=100, printerval=10)
    X̂ = [w[i]*X[i][:,S[i]] for i in eachindex(X)]
    # bc = mean(X̂ .* w)
    bc = sum(X̂)
    @test mean(bc) ≈ mean(X[1]) rtol=0.1
    # scatter(eachrow(reduce(hcat,X))...)
    # scatter!(eachrow(bc)...)

end

d = EuclideanRootDistance(domain=Continuous(), weight=unitweight)
models = [AR(ContinuousRoots([-1])), AR(ContinuousRoots([-3]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc)[1] ≈ -2

models = [AR(ContinuousRoots([-im,+im])), AR(ContinuousRoots([-3im,+3im]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc) ≈ [-2im,+2im]


d = EuclideanRootDistance(domain=Continuous())
models = [AR(ContinuousRoots([-1])), AR(ContinuousRoots([-3]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc)[1] ≈ -2

models = [AR(ContinuousRoots([-im,+im])), AR(ContinuousRoots([-3im,+3im]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc) ≈ [-2im,+2im]

models = [AR(ContinuousRoots([-im,+im])), AR(ContinuousRoots([-3im,+2im]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc) ≈ [-2im,+1.5im]


d = EuclideanRootDistance(domain=Continuous(), weight=residueweight)
models = [AR(ContinuousRoots([-5-im,-1+im])), AR(ContinuousRoots([-1-im,-1+im]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc) ≈ [-1.1538-im,-1+im] rtol=0.01




##
ζ = [0.1, 0.3, 0.7]

models = map(1:4) do _
    pol = [1]
    for i = eachindex(ζ)
        pol = SpectralDistances.polyconv(pol, [1,2ζ[i] + 0.1randn(),1+0.1randn()])
    end
    AR(Continuous(),pol)
end

Xe = barycenter(EuclideanRootDistance(domain=SpectralDistances.Continuous(),p=2), models)




using ControlSystems

d = SinkhornRootDistance(domain=SpectralDistances.Continuous(),p=2)
bc = barycenter(d, models)
# plot()
# pzmap!.(tf.(models), m=(:c,2))
# pzmap!(tf(bc), color=:blue, m=(:c,2))
##

# Y = [[1. 1], [2. 2], [3. 3]]
# X = [1.1 1.1]
# a = ones(2) ./ 2
# b = [ones(2) ./2 for _ in eachindex(Y)]
# @test alg1(X,Y,a,b;λ=0.1) ≈ a
# @show Xo,ao = alg2(X,Y,a,b;λ=0.1, θ=0.5)
# @test Xo ≈ [2 2] rtol=1e-2
#
# X = [0. 0.]
# @show Xo,ao = alg2(X,Y,a,b;λ=0.1,θ=0.5)
# @test Xo ≈ [2 2] rtol=1e-2
#
# X = [1. 3.]
# @show Xo,ao = alg2(X,Y,a,b;λ=10, θ=0.5)
# @test Xo ≈ [2 2] rtol=1e-2
#
#
# X = [0. 4.]
# @show Xo,ao = alg2(X,Y,a,b;λ=2, θ=0.5)
# @test Xo ≈ [2 2] rtol=1e-2
#
#
#
# Y = [[1. 1], [2. 2], [3. 3]]
# X = [1.1 1.1]
# a = ones(2) ./ 2
# b = [[0.2, 0.8] for _ in eachindex(Y)]
# @show alg1(X,Y,a,b;λ=10)
# @show Xo,ao = alg2(X,Y,a,b;λ=1, θ=0.5)
# @test Xo ≈ [2 2] rtol=1e-2
# @test a ≈ b[1] rtol=1e-2
