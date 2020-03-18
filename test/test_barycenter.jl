using Test, SpectralDistances, ControlSystems
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
    # scatter!(eachrow(bc)..., alpha=0.2)

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



##

@testset "Barycentric coordinates" begin
    @info "Testing Barycentric coordinates"

d = 2
k = 4
S = 4
X0 = [1 1 2 2; 1 2 1 2]

res = map(1:10) do _
    X = [X0[:,randperm(k)] .+ 50rand(d) for _ in 1:S]
    sw = ISA(X, iters=100, printerval=10)
    X̂ = [X[i][:,sw[i]] for i in eachindex(X)]
    bc = mean(X̂)
    @test mean(bc) ≈ mean(mean, X) rtol=0.01
    @test mean(bc, dims=2) ≈ mean(mean.(X, dims=2)) rtol=0.01

    w = rand(S) |> s1
    sw = ISA(X, w, iters=100, printerval=10)
    X̂ = SpectralDistances.barycentric_weighting(X,w,sw)


    a = rand(k) |> s1
    b = rand(k) |> s1

    γ = 10.0


    # α = exp.(α)
    λ0 = rand(S) |> s1
    p = repeat(ones(k) |> s1, 1, S)
    q = rand(k) |> s1

    ql = barycenter(X, λ0)
    C = [[mean(abs2, x1-x2) for x1 in eachcol(Xi), x2 in eachcol(ql)] for Xi in X]


    bch,λh = SpectralDistances.barycentric_coordinates(X,ql,p,q, γ=γ, L=32)
    # scatter(eachrow(reduce(hcat,X))...)
    # scatter!(eachrow(ql)...)
    # scatter!(eachrow(bch)..., alpha=0.5)

    norm(λ0-λh), norm(bch-ql) < mean(norm(x-ql) for x in X)
end

@test median(getindex.(res,1)) < 0.2
@test mean(getindex.(res,2)) >= 0.9
# ##
# X0 = [1. 1 2 2 3 3; 1 2 3 1 2 3]
# # X0 = [X0 X0]
# d,k = size(X0)
# S = 7
# X = [X0[:,randperm(k)] .+ 0.050randn(d) .+ 0.0001 .* randn(d,k) for _ in 1:S]
# w = ones(S) |> s1
# sw = ISA(X, w, iters=100, printerval=1)
# X̂ = SpectralDistances.barycentric_weighting(X,w,sw)
# # X̂2 = SpectralDistances.barycentric_weighting2(X,w,sw)
# scatter(eachrow(reduce(hcat,X))...)
# scatter!(eachrow(X̂)..., alpha=0.5)
# # scatter!(eachrow(X̂2)..., alpha=0.5)
# # In the plot above, the bc should have the same shape as the acnhors
#
#
# error("keep track of the cost function noted in the paper and ensure that it decreases")
# ##
#
# # Now testing for models
# ζ = [0.1, 0.3, 0.7]
# models = map(1:50) do _
#     pol = [1]
#     for i = eachindex(ζ)
#         pol = SpectralDistances.polyconv(pol, [1,2ζ[i] + 0.1randn(),1+0.1randn()])
#     end
#     AR(Continuous(),pol)
# end
#
# # λ0 = rand(length(models)) |> s1
# # qmodel = barycenter(distance, models, λ0)
#
# distance = SinkhornRootDistance(domain=Continuous(),p=2, weight=unitweight)
#
# # We choose the point to be projected equal to one of the anchor points. In this case, the barycentric coordinates is a one-hot vector
# qmodel = models[1]
# q_proj, λ = barycentric_coordinates(distance, models, qmodel, γ = 0.2, robust=true)
# @test λ[1] ≈ 1 atol=0.2
#
# qmodel = models[end]
# q_proj, λ = barycentric_coordinates(distance, models, qmodel, γ=0.2, robust=false)
# @test λ[end] ≈ 1 atol=0.2
#
#
# qbc = barycenter(distance, models, λ) # TODO: the problem might be in this method of barycenter, not sure if it's actually any good
#
#
#
# plot()
# pzmap!.(tf.(models), lab="")
# pzmap!(tf(qmodel), m=:c, lab="q")
# pzmap!(tf(q_proj), m=:c, lab="q_proj", legend=true)

end
