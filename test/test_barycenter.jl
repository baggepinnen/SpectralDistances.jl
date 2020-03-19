using Test, SpectralDistances, ControlSystems
# const Continuous = SpectralDistances.Continuous

@testset "ISA" begin
    @info "Testing ISA"

    ## Extreme case 1- each measure is like a point, the bc should be close to the Euclidean mean of the center of each measure
    d = 2
    k = 4
    X0 = [1 1 2 2; 1 2 1 2]
    X = [X0 .+ 110rand(d) for _ in 1:4]
    S = ISA(X, iters=100, printerval=100)
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
    S = ISA(X, iters=100, printerval=100)
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
    S = ISA(X, w, iters=100, printerval=100)
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



@testset "alg1 alg2" begin
    @info "Testing alg1 alg2"


Y = [[1. 1], [2. 2], [3. 3]]
X = [1.1 1.01]
a = ones(2) ./ 2
b = [ones(2) ./2 for _ in eachindex(Y)]
Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=10, θ=0.5, printerval=190, γ=0.01)
@test Xo ≈ [2 2] rtol=1e-2

X = [0. 0.1]
@show Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=10,θ=0.5, γ=0.01)
@test Xo ≈ [2 2] rtol=1e-2

X = [1. 3.]
@show Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=10.0, θ=0.5, γ=0.01)
@test Xo ≈ [2 2] rtol=1e-2


X = [0. 4.]
@show Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=10, θ=0.5, γ=0.01)
@test Xo ≈ [2 2] rtol=1e-2


Y = [[1. 2], [2. 3], [3. 4]]
X = [2.0 3]
a = ones(2) |> s1
b = [ones(2) ./2 for _ in eachindex(Y)]
a1 = SpectralDistances.alg1(X,Y,a,b;λ=10, tol=1e-5)
@test a1[1] == a1[2]

X = [2.1 3]
a1 = SpectralDistances.alg1(X,Y,a,b;λ=10, tol=1e-5)
@test a1[1] < a1[2]
@show Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=10, θ=0.5, printerval=100, iters=500, tol=1e-8, γ=0.01)
@test Xo ≈ [2 3] rtol=5e-1
# @test ao ≈ b[1] rtol=0.1




X = [0. 0.1]
@show Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=10, θ=0.5, printerval=100, innertol=1e-7, inneriters=1000, tol=1e-7, γ=0.1)
@test Xo ≈ [2 3] rtol=5e-1

X = [1. 3.]
@show Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=10, θ=0.5, printerval=100, innertol=1e-4, inneriters=100, tol=1e-5)
@test Xo ≈ [2 3] rtol=5e-1


X = [0. 4.]
@show Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=2, θ=0.5)
@test Xo ≈ [2 3] rtol=5e-1

Y = [[1. 1], [2. 2], [3. 3]]
X = [1.1 1.01]
a = ones(2) |> s1
b = [[0.2, 0.8] for _ in eachindex(Y)]
@test SpectralDistances.alg1(Y[2],Y,b[1],b;λ=10, printerval=1) == b[1]


@show Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=10, θ=0.5, printerval=100)
@test Xo ≈ [2 2] rtol=5e-1
@test_broken ao ≈ b[1] rtol=1e-2


##
d = 2
k = 4
Y0 = 0.1*[1 1 2 2; 1 2 1 2]
Y = [Y0[:,randperm(k)] .+ 1rand(d) for _ in 1:4]

X = mean(Y) .+ 0.05 .* randn.()
a = ones(k) |> s1
b = [ones(k) |> s1 for _ in eachindex(Y)]
a1 = SpectralDistances.alg1(X,Y,a,b;λ=100.0, printerval=100)
# @test a1 ≈ a rtol=0.01


@show Xo,ao = SpectralDistances.alg2(X,Y,a,b;λ=10, innertol=1e-5, tol=1e-6, printerval=20, inneriters=1000, solver=IPOT, γ=0.01)
@test Xo ≈ mean(Y) rtol=1e-1
@test ao ≈ a rtol = 0.1

scatter(eachrow(reduce(hcat,Y))...)
scatter!(eachrow(X)..., alpha=0.8)
scatter!(eachrow(Xo)..., alpha=0.4)
##

using JuMP, GLPK
function plan(D, P1, P2)
    n = length(P1)
    model = Model(GLPK.Optimizer)
    @variable(model, γ[1:n^2])
    @objective(model, Min, γ'vec(D))
    Γ = reshape(γ,n,n)
    con1 = @constraint(model, con1,  sum(Γ, dims=1)[:] .== P2)
    con2 = @constraint(model, con2,  sum(Γ, dims=2)[:] .== P1)
    con3 = @constraint(model, con3,  γ .>= 0)
    JuMP.optimize!(model)
    if Int(termination_status(model)) != 1
        @error Int(termination_status(model))
    end
    α, β = -dual.(con2)[:], -dual.(con1)[:]
    α .-= mean(α)
    β .-= mean(β)
    reshape(value.(γ),n,n), α, β, -dual.(con3)[:]
end

r3(x) = round.(x, digits=3)
ip(x,y) = x'y/norm(x)/norm(y)

M = SpectralDistances.distmat_euclidean(X,Y[1])

g1,a1,b1 = plan(M,a,b[1]) .|> r3
g2,a2,b2 = sinkhorn_log(M,a,b[1], β=0.0001, iters=50000, printerval=100) .|> r3
g3,a3,b3 = IPOT(M,a,b[1], β=0.01, iters=10000, printerval=1) .|> r3
@test ip(a1,a2) ≈ 1 atol=1e-2
@test ip(a1,a3) ≈ 1 atol=1e-2

@test ip(b1,b2) ≈ 1 atol=1e-2
@test ip(b1,b3) ≈ 1 atol=1e-2


a = ones(k) |> s1
b = [[1,2,3,4] |> s1 for _ in eachindex(Y)]
M = SpectralDistances.distmat_euclidean(Y[1],Y[1])
g1,a1,b1 = plan(M,a,b[1]) .|> r3
g2,a2,b2 = sinkhorn_log(M,a,b[1], β=0.0001, iters=50000, printerval=100) .|> r3
g3,a3,b3 = IPOT(M,a,b[1], β=0.01, iters=10000, printerval=1) .|> r3
@test ip(a1,a2) ≈ 1 atol=1e-1
@test ip(a1,a3) ≈ 1 atol=1e-1

@test ip(b1,b2) ≈ 1 atol=1e-1
@test ip(b1,b3) ≈ 1 atol=1e-1


end

##

@testset "Barycentric coordinates" begin
    @info "Testing Barycentric coordinates"

d = 2
k = 4
S = 4
X0 = [1 1 2 2; 1 2 1 2]

res = map(1:10) do _
    X = [X0[:,randperm(k)] .+ 10rand(d) for _ in 1:S]
    sw = ISA(X, iters=100, printerval=100)
    X̂ = [X[i][:,sw[i]] for i in eachindex(X)]
    bc = mean(X̂)
    @test mean(bc) ≈ mean(mean, X) rtol=0.01
    @test mean(bc, dims=2) ≈ mean(mean.(X, dims=2)) rtol=0.01

    w = rand(S) |> s1
    sw = ISA(X, w, iters=100, printerval=100)
    X̂ = SpectralDistances.barycentric_weighting(X,w,sw)


    a = rand(k) |> s1
    b = rand(k) |> s1

    γ = 10.0


    # α = exp.(α)
    λ0 = rand(S) |> s1
    p = repeat(ones(k) |> s1, 1, S)
    q = rand(k) |> s1

    ql = barycenter(X, λ0)
    ql2 = SpectralDistances.barycenter2(X, λ0, printerval=50, γ=0.1, θ=0.005, iters=400)
    C = [[mean(abs2, x1-x2) for x1 in eachcol(Xi), x2 in eachcol(ql)] for Xi in X]


    bch,λh = SpectralDistances.barycentric_coordinates(X,ql,p,q, γ=γ, L=32)
    scatter(eachrow(reduce(hcat,X))..., lab="X")
    scatter!(eachrow(ql)..., lab="ql")
    scatter!(eachrow(ql2)..., lab="ql2")
    scatter!(eachrow(bch)..., lab="bc")

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
