using Test, SpectralDistances, ControlSystems, Optim
import SpectralDistances: softmax
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
    if isinteractive()
        scatter(eachrow(reduce(hcat,X))...)
        scatter!(eachrow(bc)...)
    end

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
    if isinteractive()
        scatter(eachrow(reduce(hcat,X))...)
        scatter!(eachrow(bc)...)
    end

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
    if isinteractive()
        scatter(eachrow(reduce(hcat,X))...)
        scatter!(eachrow(bc)..., alpha=0.2)
    end

end


@testset "ERD barycenters" begin
    @info "Testing ERD barycenters"

    d = EuclideanRootDistance(domain=Continuous(), weight=unitweight)
    models = [AR(ContinuousRoots([-1])), AR(ContinuousRoots([-3]))]
    bc = @inferred barycenter(d,models)
    @test bc isa AR
    @test roots.(Continuous(), bc)[1] ≈ -2

    models = [AR(ContinuousRoots([-1.0im,+im])), AR(ContinuousRoots([-3.0im,+3im]))]
    bc = @inferred barycenter(d,models)
    @test bc isa AR
    @test roots.(Continuous(), bc) ≈ [-2.0im,+2im]


    d = @inferred EuclideanRootDistance(domain=Continuous())
    models = [AR(ContinuousRoots([-1.0])), AR(ContinuousRoots([-3.0]))]
    bc = @inferred barycenter(d,models)
    @test bc isa AR
    @test roots.(Continuous(), bc)[1] ≈ -2

    models = [AR(ContinuousRoots([-im,+1.0im])), AR(ContinuousRoots([-3.0im,+3im]))]
    bc = barycenter(d,models)
    @test bc isa AR
    @test roots.(Continuous(), bc) ≈ [-2.0im,+2im]

    models = [AR(ContinuousRoots([-im,+im])), AR(ContinuousRoots([-3.0im,+2im]))]
    bc = barycenter(d,models)
    @test bc isa AR
    @test roots.(Continuous(), bc) ≈ [-2im,+1.5im]


    d = EuclideanRootDistance(domain=Continuous(), weight=residueweight)
    models = [AR(ContinuousRoots([-5.0-im,-1+im])), AR(ContinuousRoots([-1.0-im,-1+im]))]
    bc = barycenter(d,models)
    @test bc isa AR
    @test roots.(Continuous(), bc) ≈ [-1.1538-im,-1+im] rtol=0.01

end

##

@testset "alg1 alg2" begin
    @info "Testing alg1 alg2"


    Y = [[1. 1], [2. 2], [3. 3]] .+ Ref([0 0.001])
    X = [1.1 1.01]
    a = ones(2) ./ 2
    b = [ones(2) ./2 for _ in eachindex(Y)]
    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/3, θ=0.5, printerval=190, γ=0.01, iters=50000, inneriters=50000, uniform=true) # TODO: there are type instabilities in this function but Cthulhu is not working at the time of writing.
    @test Xo ≈ [2 2] rtol=1e-2

    X = [0. 0.1]
    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/3,θ=0.5, γ=0.01, inneriters=50000, uniform=true)
    @test Xo ≈ [2 2] rtol=1e-2

    X = [1. 3.]
    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/10.0, θ=0.5, γ=0.01, inneriters=50000, uniform=true)
    @test Xo ≈ [2 2] rtol=1e-2


    X = [0. 4.]
    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/3, θ=0.5, γ=0.01, inneriters=50000, uniform=true)
    @test Xo ≈ [2 2] rtol=1e-2


    Y = [[1. 2], [2. 3], [3. 4]]
    X = [2.0 3]
    a = ones(2) |> s1
    b = [ones(2) ./2 for _ in eachindex(Y)]
    a1 = SpectralDistances.alg1(X,Y,a,b;β=1/3, tol=1e-5)
    @test a1[1] == a1[2]

    X = [2.1 3]
    a1 = SpectralDistances.alg1(X,Y,a,b;β=1/3, tol=1e-3, printerval=1)
    @test a1[1] < a1[2]
    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/3, θ=0.5, printerval=100, iters=500, tol=1e-3, γ=0.01, innertol=1e-3, inneriters=50000, uniform=true)
    @test Xo ≈ [2 3] rtol=5e-1
    # @test ao ≈ b[1] rtol=0.1




    X = [0. 0.1]
    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/3, θ=0.5, printerval=100, innertol=1e-3, inneriters=50000, tol=1e-3, γ=0.1, uniform=true)
    @test Xo ≈ [2 3] rtol=5e-1

    X = [1. 3.]
    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/3, θ=0.5, printerval=100, innertol=1e-4, inneriters=100, tol=1e-5, uniform=true)
    @test Xo ≈ [2 3] rtol=5e-1


    X = [0. 4.]
    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/2, θ=0.5, uniform=true)
    @test Xo ≈ [2 3] rtol=5e-1

    Y = [[1. 1], [2. 2], [3. 3]]
    X = [1.1 1.01]
    a = ones(2) |> s1
    b = [[0.2, 0.8] for _ in eachindex(Y)]
    @test SpectralDistances.alg1(Y[2],Y,b[1],b;β=1/3, printerval=1) == b[1]


    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/3, θ=0.5, printerval=100)
    @test Xo ≈ [2 2] rtol=5e-1
    @test_broken ao ≈ b[1] rtol=1e-2


    ##
    Random.seed!(0)
    d = 2
    k = 4
    Y0 = 0.1*[1 1 2 2; 1 2 1 2]
    Y = [Y0[:,randperm(k)] .+ 1rand(d) for _ in 1:4]
    X = mean(Y) .+ Y0
    a = ones(k) |> s1
    b = [ones(k) |> s1 for _ in eachindex(Y)]
    a1 = SpectralDistances.alg1(X,Y,a,b;β=1/100.0, printerval=100)
    # @test a1 ≈ a rtol=0.01


    Xo,ao = SpectralDistances.alg2(X,Y,a,b;β=1/2, innertol=1e-5, tol=1e-6, printerval=20, inneriters=100000, solver=IPOT, γ=0.01, uniform=true)
    @test Xo ≈ mean(Y) rtol=1e-1
    @test ao ≈ a rtol = 0.1

    if isinteractive()
        scatter(eachrow(reduce(hcat,Y))...)
        scatter!(eachrow(X)..., alpha=0.8)
        scatter!(eachrow(Xo)..., alpha=0.4)
    end
    ##

    using JuMP, GLPK

    r6(x) = x#round.(x, digits=6)
    ip(x,y) = x'y/norm(x)/norm(y)

    M = @inferred SpectralDistances.distmat_euclidean(X,Y[1])
    M2 = similar(M)
    @inferred SpectralDistances.distmat_euclidean!(M2,X,Y[1])
    @test M == M2

    g1,a1,b1 = SpectralDistances.ot_jump(M,a,b[1])
    g2,a2,b2 = @inferred sinkhorn_log!(M,a,b[1], β=0.0001, iters=1000000, tol=1e-7)
    g3,a3,b3 = @inferred IPOT(M,a,b[1], β=0.5, iters=100000)
    @test isapprox(ip(a1,a2), 1, atol=1e-1) || isapprox(ip(a1,a3), 1, atol=1e-1)
    @test isapprox(ip(b1,b2), 1, atol=1e-1) || isapprox(ip(b1,b3), 1, atol=1e-1)


    a = ones(k) |> s1
    b = [[1,2,3,4] |> s1 for _ in eachindex(Y)]
    M = SpectralDistances.distmat_euclidean(X,Y[1])
    g1,a1,b1 = SpectralDistances.ot_jump(M,a,b[1]) .|> r6
    g2,a2,b2 = sinkhorn_log(M,a,b[1], β=0.001, iters=50000, printerval=5000, tol=1e-9) .|> r6
    g3,a3,b3 = IPOT(M,a,b[1], β=0.5, iters=10000, printerval=5000, tol=1e-9) .|> r6
    @test isapprox(ip(a1,a2), 1, atol=1e-1) || isapprox(ip(a1,a3), 1, atol=1e-1)
    @test isapprox(ip(b1,b2), 1, atol=1e-1) || isapprox(ip(b1,b3), 1, atol=1e-1)


end

##

@testset "Barycentric coordinates" begin
    @info "Testing Barycentric coordinates"
    Random.seed!(0)
    d = 2
    k = 4
    S = 3
    X0 = [1 1 2 2; 1 2 1 2]

    res = map(1:10) do _
        X = [X0[:,randperm(k)] .+ 0.01.*randn.() .+ 10rand(d) for _ in 1:S]
        λ0 = randn(S) |> SpectralDistances.softmax
        p = [ones(k) |> s1 for _ in 1:S]
        q = ones(k) |> s1

        β = 0.1
        ql = barycenter(X, λ0, inneriters=200000, tol=1e-9, innertol=1e-8, β=β, solver=sinkhorn_log!)

        λh = barycentric_coordinates(X,ql,p,q, β=β, solver=sinkhorn_log!, robust=false, tol=1e-10)
        if isinteractive()
            scatter(eachrow(reduce(hcat,X))..., lab="X")
            scatter!(eachrow(ql)..., lab="initial bc")
        end
        costfun = λ -> SpectralDistances.sinkhorn_cost(X, ql, p, q, λ;
            solver = sinkhorn_log!,
            tol    = 1e-10,
            β      = β)


        isinteractive() && @show norm(λ0-λh)
    end
    @test median(res) < 0.01
    # mean(norm(s1(rand(4)) - s1(rand(4))) for _ in 1:10000)


    res = map(1:5) do _
        X = [X0[:,randperm(k)] .+ 10rand(d) for _ in 1:S]
        p = [randn(k) |> softmax for _ in 1:S]

        ql = X[1] .+ 0.01 .* randn.()
        q = p[1]
        λ0 = [1,0,0]

        β = 1/10.0
        λh = barycentric_coordinates(X,ql,p,q, β=β, solver=sinkhorn_log!, robust=false)
        if isinteractive()
            scatter(eachrow(reduce(hcat,X))..., lab="X")
            scatter!(eachrow(ql)..., lab="ql")
        end

        isinteractive() && @show norm(λ0-λh)
    end
    @test median(res) < 0.01



end

# using DoubleFloats
# df(v) = reduce(hcat, v)
# v = [[Double64(1) for _ in 1:2] for _ in 1:2]
# Test.@inferred df(v)

# v = randn(ComplexF64, 3)
# tv(v) = @views [real(v[1:2]); imag(v[1:2])]
# @inferred tv(v)


@testset "barycentric coordinates with models" begin
    @info "Testing barycentric coordinates with models"
    # The example below is likely to mix up the two lightly damped poles with euclidean root distance, making the bc poles end up inbetween the two clusters. The SRD should fix it

    models = @inferred examplemodels(10)
    d = @inferred EuclideanRootDistance(domain=SpectralDistances.Continuous(),p=2)
    Xe = @inferred barycenter(d, models)
    # models = change_precision.(Float64, models)
    # Xe = change_precision(Float64, Xe)
    λ = barycentric_coordinates(d,models, Xe)
    @test λ ≈ s1(ones(length(λ))) atol=0.02

    λ = barycentric_coordinates(d,models, models[1], verbose=false, iters=5000, α0=20)
    isinteractive() && bar(λ)
    @test λ[1] > 0.5
    @test sum(λ) ≈ 1

    G = tf.(models)
    if isinteractive()
        plot()
        pzmap!.(G)
        pzmap!(tf(Xe), m=:c, title="Barycenter EuclideanRootDistance")
    end
    ##
    d = @inferred OptimalTransportRootDistance(domain=SpectralDistances.Continuous(),p=2, weight=residueweight, β=0.01)
    Xe = barycenter(d, models, solver=sinkhorn_log!)

    G = tf.(models)
    if isinteractive()
        plot()
        pzmap!.(G)
        pzmap!(tf(Xe), m=:c, title="Barycenter OptimalTransportRootDistance", lab="BC")
    end

options = Optim.Options(
    store_trace = true,
    show_trace = false,
    show_every = 1,
    iterations = 50,
    allow_f_increases = true,
    time_limit = 100,
    x_tol = 1e-7,
    f_tol = 1e-7,
    g_tol = 1e-7,
    f_calls_limit = 0,
    g_calls_limit = 0,
)


    using Optim
    method = LBFGS()
    method=ParticleSwarm()
    # d = OptimalTransportRootDistance(domain=SpectralDistances.Continuous(),p=2, weight=unitweight, β=0.1)
    λ = @inferred barycentric_coordinates(d,models,Xe, method, options=options, solver=sinkhorn_log!, robust=true, uniform=true, tol=1e-6)
    isinteractive() && bar(λ)

    @test median(λ) > 0.02

    G = tf.(models)
    if isinteractive()
        plot()
        pzmap!.(G, lab="")
        pzmap!(tf(Xe), m=:c, title="Barycenter OptimalTransportRootDistance", lab="BC")
        pzmap!(G[argmax(λ)], m=:c, lab="Largest bc coord", legend=true)
        # It's okay if the green dot do not match the blue exactly, there are limited models to choose from.
    end
end


@testset "kbarycenters" begin
    @info "Testing kbarycenters"

    using Clustering


    d = 2
    k = 4
    X0 = 0.1*[1 1 2 2; 1 2 1 2]
    X1 = 0.1*[1 2 3 4; 1 2 1 2]
    X00 = [X0[:,randperm(k)] .+ 0.01 .*randn.() .+ 1rand(d) for _ in 1:4]
    X11 = [X1[:,randperm(k)] .+ 0.01 .*randn.() .+ 1rand(d) .+ 2 for _ in 1:4]
    X = [X00; X11]

    p = [ones(k) |> s1 for _ in eachindex(X)]

    ##
    Q = SpectralDistances.kbarycenters(X,p,2, iters=7, solver=sinkhorn_log!, uniform=true, seed=:rand, tol=1e-3, innertol=1e-3, β=0.2, inneriters=50000, verbose=true)

    if isinteractive()
        scatter(eachrow(reduce(hcat,X))..., markerstrokewidth=false)
        scatter!(eachrow(reduce(hcat,Q.barycenters))..., markerstrokewidth=false, legend=false)
    end

    ##

    C = @inferred SpectralDistances.distmat_euclidean(X[1], X[5])
    c1 = @inferred SpectralDistances.kwcostfun(C,X[1], X[5],p[1],p[1],solver=IPOT, tol=1e-5, β=0.5, printerval=1, iters=200)
    c2 = @inferred SpectralDistances.kwcostfun(C,X[1], X[5],p[1],p[1],solver=sinkhorn_log, β=0.01)
    @test c1 ≈ c2 rtol=0.01

end


# d = 2
# k = 4
# S = 4
# X0 = [1 1 2 2; 1 2 1 2]
#
# using JuMP, GLPK
# ##
# X = [X0[:,randperm(k)] .+ 10rand(d) for _ in 1:S]
#
# λ0 = [0.9; 0.1ones(S-1)] |> s1
# ql = barycenter(X, λ0)
#
# ql2 = SpectralDistances.barycenter2(copy(X), λ0, printerval=10, γ=1, θ=0.5, iters=400, tol=1e-6, solver=SpectralDistances.sinkhorn_log)
#
# scatter(eachrow(reduce(hcat,X[2:end]))..., lab="X")
# scatter!(eachrow(X[1])..., lab="X1")
# # scatter!(eachrow(ql)..., lab="ql")
# scatter!(eachrow(ql2)..., lab="ql2")
## Measures with nonuniform weights, the bc should be pulled to atom 1 and 4 in the first measure. The trick seems to be to run ISA without weights and then use the weights to form the estimate

# d = 2
# k = 4
# X0 = [1 1 2 2; 1 2 1 2]
# X = [X0 .+ 0.3rand(d,k) .+ 0.020rand(d) for _ in 1:6]
# w = [ones(1,k) for _ in 1:length(X)]
# w[1][1] = 100
# w[1][4] = 100
# for i = 2:length(X)
#     w[i][1] = 0.01
#     w[i][4] = 0.01
# end
# w = s1.(w)
# W = reduce(vcat,w)
# W ./= sum(W,dims=1)
# w2 = [W[i,:]' for i in 1:length(X)]
# S = ISA(X, iters=1000, printerval=10)
# X̂ = [w2[i].*X[i][:,S[i]] for i in eachindex(X)]
# bc = sum(X̂)
# # @test mean(bc) ≈ mean(X[1]) rtol=0.1
# scatter(eachrow(reduce(hcat,X))...)
# scatter!([X[1][1,:]],[X[1][2,:]])
# scatter!(eachrow(bc)..., m=:square, legend=false, alpha=0.4)
# =============================================================================
## Tests for sinkhorn_diff, it does not seem to produce the correct gradient ==============
# C = [[mean(abs2, x1-x2) for x1 in eachcol(Xi), x2 in eachcol(ql)] for Xi in X]
# λl = SpectralDistances.softmax(1e-3randn(S))
#
# c,Pl,w = sinkhorn_diff(X,ql,p,q,C,λl, β=β, L=32, solver=sinkhorn_log)
# display(w)
#
# using ForwardDiff
# g = ForwardDiff.gradient(λl->SpectralDistances.sinkhorn_cost(X,ql,p,q,C,λl, β=β, L=32, solver=sinkhorn_log), λl)
# display(g)
# @test w ≈ g
## =================================================

# @test mean(getindex.(res,2)) >= 0.9
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
# distance = OptimalTransportRootDistance(domain=Continuous(),p=2, weight=unitweight)
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
