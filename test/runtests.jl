@info "Running tests"
using SpectralDistances# Distributions
using Test, LinearAlgebra, Statistics, Random, ControlSystems, InteractiveUtils # For subtypes
using DSP, Distances, DoubleFloats

using SpectralDistances: ngradient, nhessian, njacobian, polyconv

Random.seed!(1)


# using Zygote
# function jacobian(m,x)
#     y  = m(x)
#     k  = length(y)
#     n  = length(x)
#     J  = Matrix{eltype(x)}(undef,k,n)
#     for i = 1:k
#         g = Zygote.gradient(x->m(x)[i], x)[1]
#         J[i,:] .= g
#     end
#     J
# end
# using ForwardDiff, FiniteDifferences
#
# function hessian(f, a)
#     H = jacobian(f', a)
# end
#
#
# function hessian_fd_reverse(f,a)
#     fd = central_fdm(9,1)
#     FiniteDifferences.jacobian(fd, f', a)
# end
#
# function nhessian_reverse(f,a)
#     njacobian(f', a)
# end
#
# a = randn(11)
# a[1] = 1
# using DoubleFloats
# m = AR(ContinuousRoots([-Float64(0.1)+im, -Float64(0.1)-im]))
# m2 = AR(ContinuousRoots([-Float64(0.11)+im, -Float64(0.12)-im]))
# a = Vector(m.ac)
# dist = CoefficientDistance(domain=Discrete(),distance=CosineDist())
# dist = EuclideanRootDistance(domain=Continuous(), p=2, weight=residueweight)
# dist = EuclideanRootDistance(domain=Continuous(), p=2, weight=unitweight)
# dist = OptimalTransportRootDistance(domain=Continuous(), p=2, iters=500, β=0.5)
# function forward(b)
#     real(dist(a, b))
# end
# forward(a)
# forward'(a)
# Zygote.refresh()
# dist(m,m2) |> typeof
# g = Zygote.gradient(Float64 ∘ dist, m.pc, m2.pc)
#
# mdist = ModelDistance(TLS(na=4), dist)
# t = 0:0.01:100
# x1 = sin.(70 .*t) .+ sin.(150 .* t) .+ 0.1 .*randn.()
# # x1 = filt(ones(10),[10],randn(10000))
# x2 = randn(10000)
# mdist(x1,x2)
#
# using TotalLeastSquares
# function TotalLeastSquares.tls(A::AbstractArray,y::AbstractArray)
#     AA  = [A y]
#     U,S,V   = svd(AA)
#     n   = size(A,2)
#     V21 = V[1:n,n+1:end]
#     V22 = V[n+1:end,n+1:end]
#     x   = -V21/V22
# end
#
# shift1(x) = x .+ x[2]
# g1,g2 = Zygote.gradient(abs ∘ mdist, x1, x2)
# gradfun(xo,x2) = xo .= Zygote.gradient(abs ∘ mdist, x2, x1)[1]
# gradfun(x2) = Zygote.gradient(abs ∘ mdist, x2, x1)[1]
# gradfun(x2)
# # x1 = filt(ones(10),[10],randn(10000))
# # x2 = randn(10000)
# for i = 1:1000
#     if i % 40 == 0
#         P = welch_pgram(x1)
#         plot(P)
#         P = welch_pgram(x2)
#         plot!(P) |> display
#     end
#     @show mdist(x1,x2)
#     x2 .-= 30 .* gradfun(x2)
# end
# using Optim
# res = Optim.optimize(x2->mdist(x2,x1), gradfun, x2, LBFGS(), Optim.Options(store_trace=true, show_trace=true, show_every=1, iterations=100, allow_f_increases=true, time_limit=100, x_tol=0, f_tol=0, g_tol=1e-8, f_calls_limit=100, g_calls_limit=0))
# x2 = res.minimizer

# ngradient(Float64 ∘ dist, m, m2)


# Zygote.refresh()
#
# # NOTE: Gradient at a should be almost zero as it is the minimum, everything else is incorrect. Hessian should be pos.def.
# @time forward'(a)
# @time H = hessian_fd_reverse(forward, a)
# @time H = nhessian_reverse(forward, a)
# c∇ = central_fdm(3,1)
#
# grad(c∇, forward, (a))
# ngradient(forward, copy(a))
#
#
# @btime H = hessian_fd_reverse($(forward), $a)
#
# eigvals(H[2:end,2:end])
#
# hessian(forward, a)




@testset "SpectralDistances.jl" begin


    @testset "time" begin
        @info "Testing time"
        include("test_time.jl")
    end

    @testset "Solvers" begin
        @info "Testing Solvers"

        C = Float64.(Matrix(I(2)))
        a = [1., 0]
        b = [0, 1.]
        Γ,u,v = sinkhorn(C,a,b,β=0.01)
        @test Γ ≈ [0 1; 0 0]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = sinkhorn(C,a,b,β=0.1)
        @test Γ ≈ [0 0.5; 0 0.5]

        a = [1., 0]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = sinkhorn_log(C,a,b,β=0.01)
        @test Γ ≈ [0 1; 0 0]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = sinkhorn_log(C,a,b,β=0.01)
        @test Γ ≈ [0 0.5; 0 0.5]


        a = [1., 0]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = IPOT(C,a,b,β=1)
        @test Γ ≈ [0 1; 0 0]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = SpectralDistances.IPOT(C,a,b)
        @test Γ ≈ [0 0.5; 0 0.5]

    end


    # true && get(ENV, "TRAVIS_BRANCH", nothing) == nothing && @testset "gradients" begin
    @testset "gradients" begin
        @info "Testing gradients"
        using FiniteDifferences
        using SpectralDistances: njacobian, ngradient, nhessian
        using Zygote
        using SpectralDistances: getARXregressor, getARregressor
        y = (randn(10))
        u = (randn(10))
        @test ngradient((y)->sum(getARXregressor(y,u,2,2)[2]), y) == Zygote.gradient((y)->sum(getARXregressor(y,u,2,2)[2]), y)[1]
        @test ngradient((y)->sum(getARXregressor(y,u,2,2)[1]), y) == Zygote.gradient((y)->sum(getARXregressor(y,u,2,2)[1]), y)[1]
        @test ngradient((y)->sum(getARXregressor(y,u,5,2)[2]), y) == Zygote.gradient((y)->sum(getARXregressor(y,u,5,2)[2]), y)[1]
        @test ngradient((y)->sum(getARXregressor(y,u,5,2)[1]), y) == Zygote.gradient((y)->sum(getARXregressor(y,u,5,2)[1]), y)[1]

        @test ngradient((y)->sum(getARXregressor(y,u,2,1)[2]), y) == Zygote.gradient((y)->sum(getARXregressor(y,u,2,1)[2]), y)[1]
        @test ngradient((y)->sum(getARXregressor(y,u,2,1)[1]), y) == Zygote.gradient((y)->sum(getARXregressor(y,u,2,1)[1]), y)[1]
        @test ngradient((y)->sum(getARXregressor(y,u,5,1)[2]), y) == Zygote.gradient((y)->sum(getARXregressor(y,u,5,1)[2]), y)[1]
        @test ngradient((y)->sum(getARXregressor(y,u,5,1)[1]), y) == Zygote.gradient((y)->sum(getARXregressor(y,u,5,1)[1]), y)[1]



        y = (randn(10))
        @test ngradient((y)->sum(getARregressor(y,2)[2]), y) == Zygote.gradient((y)->sum(getARregressor(y,2)[2]), y)[1]
        @test ngradient((y)->sum(getARregressor(y,2)[1]), y) == Zygote.gradient((y)->sum(getARregressor(y,2)[1]), y)[1]
        @test ngradient((y)->sum(getARregressor(y,5)[2]), y) == Zygote.gradient((y)->sum(getARregressor(y,5)[2]), y)[1]
        @test ngradient((y)->sum(getARregressor(y,5)[1]), y) == Zygote.gradient((y)->sum(getARregressor(y,5)[1]), y)[1]

        a,b = randn(3), randn(4)
        @test ngradient(a->sum(SpectralDistances.polyconv(a,b)), a) ≈ Zygote.gradient(a->sum(SpectralDistances.polyconv(a,b)),a)[1] rtol=1e-3
        @test ngradient(b->sum(SpectralDistances.polyconv(a,b)), b) ≈ Zygote.gradient(b->sum(SpectralDistances.polyconv(a,b)),b)[1] rtol=1e-3



        @test_skip let Gc = tf(1,[1,1,1,1])
            w = c2d(Gc,1).matrix[1] |> ControlSystems.denvec
            @test d2c(w) ≈ pole(Gc)
        end

        # y = randn(5000)
        # fm = PLR(na=40, nc=2)
        # @test_skip Zygote.gradient(y->sum(fm(y)[1]), y)[1] ≈ ForwardDiff.gradient(y->sum(fm(y)[1]), y)
        # @test_skip Zygote.gradient(y->sum(fm(y)[2]), y)[1] ≈ ForwardDiff.gradient(y->sum(fm(y)[2]), y)


        p = [1.,1,1]
        # @btime riroots(p)
        fd = central_fdm(3, 1)
        @test Zygote.gradient(p) do p
            r = roots(p)
            sum(abs2, r)
        end[1][1:end-1] ≈ FiniteDifferences.grad(fd, p->begin
            r = roots(p)
            sum(abs2, r)
        end, p)[1][1:end-1]

        fm = TLS(na=4)
        y = randn(50)
        sum(abs2, fm(y).pc)
        @test_broken Zygote.gradient(y) do y
            sum(abs2, fitmodel(fm,y,true).pc)
        end




        a = randn(5)
        a[end] = 1
        f = x -> sum(abs2(r) for r in roots(x))
        G = Zygote.gradient(f, a)[1]


        fdm = central_fdm(5,1)
        @test G[1:end-1] ≈ FiniteDifferences.grad(fdm, f, a)[1][1:end-1]


        fd = central_fdm(5,1)
        a = randn(30)
        a[1] = 1
        r = roots(reverse(a)) |> ContinuousRoots
        residues(a,1,r)
        f = a -> sum(abs2, residues(a,1,r))
        g = a -> real(residues(complex.(a),1,r)[2])
        @test sum(abs, f'(complex.(a))[2:end] - grad(fd,f,a)[1][2:end]) < sqrt(eps())
        @test sum(abs, g'((a))[2:end] - grad(fd,g,a)[1][2:end]) < sqrt(eps())

        fd = central_fdm(7,1)
        a = randn(30)
        a[1] = 1
        residues(a,1)
        f = a -> sum(abs2, residues(a,1))
        g = a -> real(residues(complex.(a),1)[2])
        @test sum(abs, f'(complex.(a))[2:end] - grad(fd,f,a)[1][2:end]) < 1e-5
        @test_skip sum(abs, g'((a))[2:end] - grad(fd,g,a)[1][2:end]) < sqrt(eps()) # Not robust

        @testset "Numerical curvature" begin
            @info "Testing Numerical curvature"
            a = randn(11)
            a[1] = 1
            using DoubleFloats
            m = AR(ContinuousRoots([-Double64(0.1)+im, -Double64(0.1)-im]))
            a = Vector(m.ac)
            for dist in [EuclideanRootDistance(domain=Continuous(), p=1, weight=residueweight),
                        OptimalTransportRootDistance(domain=Continuous(), p=2, β=0.01)]
                    H = SpectralDistances.curvature(dist, a)
                    @test all(>(0) ∘ real, eigvals(H[2:end,2:end]))
                    @test all(==(0) ∘ imag, eigvals(H[2:end,2:end]))
            end

        end

        # Zygote segfaults here
        # @testset "Sinkhorn zygote" begin
        #     @info "testing sinkhorn zygote"
        #     function sinkdist(D,a,b)
        #         ai = s1(a)
        #         bi = s1(a+b)
        #         P,u,v = sinkhorn(D,ai,bi, iters=1000, β=0.1)
        #         sum(P.*D)
        #     end
        #     a,b = abs.(randn(6)),abs.(randn(6))
        #     D = SpectralDistances.distmat_euclidean(1:length(a), 1:length(a))
        #     dD, da, db = Zygote.gradient(sinkdist, D,a,b)
        #     @test n1(ForwardDiff.gradient(a->sinkdist(D,a,b), a))'n1(da) > 0.9
        # end


    end

    a = -5:-1
    @test SpectralDistances.roots2poly(a) ≈ SpectralDistances.roots2poly_zygote(a)


    @testset "Energy" begin
        @info "testing Energy"
        for σ² = [0.1, 1., 2., 3]
            m = AR(ContinuousRoots([-1.]), σ²)
            e = spectralenergy(Continuous(),m)
            @test e ≈ σ²
            x = sqrt(σ²)randn(10000)
            m = TLS(na=10)(x)
            @test spectralenergy(Continuous(),m) ≈ σ² atol=0.05
        end
        # y = filt(numvec(Discrete(),m), denvec(Discrete(),m), x)
        # @test var(y) ≈ var(x)
    end

    @testset "Model estimation" begin
        y = sin.(0:0.1:100)
        fm = LS(na=2, λ=0)
        m = fm(y)
        @test imag.(m.pc) ≈ [-0.1, 0.1] rtol=1e-4
        mc = SpectralDistances.change_precision(Float32, m)
        @test m ≈ mc
        @test eltype(mc.a) == Float32

        fm = TLS(na=2)
        m = fm(y)
        @test imag.(m.pc) ≈ [-0.1, 0.1] rtol=1e-4

        fm = IRLS(na=2)
        m = fm(y)
        @test_broken imag.(m.pc) ≈ [-0.1, 0.1] rtol=1e-4

        y = sin.(0:0.1:1000) .+ 0.01 .*randn.()
        fm = PLR(na=2, nc=1)
        m = fm(y)
        @test imag.(log(m.p)) ≈ [-0.1, 0.1] rtol=1e-3


        y = sin.(0:0.1:100) .+ sin.(2 .* (0:0.1:100) .+ 0.3)
        fm = LS(na=4, λ=0)
        m = fm(y)
        @test imag.(m.pc) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=1e-1

        fm = TLS(na=4)
        m = fm(y)
        @test spectralenergy(Continuous(), m) ≈ var(y) rtol=1e-3
        @test imag.(m.pc) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=1e-4

        fm = IRLS(na=4)
        m = fm(y)
        @test spectralenergy(Continuous(), m) ≈ var(y) rtol=1e-3
        @test_broken imag.(m.pc) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=1e-4

        y = sin.(0:0.1:1000) .+ sin.(2 .* (0:0.1:1000)) .+ 0.001 .*randn.()
        fm = PLR(na=4, nc=1, λ=0.0)
        m = fm(y)
        @test imag.(m.pc) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=0.6

    end




@testset "modeldistance" begin
    t = 1:300
    ϵ = 1e-7
    for fitmethod in [LS, TLS]
        @info "Testing fitmethod $(string(fitmethod))"
        ls_loss = ModelDistance(fitmethod(na=2), EuclideanRootDistance(domain=Discrete()))
        @test ls_loss(sin.(t), sin.(t)) < ϵ
        @test ls_loss(sin.(t), -sin.(t)) < ϵ # phase flip invariance
        @test ls_loss(sin.(t), sin.(t .+ 1)) < ϵ # phase shift invariance
        @test ls_loss(sin.(t), sin.(t .+ 0.1)) < ϵ # phase shift invariance
        @test ls_loss(10sin.(t), sin.(t .+ 1)) < ϵ # amplitude invariance
        @test ls_loss(sin.(t), sin.(1.1 .* t)) < 0.2 # small frequency shifts gives small errors
        @test ls_loss(sin.(0.1t), sin.(1.1 .* 0.1t)) < 0.1 # small frequency shifts gives small errors

        @test ls_loss(sin.(t), sin.(1.1 .* t)) ≈ ls_loss(sin.(0.1t), sin.(0.2t)) rtol=1e-2  # frequency shifts of relative size should result in the same error, probably only true for p=1
        ls_loss = ModelDistance(fitmethod(na=10), OptimalTransportRootDistance(domain=Discrete()))
        @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < 0.1 # Filtered through same filter, this test is very non-robust for TLS
        @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < ls_loss(filtfilt(ones(4),[4], randn(1000)), filtfilt(ones(10),[10], randn(1000))) # Filtered through different filters, this test is not robust
    end
    @testset "PLR" begin
        fitmethod = PLR
        @info "Testing fitmethod $(string(fitmethod))"
        t = 1:1000 .+ 0.01 .* randn.()
        ϵ = 0.01
        ls_loss = ModelDistance(fitmethod(na=2,nc=1), EuclideanRootDistance(domain=Discrete()))
        @test ls_loss(sin.(t), sin.(t)) < ϵ
        @test ls_loss(sin.(t), -sin.(t)) < ϵ # phase flip invariance
        @test ls_loss(sin.(t), sin.(t .+ 1)) < ϵ # phase shift invariance
        @test ls_loss(sin.(t), sin.(t .+ 0.1)) < ϵ # phase shift invariance
        @test ls_loss(10sin.(t), sin.(t .+ 1)) < ϵ # amplitude invariance
        @test ls_loss(sin.(t), sin.(1.1 .* t)) < 0.2 # small frequency shifts gives small errors
        @test ls_loss(sin.(0.1t), sin.(1.1 .* 0.1t)) < 0.1 # small frequency shifts gives small errors

        @test ls_loss(sin.(t), sin.(1.1 .* t)) ≈ ls_loss(sin.(0.1t), sin.(0.2t)) rtol=1e-3  # frequency shifts of relative size should result in the same error, probably only true for p=1
        ls_loss = ModelDistance(fitmethod(na=10,nc=2), OptimalTransportRootDistance(domain=Discrete()))
        @test ls_loss(randn(1000), randn(1000)) > 0.05
        # @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < 1 # Filtered through same filter, this test is very non-robust for TLS
        # @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < ls_loss(filtfilt(ones(4),[4], randn(1000)), filtfilt(ones(10),[10], randn(1000))) # Filtered through different filters, this test is not robust
    end
end


@testset "discrete_grid_transportplan" begin
    x = [1.,0,0]
    y = [0,0.5,0.5]

    g = SpectralDistances.discrete_grid_transportplan(x,y)
    @test sum(g,dims=1)[:] == y
    @test sum(g,dims=2)[:] == x

    # test robustness for long vectors
    x = s1(rand(1000))
    y = s1(rand(1000))
    g = SpectralDistances.discrete_grid_transportplan(x,y)
    @test sum(g,dims=1)[:] ≈ y
    @test sum(g,dims=2)[:] ≈ x
    g = SpectralDistances.discrete_grid_transportplan(y,x)
    @test sum(g,dims=1)[:] ≈ x
    @test sum(g,dims=2)[:] ≈ y

    # test exception for unequal masses
    x = s1(rand(Float32,1000))
    y = rand(Float32,1000)
    @test_throws ErrorException SpectralDistances.discrete_grid_transportplan(x,y)
    @test_throws ErrorException SpectralDistances.discrete_grid_transportplan(y,x)

end


# p = randn(20)
# @btime Zygote.gradient(p) do p
#     r = riroots(p)
#     sum([r[1]; r[2]])
# end
#
#
# jacobian(fd, p->begin
#     r = riroots(p)
#     sum([r[1]; r[2]])
# end, p)

@testset "eigenvalue manipulations" begin
    @test SpectralDistances.reflectd(2) ≈ 0.5 + 0.0im
    @test SpectralDistances.reflectd(complex(0,2)) ≈ 0 + 0.5im
    @test SpectralDistances.reflectd(complex(0,-2)) ≈ 0 - 0.5im

    e = roots(randn(7))
    ed = DiscreteRoots(e)
    @test real(e) == real.(e)
    @test real(ed) == real.(ed)
    @test issorted(ed, by=angle)
    @test all(<(1) ∘ abs, ed)
    ec = log(ed)
    @test issorted(ec, by=imag)
    @test all(<(0) ∘ real, ec)
    @test domain_transform(Continuous(), ed) isa ContinuousRoots
    @test domain_transform(Continuous(), ed) == log.(ed) == ContinuousRoots(ed)
    @test domain_transform(Discrete(), ed) == ed
    @test domain(ed) isa Discrete
    @test domain(ContinuousRoots(ed)) isa Continuous
    @test log(ed) isa ContinuousRoots

    @test all(<(1) ∘ abs, reflect(ed))
    @test all(<(0) ∘ real, reflect(ec))


    @test SpectralDistances.determine_domain(0.1randn(10)) isa Discrete
    @test SpectralDistances.determine_domain(randn(10).-4) isa Continuous
    @test_throws Exception SpectralDistances.determine_domain(0.1randn(10).-0.3)


    @testset "weight functions" begin
        @info "Testing weight functions"
        r = ContinuousRoots(randn(ComplexF64, 4))
        m = AR(r, 2.0)
        @test sum(unitweight(r)) == 1
        @test sum(unitweight(m)) == 1
        @test length(unitweight(r)) == 4
        @test length(unitweight(m)) == 4

        @test m.b^2*sum(residueweight(r)) ≈ sum(residueweight(m)) # Calling residueweight with a model should mutiply energy by b²
        @test length(residueweight(r)) == 4
        @test length(residueweight(m)) == 4
    end


end

@testset "ControlSystems interoperability" begin
    @infor "Testing ControlSystems interoperability"
    m = AR(ContinuousRoots([-1]))
    g = tf(1,[1.,1])
    @test tf(m) == g
    @test m*g == g*g
    @test denvec(Continuous(), m) == denvec(g)[1]
    @test numvec(Continuous(), m) == numvec(g)[1]
    @test pole(Continuous(), m) == pole(g)
    @test all(ControlSystems.bode(m) .≈ bode(g))
    @test all(ControlSystems.nyquist(m) .≈ nyquist(g))
    @test ControlSystems.freqresp(m, exp10.(LinRange(-1, 1, 10))) ≈ freqresp(g, exp10.(LinRange(-1, 1, 10)))
    @test all(ControlSystems.step(m, 10) .≈ step(g, 10))

    bodeplot(m)
    pzmap(m)
    nyquistplot(m)

end

@testset "polynomial acrobatics" begin
    a = randn(5)
    b = randn(5)
    @test polyconv(a,b) ≈ DSP.conv(a,b)

    a = randn(5)
    b = randn(10)
    @test polyconv(a,b) ≈ DSP.conv(a,b)
    @test polyconv(b,a) ≈ DSP.conv(b,a)

    a[1] = 1
    @test roots2poly(roots(reverse(a))) ≈ a


end

@testset "preprocess roots with residue weight" begin
    rd  = EuclideanRootDistance(domain=Continuous(), weight=residueweight)
    m = AR([1., -0.1])
    @test rd(m,m) == 0
    @test SpectralDistances.preprocess_roots(rd, m)[] ≈ -2.3025850929940455
end

@testset "residues and roots" begin

    # @test inv(1:3) == 1:3
    # @test inv([1,1,2]) == [1,1,3]
    # @test inv([1,2,2]) == [1,1,2]

    a = randn(5); a[1]=1;a
    r = roots(reverse(a))
    @test roots2poly(r) ≈ a


    b,a = 1,0.1randn(5)
    a[1] = 1
    r   = roots(reverse(a))
    G   = tf(b,a)
    w   = im
    F = evalfr(G,w)[]
    a2 = roots2poly(r)
    @assert a ≈ a2
    roots(reverse(a2))
    @test sum(r) do r
        powers = length(a)-1:-1:1
        ap     = powers.*a[1:end-1]
        G      = tf(b,ap)
        evalfr(G,r)[] *1/(w - r)
    end ≈ F

    res = residues(a,1)
    @test sum(eachindex(r)) do i
        res[i]/(w-r[i])
    end ≈ F

    @test prod(eachindex(r)) do i
        1/(w-r[i])
    end ≈ F

    n = 4
    a1 = randn(n+1); a1[1] = 1
    r1 = SpectralDistances.reflectc.(roots(reverse(a1)))
    r1 = complex.(0.01real.(r1), imag.(r1))
    r1 = SpectralDistances.normalize_energy(ContinuousRoots(r1))

    a2 = randn(n+1); a2[1] = 1
    r2 = SpectralDistances.reflectc.(roots(reverse(a2)))
    r2 = complex.(0.01real.(r2), imag.(r2))
    r2o = SpectralDistances.normalize_energy(ContinuousRoots(r2))

    m1,m2 = AR(a1), AR(a2)

    dist2 = RationalOptimalTransportDistance(domain=Continuous(), p=2, interval=(-20.,20.))
    f    = w -> evalfr(SpectralDistances.domain(dist2), SpectralDistances.magnitude(dist2), w, m1)
    @test SpectralDistances.c∫(f,dist2.interval...)[end] ≈ spectralenergy(Continuous(), m1) rtol=1e-3
    f    = w -> evalfr(SpectralDistances.domain(dist2), SpectralDistances.magnitude(dist2), w, m2)
    @test SpectralDistances.c∫(f,dist2.interval...)[end] ≈ spectralenergy(Continuous(), m2) rtol=1e-3

    @test spectralenergy(Continuous(), AR(ContinuousRoots([-1.]))) ≈ spectralenergy(tf(1,[1,1])) rtol=1e-3


    m1,m2 = ARMA([1], a1), ARMA([1], a2)

    f    = w -> evalfr(SpectralDistances.domain(dist2), SpectralDistances.magnitude(dist2), w, m1)
    @test_broken SpectralDistances.c∫(f,dist2.interval...)[end] ≈ spectralenergy(Continuous(), m1) rtol=1e-3 # The error is in the integration, spectgralenergy produces same result as for AR above
    f    = w -> evalfr(SpectralDistances.domain(dist2), SpectralDistances.magnitude(dist2), w, m2)
    @test_broken SpectralDistances.c∫(f,dist2.interval...)[end] ≈ spectralenergy(Continuous(), m2) rtol=1e-3


end

@testset "d(m,m)" begin
    a = roots2poly([0.9 + 0.1im, 0.9 - 0.1im])
    m = AR(a)
    for D in [  subtypes(SpectralDistances.AbstractDistance);
                subtypes(SpectralDistances.AbstractRootDistance);
                subtypes(SpectralDistances.AbstractCoefficientDistance)]
        (!isempty(methods(D)) && (:domain ∈ fieldnames(D))) || continue
        d = D(domain=Continuous())
        println(D)
        @test_throws ErrorException d(m,NaN*m)
        @test d(m,m) < eps() + 0.001*(d isa OptimalTransportRootDistance)
        d isa Union{RationalOptimalTransportDistance, RationalCramerDistance} && continue
        d = D(domain=Discrete())
        println(D)
        @test d(m,m) < eps() + 0.001*(d isa OptimalTransportRootDistance)
    end
end

@testset "d(m,m̃)" begin
    a1 = [1,-0.1,0.8]
    m1 = AR(a1)
    a2 = [1,-0.1,0.801]
    m2 = AR(a2)
    for D in [  subtypes(SpectralDistances.AbstractDistance);
                subtypes(SpectralDistances.AbstractRootDistance);
                subtypes(SpectralDistances.AbstractCoefficientDistance)]
        (!isempty(methods(D)) && (:domain ∈ fieldnames(D))) || continue
        d = D(domain=Continuous())
        println(D)
        # @show d(m1,m2)
        @test d(m1,m2) > 1e-10
        d isa Union{RationalOptimalTransportDistance, RationalCramerDistance} && continue
        d = D(domain=Discrete())
        println(D)
        # @show d(m1,m2)
        @test d(m1,m2) > 1e-10
    end
end


@testset "distmat" begin
    e = complex.(randn(3), randn(3))
    D = SpectralDistances.distmat(SqEuclidean(), e)
    @test issymmetric(D)
    @test tr(D) == 0
    @test SpectralDistances.distmat_euclidean(e,e) ≈ D
end


@testset "Welch" begin
    x1 = SpectralDistances.bp_filter(randn(3000), (0.01,0.1))
    x2 = SpectralDistances.bp_filter(randn(3000), (0.01,0.12))
    x3 = SpectralDistances.bp_filter(randn(3000), (0.01,0.3))
    dist = WelchOptimalTransportDistance(p=1)
    @test dist(x1,x2) < dist(x1,x3)
    dist = WelchOptimalTransportDistance(p=2)
    @test dist(x1,x2) < dist(x1,x3)
    @test_throws ErrorException dist(NaN*x1,x3)

    dist = WelchLPDistance(p=1)
    @test dist(x1,x2) < dist(x1,x3)
    dist = WelchLPDistance(p=2)
    @test dist(x1,x2) < dist(x1,x3)

    dist = EnergyDistance()
    @test dist(x1,x2) < dist(x1,x3)
end

@testset "Bures" begin
    fm = LS(na=4)
    x1 = SpectralDistances.bp_filter(randn(3000), (0.01,0.1))  |> fm
    x2 = SpectralDistances.bp_filter(randn(3000), (0.01,0.12)) |> fm
    x3 = SpectralDistances.bp_filter(randn(3000), (0.01,0.3))  |> fm
    dist = BuresDistance()
    @test dist(x1,x2) < dist(x1,x3)
end

@testset "KernelWassersteinRootDistance" begin
    fm = LS(na=4)
    x1 = SpectralDistances.bp_filter(randn(3000), (0.01,0.1))  |> fm
    x2 = SpectralDistances.bp_filter(randn(3000), (0.01,0.12)) |> fm
    x3 = SpectralDistances.bp_filter(randn(3000), (0.01,0.3))  |> fm
    dist = KernelWassersteinRootDistance(domain=Continuous())
    @test dist(x1,x2) < dist(x1,x3)
end

@testset "ClosedForm" begin
    x1 = SpectralDistances.bp_filter(randn(3000), (0.01,0.1))
    x2 = SpectralDistances.bp_filter(randn(3000), (0.01,0.12))
    x3 = SpectralDistances.bp_filter(randn(3000), (0.01,0.3))
    fm = TLS(na=4)
    dist = ModelDistance(fm, RationalOptimalTransportDistance(domain=Continuous(), p=1, interval=(-15., 15)))
    @test dist(x1,x2) < dist(x1,x3)
    @test dist(x1,x2) < dist(x1,x3)
    dist = RationalOptimalTransportDistance(domain=Continuous(), p=1, interval=(0., 15.))
    @test dist(fm(x1),welch_pgram(x2)) < dist(fm(x1),welch_pgram(x3))
    @test dist(fm(x1),welch_pgram(x2)) < dist(fm(x1),welch_pgram(x3))

    dist = ModelDistance(fm, RationalOptimalTransportDistance(domain=Continuous(), p=2, interval=(-15., 15)))
    @test dist(x1,x2) < dist(x1,x3)
    @test dist(x1,x2) < dist(x1,x3)
    dist = RationalOptimalTransportDistance(domain=Continuous(), p=2, interval=(0., 15.))
    @test dist(fm(x1),welch_pgram(x2)) < dist(fm(x1),welch_pgram(x3))
    @test dist(fm(x1),welch_pgram(x2)) < dist(fm(x1),welch_pgram(x3))

    w = LinRange(0, 2pi*15, 3000)
    dist = ModelDistance(fm, RationalOptimalTransportDistance(domain=Continuous(), p=1, interval=(0., 15)))
    ddist = ModelDistance(fm, DiscretizedRationalDistance(w, sqrt.(SpectralDistances.distmat_euclidean(w,w))))
    @test ddist(x1, x1) == 0
    @test_broken ddist(x1, x2) ≈ dist(x1, x2)

    w = LinRange(0, 2pi*15, 3000)
    dist = ModelDistance(fm, RationalOptimalTransportDistance(domain=Continuous(), p=2, interval=(0., 15)))
    ddist = ModelDistance(fm, DiscretizedRationalDistance(w, (SpectralDistances.distmat_euclidean(w,w))))
    @test ddist(x1, x1) == 0
    @test_broken ddist(x1, x2) ≈ dist(x1, x2)

end

@testset "Histogram" begin
    x1 = randn(100)
    x2 = randn(100) .+ 1
    x3 = randn(100) .+ 2
    dist = SpectralDistances.OptimalTransportHistogramDistance(p=1)
    @test dist(x1,x2) < dist(x1,x3)
    dist = SpectralDistances.OptimalTransportHistogramDistance(p=2)
    @test dist(x1,x2) < dist(x1,x3)
end




@testset "Theoretical scaling from paper" begin
    n = 4
    a1 = randn(n+1); a1[1] = 1
    r1 = SpectralDistances.reflectc.(roots(reverse(a1)))
    r1 = ContinuousRoots(complex.(0.01real.(r1), imag.(r1)))
    # r1 = SpectralDistances.normalize_energy(ContinuousRoots(r1))

    a2 = randn(n+1); a2[1] = 1
    r2 = SpectralDistances.reflectc.(roots(reverse(a2)))
    r2 = ContinuousRoots(complex.(0.01real.(r2), imag.(r2)))
    # r2 = SpectralDistances.normalize_energy(ContinuousRoots(r2))


    function scaleddist(alpha,p)
        r1a,r2a = ContinuousRoots(alpha.*r1), ContinuousRoots(alpha.*r2)
        m1,m2 = AR(r1a), AR(r2a)
        # dist = RationalOptimalTransportDistance(domain=Continuous(), p=p, interval=(-20.,20.))
        rdist = EuclideanRootDistance(domain=Continuous(), weight=residueweight, p=p)
        rdist(m1,m2)
    end
    for α ∈ 1:0.5:3, p ∈ 1:3
        @test α^(1. -2n +p) * scaleddist(1,p) ≈ scaleddist(α,p)

        @test α^(1. -n)*residues(ContinuousRoots(r1)) ≈ residues(ContinuousRoots(α*r1))
        @test α^(1. -2n)*residueweight(r1) ≈ residueweight(α*r1)
    end

end


@testset "Interpolations" begin
    # NOTE: Only testing that it runs, not that it does the right thing. Have a look at the figure in the docs to deteermine if it works well
    n = 4
    r1 = complex.(-0.01 .+ 0.001randn(3), 2randn(3))
    r1 = ContinuousRoots([r1; conj.(r1)])
    r2 = complex.(-0.01 .+ 0.001randn(3), 2randn(3))
    r2 = ContinuousRoots([r2; conj.(r2)])
    r1,r2 = normalize_energy.((r1, r2))
    A1 = AR(r1)
    A2 = AR(r2)
    t = 0.1
    dist = RationalOptimalTransportDistance(domain=Continuous(), p=2, interval=(0., exp10(1.01)))
    interp = SpectralDistances.interpolator(dist, A1, A2)
    w = exp10.(LinRange(-1.5, 1, 300))
    for t = LinRange(0, 1, 7)
        Φ = clamp.(interp(w,t), 1e-10, 100)
    end

    rdist = EuclideanRootDistance(domain=Continuous(), p=2)
    interp = SpectralDistances.interpolator(rdist, A1, A2, normalize=false)
    w = exp10.(LinRange(-1.5, 1, 300))
    for t = LinRange(0, 1, 7)
        Φ = interp(w,t)
    end

@testset "barycenter" begin
    @info "Testing barycenter"
    include("test_barycenter.jl")
end

@testset "Slerp" begin
    a = n1(randn(3))
    b = n1(randn(3))
    c = SpectralDistances.slerp(a,b,0.5)
    @test c ≈ n1((a+b)/2)
end


@testset "Utils" begin
    X = randn(1000,10)
    x1,x2 = twoD(X)
    @test length(x1) == length(x2) == size(X,1)
    x1,x2,x3 = threeD(X)
    @test length(x1) == length(x3) == size(X,1)

    @test norm(n1(x1)) ≈ 1
    @test sum(s1(abs.(x1))) ≈ 1
    @test var(v1(x1)) ≈ 1
    @test mean(v1(x1)) ≈ 0 atol=sqrt(eps())
    @test mean(SpectralDistances.m1(x1)) ≈ 0 atol=0.08
    @test var(SpectralDistances.m1(x1)) ≈ 1 atol=0.2


    # @test norm(n1(X,1)) ≈ 1
    @test  all(sum(s1(abs.(X),1), dims=1) .≈ 1)
    @test  all(var(v1(X,1), dims=1) .≈ 1)
    @test  mean(v1(X,1)) ≈ 0 atol = sqrt(eps())
    @test  mean(SpectralDistances.m1(X,1)) ≈ 0 atol=0.08
    @test  var(SpectralDistances.m1(X,1)) ≈ 1 atol=0.2


end

end

end

# x1 = randn(1000)
# x2 = randn(1000)
# fm = TLS(na=4)
# inner = EuclideanRootDistance(domain=Continuous(), p=2)
# dist = ModelDistance(fm, inner)
# dist(x1,x2)
# using Zygote
# m1,m2 = fm(x1), fm(x2)
# Zygote.gradient(inner, m1,m2)
#
# function di(e1,e2)
#     w1 = ones(1)
#     l = 0.
#     for i in 1:length(e1)
#         l += abs(w1[1]*e1[i]-e2[i])
#     end
#     l
# end
#
# Zygote.gradient(di, [complex(1.2)], [complex(1.2)])
#
# function g(v)
#     z = complex(v[1],v[2])
#     sum(eachindex(z)) do i
#         abs2(1*z)
#     end
# end
