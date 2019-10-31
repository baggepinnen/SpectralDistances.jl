@info "Running tests"
using SpectralDistances# Distributions
using Test, LinearAlgebra, Statistics, Random, ControlSystems, InteractiveUtils # For subtypes
using DSP, Distances


Random.seed!(1)


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
#
# function hessian(f, a)
#     H = jacobian(f', a)
# end
#
# function hessian_fd_reverse(f,a)
#     fd = central_fdm(9,1)
#     FiniteDifferences.jacobian(fd, f', a)
# end
#
# a = randn(11)
# a[1] = 1
#
# m = AR(ContinuousRoots([-0.1+im, -0.1-im]))
# a = Vector(m.a)
# dist = EuclideanRootDistance(domain=Continuous(), p=2, weight=residueweight)
# function forward(b)
#     dist(m, AR((b)))
# end
# forward(a)
# @time forward'(a)
# @time H = hessian_fd_reverse(forward, a)
# fd = central_fdm(5,1)
#
# eigvals(H[2:end,2:end])
#
#
# grad(fd, forward, big.(a))
#
# @btime H = hessian_fd_reverse($(forward), $a)
#
#
# hessian(forward, a)




f = a -> sum(polyderivative(a))

f'(a)
function polyderivative(a)
    a = Vector(a)
    n = length(a)-1
    powers = n:-1:1
    ap     = powers.*a[1:end-1]
end
f'(SVector{length(a)}(a))



a = randn(9)
a[1] = 1
r = roots(reverse(a)) |> ContinuousRoots

forward,backward = wrdcurv(a)
forward(a)
H = backward(a)
cond(H)



@testset "SpectralDistances.jl" begin

    get(ENV, "TRAVIS_BRANCH", nothing) == nothing && @testset "gradients" begin

        using ForwardDiff, FiniteDifferences
        using Zygote
        using SpectralDistances: getARXregressor, getARregressor
        y = (randn(10))
        u = (randn(10))
        jacobian((y)->vec(getARXregressor(y,u,2,2)[2]), y)
        @test jacobian((y)->vec(getARXregressor(y,u,2,2)[2]), y) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u,2,2)[2]), y)
        @test jacobian((y)->vec(getARXregressor(y,u,2,2)[1]), y) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u,2,2)[1]), y)
        @test jacobian((y)->vec(getARXregressor(y,u,5,2)[2]), y) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u,5,2)[2]), y)
        @test jacobian((y)->vec(getARXregressor(y,u,5,2)[1]), y) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u,5,2)[1]), y)

        @test jacobian((y)->vec(getARXregressor(y,u,2,1)[2]), y) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u,2,1)[2]), y)
        @test jacobian((y)->vec(getARXregressor(y,u,2,1)[1]), y) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u,2,1)[1]), y)
        @test jacobian((y)->vec(getARXregressor(y,u,5,1)[2]), y) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u,5,1)[2]), y)
        @test jacobian((y)->vec(getARXregressor(y,u,5,1)[1]), y) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u,5,1)[1]), y)



        y = (randn(10))
        @test jacobian((y)->vec(getARregressor(y,2)[2]), y) == ForwardDiff.jacobian((y)->vec(getARregressor(y,2)[2]), y)
        @test jacobian((y)->vec(getARregressor(y,2)[1]), y) == ForwardDiff.jacobian((y)->vec(getARregressor(y,2)[1]), y)
        @test jacobian((y)->vec(getARregressor(y,5)[2]), y) == ForwardDiff.jacobian((y)->vec(getARregressor(y,5)[2]), y)
        @test jacobian((y)->vec(getARregressor(y,5)[1]), y) == ForwardDiff.jacobian((y)->vec(getARregressor(y,5)[1]), y)

        a,b = randn(3), randn(4)
        @test jacobian(a->SpectralDistances.polyconv(a,b), a) ≈ ForwardDiff.jacobian(a->SpectralDistances.polyconv(a,b),a)

        @test jacobian(b->SpectralDistances.polyconv(a,b), b) ≈ ForwardDiff.jacobian(b->SpectralDistances.polyconv(a,b),b)



        @test_skip let Gc = tf(1,[1,1,1,1])
            w = c2d(Gc,1).matrix[1] |> ControlSystems.denvec
            @test d2c(w) ≈ pole(Gc)
        end

        y = randn(5000)
        fm = PLR(na=40, nc=2)
        @test_skip Zygote.gradient(y->sum(fm(y)[1]), y)[1] ≈ ForwardDiff.gradient(y->sum(fm(y)[1]), y)
        @test_skip Zygote.gradient(y->sum(fm(y)[2]), y)[1] ≈ ForwardDiff.gradient(y->sum(fm(y)[2]), y)


        p = [1.,1,1]
        # @btime riroots(p)
        fd = central_fdm(3, 1)
        @test Zygote.gradient(p) do p
            r = roots(p)
            sum(abs2, r)
        end[1][1:end-1] ≈ FiniteDifferences.grad(fd, p->begin
            r = roots(p)
            sum(abs2, r)
        end, p)[1:end-1]

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
        @test G[1:end-1] ≈ FiniteDifferences.grad(fdm, f, a)[1:end-1]


        fd = central_fdm(5,1)
        a = randn(30)
        a[1] = 1
        r = roots(reverse(a)) |> ContinuousRoots
        residues(a,r)
        f = a -> sum(abs2, residues(a,r))
        g = a -> real(residues(complex.(a),r)[2])
        @test sum(abs, f'(complex.(a))[2:end] - grad(fd,f,a)[2:end]) < sqrt(eps())
        @test sum(abs, g'((a))[2:end] - grad(fd,g,a)[2:end]) < sqrt(eps())

        fd = central_fdm(7,1)
        a = randn(30)
        a[1] = 1
        residues(a)
        f = a -> sum(abs2, residues(a))
        g = a -> real(residues(complex.(a))[2])
        @test sum(abs, f'(complex.(a))[2:end] - grad(fd,f,a)[2:end]) < sqrt(eps())
        @test_broken sum(abs, g'((a))[2:end] - grad(fd,g,a)[2:end]) < sqrt(eps())


        @testset "Sinkhorn" begin

            function sinkdist(D,a,b)
                ai = s1(a)
                bi = s1(a+b)
                P,u,v = sinkhorn(D,ai,bi, iters=1000, β=0.1)
                sum(P.*D)
            end
            a,b = abs.(randn(6)),abs.(randn(6))
            D = SpectralDistances.distmat_euclidean(1:length(a), 1:length(a))
            dD, da, db = Zygote.gradient(sinkdist, D,a,b)
            @test n1(ForwardDiff.gradient(a->sinkdist(D,a,b), a))'n1(da) > 0.9
        end


    end

    a = randn(5); a[1]=1
    @test SpectralDistances.roots2poly(a) ≈ SpectralDistances.roots2poly_zygote(a)


    @testset "Energy" begin
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

        fm = TLS(na=2)
        m = fm(y)
        @test imag.(m.pc) ≈ [-0.1, 0.1] rtol=1e-4

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
        @test_broken spectralenergy(Continuous(), m) ≈ 2π*var(y)
        @test imag.(m.pc) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=1e-4

        y = sin.(0:0.1:1000) .+ sin.(2 .* (0:0.1:1000)) .+ 0.01 .*randn.()
        fm = PLR(na=4, nc=1, λ=0.0)
        m = fm(y)
        @test_broken imag.(log(m.p)) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=1e-3

    end




@testset "modeldistance" begin
    t = 1:100
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

        @test ls_loss(sin.(t), sin.(1.1 .* t)) ≈ ls_loss(sin.(0.1t), sin.(0.2t)) rtol=1e-3  # frequency shifts of relative size should result in the same error, probably only true for p=1
        ls_loss = ModelDistance(fitmethod(na=10), SinkhornRootDistance(domain=Discrete()))
        @test ls_loss(randn(100), randn(100)) > 0.05
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
        ls_loss = ModelDistance(fitmethod(na=10,nc=2), SinkhornRootDistance(domain=Discrete()))
        @test ls_loss(randn(1000), randn(1000)) > 0.05
        @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < 0.1 # Filtered through same filter, this test is very non-robust for TLS
        @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < ls_loss(filtfilt(ones(4),[4], randn(1000)), filtfilt(ones(10),[10], randn(1000))) # Filtered through different filters, this test is not robust
    end
end


@testset "discrete_grid_transportplan" begin
    x = [1.,0,0]
    y = [0,0.5,0.5]

    g = SpectralDistances.discrete_grid_transportplan(x,y)
    @test sum(g,dims=1)[:] == y
    @test sum(g,dims=2)[:] == x
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

end

@testset "ControlSystems interoperability" begin
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
end

@testset "polynomial acrobatics" begin
    a = randn(5)
    b = randn(5)
    @test polyconv(a,b) ≈ DSP.conv(a,b)

    a = randn(5)
    b = randn(10)
    @test polyconv(a,b) ≈ DSP.conv(a,b)

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

    res = residues(a)
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


end

@testset "d(m,m)" begin
    a = roots2poly([0.9 + 0.1im, 0.9 - 0.1im])
    m = AR(a)
    for D in [  subtypes(SpectralDistances.AbstractDistance);
                subtypes(SpectralDistances.AbstractRootDistance);
                subtypes(SpectralDistances.AbstractCoefficientDistance)]
        (!isempty(methods(D)) && (:domain ∈ fieldnames(D))) || continue
        @show d = D(domain=Continuous())
        @test d(m,m) < eps() + 0.001*(d isa SinkhornRootDistance)
        d isa Union{RationalOptimalTransportDistance, RationalCramerDistance} && continue
        @show d = D(domain=Discrete())
        @test d(m,m) < eps() + 0.001*(d isa SinkhornRootDistance)
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
        @show d = D(domain=Continuous())
        # @show d(m1,m2)
        @test d(m1,m2) > 1e-10
        d isa Union{RationalOptimalTransportDistance, RationalCramerDistance} && continue
        @show d = D(domain=Discrete())
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
end

@testset "ClosedForm" begin
    x1 = SpectralDistances.bp_filter(randn(3000), (0.01,0.1))
    x2 = SpectralDistances.bp_filter(randn(3000), (0.01,0.12))
    x3 = SpectralDistances.bp_filter(randn(3000), (0.01,0.3))
    fm = TLS(na=4)
    dist = ModelDistance(fm, RationalOptimalTransportDistance(domain=Continuous(), p=1))
    @test dist(x1,x2) < dist(x1,x3)
    @test dist(x1,x2) < dist(x1,x3)
    dist = RationalOptimalTransportDistance(domain=Continuous(), p=1, interval=(0., 15.))
    @test dist(fm(x1),welch_pgram(x2)) < dist(fm(x1),welch_pgram(x3))
    @test dist(fm(x1),welch_pgram(x2)) < dist(fm(x1),welch_pgram(x3))

    w = LinRange(0, 2pi*15, 3000)
    dist = ModelDistance(fm, RationalOptimalTransportDistance(domain=Continuous(), p=1, interval=(0., 15)))
    ddist = ModelDistance(fm, DiscretizedRationalDistance(w, sqrt.(SpectralDistances.distmat_euclidean(w,w))))
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
