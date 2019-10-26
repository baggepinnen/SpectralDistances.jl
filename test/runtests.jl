@info "Running tests"
using SpectralDistances# Distributions
using Test, LinearAlgebra, Statistics, Random, ControlSystems, InteractiveUtils # For subtypes
using DSP


Random.seed!(0)


function jacobian(m,x)
    y  = m(x)
    k  = length(y)
    n  = length(x)
    J  = Matrix{eltype(x)}(undef,k,n)
    for i = 1:k
        g = Zygote.gradient(x->m(x)[i], x)[1]
        J[i,:] .= g
    end
    J
end

@testset "SpectralDistances.jl" begin

    get(ENV, "TRAVIS_BRANCH", nothing) == nothing && @testset "gradients" begin

        using ForwardDiff, FiniteDifferences
        using ForwardDiff
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
        f(x) = sum(abs2(r) for r in roots(x))
        G = Zygote.gradient(f, a)[1]
        f'(a)


        fdm = central_fdm(5,1)
        @test G[1:end-1] ≈ FiniteDifferences.grad(fdm, f, a)[1:end-1]


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
end


@testset "trivial_transport" begin
    x = [1.,0,0]
    y = [0,0.5,0.5]

    g = SpectralDistances.trivial_transport(x,y)
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

    dist2 = ClosedFormSpectralDistance(domain=Continuous(), p=2, interval=(-200.,200.))
    f    = w -> evalfr(SpectralDistances.domain(dist2), SpectralDistances.magnitude(dist2), w, m1)
    @test SpectralDistances.c∫(f,dist2.interval...)[end] ≈ SpectralDistances.spectralenergy(Continuous(), m1) rtol=1e-3
    f    = w -> evalfr(SpectralDistances.domain(dist2), SpectralDistances.magnitude(dist2), w, m2)
    @test SpectralDistances.c∫(f,dist2.interval...)[end] ≈ SpectralDistances.spectralenergy(Continuous(), m2) rtol=1e-3


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
        d isa Union{ClosedFormSpectralDistance, CramerSpectralDistance} && continue
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
        d isa Union{ClosedFormSpectralDistance, CramerSpectralDistance} && continue
        @show d = D(domain=Discrete())
        # @show d(m1,m2)
        @test d(m1,m2) > 1e-10
    end
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
        # dist = ClosedFormSpectralDistance(domain=Continuous(), p=p, interval=(-20.,20.))
        rdist = EuclideanRootDistance(domain=Continuous(), weight=residueweight, p=p)
        rdist(m1,m2)
    end
    for α ∈ 1:0.5:3, p ∈ 1:3
        @test α^(1. -2n +p) * scaleddist(1,p) ≈ scaleddist(α,p)

        @test α^(1. -n)*residues(ContinuousRoots(r1)) ≈ residues(ContinuousRoots(α*r1))
        @test α^(1. -2n)*residueweight(r1) ≈ residueweight(α*r1)
    end

end

end
