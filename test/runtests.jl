@info "Running tests"
using SpectralDistances# Distributions
using Test, LinearAlgebra, Statistics, Random, ControlSystems, InteractiveUtils # For subtypes
# using ForwardDiff
using DSP


Random.seed!(0)

@testset "SpectralDistances.jl" begin

#     @testset "gradients" begin
#         y = param(randn(10))
#         u = param(randn(10))
#         using ForwardDiff
#         @test Flux.jacobian((y)->vec(getARXregressor(y,u,2,2)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,2,2)[2]), y.data)
#         @test Flux.jacobian((y)->vec(getARXregressor(y,u,2,2)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,2,2)[1]), y.data)
#         @test Flux.jacobian((y)->vec(getARXregressor(y,u,5,2)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,5,2)[2]), y.data)
#         @test Flux.jacobian((y)->vec(getARXregressor(y,u,5,2)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,5,2)[1]), y.data)
#
#         @test Flux.jacobian((y)->vec(getARXregressor(y,u,2,1)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,2,1)[2]), y.data)
#         @test Flux.jacobian((y)->vec(getARXregressor(y,u,2,1)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,2,1)[1]), y.data)
#         @test Flux.jacobian((y)->vec(getARXregressor(y,u,5,1)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,5,1)[2]), y.data)
#         @test Flux.jacobian((y)->vec(getARXregressor(y,u,5,1)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,5,1)[1]), y.data)
#
#
#
#         y = param(randn(10))
#         @test Flux.jacobian((y)->vec(getARregressor(y,2)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARregressor(y,2)[2]), y.data)
#         @test Flux.jacobian((y)->vec(getARregressor(y,2)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARregressor(y,2)[1]), y.data)
#         @test Flux.jacobian((y)->vec(getARregressor(y,5)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARregressor(y,5)[2]), y.data)
#         @test Flux.jacobian((y)->vec(getARregressor(y,5)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARregressor(y,5)[1]), y.data)
#
#
#
#         let Gc = tf(1,[1,1,1,1])
#             w = c2d(Gc,1).matrix[1] |> ControlSystems.denvec
#             @test d2c(w) ≈ pole(Gc)
#         end
#
#         y = randn(5000)
#         plr(y,40,2)
#         @test Flux.gradient(y->sum(plr(y,4,2)[1]), y)[1][1].data ≈ ForwardDiff.gradient(y->sum(plr(y,4,2)[1]), y)
#         @test Flux.gradient(y->sum(plr(y,4,2)[2]), y)[1][1].data ≈ ForwardDiff.gradient(y->sum(plr(y,4,2)[2]), y)
#
#
#         p = [1.,1,1]
#         # @btime riroots(p)
#         fd = central_fdm(3, 1)
#         @test Flux.gradient(p) do p
#             r = riroots(p)
#             sum([r[1]; r[2]])
#         end[1][1].data ≈ FiniteDifferences.grad(fd, p->begin
#         r = roots(p)
#         sum([real(r); imag(r)])
#     end, p)
#
# end


@testset "modeldistance" begin
    t = 1:100
    ϵ = 1e-7
    ls_loss = ModelDistance(LS(na=2), EuclideanRootDistance(domain=Discrete()))
    @test ls_loss(sin.(t), sin.(t)) < ϵ
    @test ls_loss(sin.(t), -sin.(t)) < ϵ # phase flip invariance
    @test ls_loss(sin.(t), sin.(t .+ 1)) < ϵ # phase shift invariance
    @test ls_loss(sin.(t), sin.(t .+ 0.1)) < ϵ # phase shift invariance
    @test ls_loss(10sin.(t), sin.(t .+ 1)) < ϵ # amplitude invariance
    @test ls_loss(sin.(t), sin.(1.1 .* t)) < 0.2 # small frequency shifts gives small errors
    @test ls_loss(sin.(0.1t), sin.(1.1 .* 0.1t)) < 0.1 # small frequency shifts gives small errors

    @test_broken ls_loss(sin.(t), sin.(1.1 .* t)) ≈ ls_loss(sin.(0.1t), sin.(1.1 .* 0.1t)) < 0.1 # frequency shifts of relative size should result in the same error
    ls_loss = ModelDistance(LS(na=10), CoefficientDistance(domain=Discrete()))
    @test ls_loss(randn(100), randn(100)) > 0.05
    @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < 0.3 # Filtered through same filter
    @test_broken ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < ls_loss(filtfilt(ones(9),[9], randn(1000)), filtfilt(ones(10),[10], randn(1000))) # Filtered through different filters
end


@testset "trivial_transport" begin
    x = [1.,0,0]
    y = [0,0.5,0.5]

    g = SpectralDistances.trivial_transport(x,y)
    @test sum(g,dims=1)[:] == y
    @test sum(g,dims=2)[:] == x
end


# p = randn(20)
# @btime Flux.gradient(p) do p
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
    ec = log(ed)
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
    r1 = complex.(0.01real.(r1), imag.(r1))
    # r1 = SpectralDistances.normalize_energy(ContinuousRoots(r1))

    a2 = randn(n+1); a2[1] = 1
    r2 = SpectralDistances.reflectc.(roots(reverse(a2)))
    r2 = complex.(0.01real.(r2), imag.(r2))
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

        @test α^(1. -n)*residues(ContinuousRoots(r1)) ≈ residues(ContinuousRoots(α.*r1))
        @test α^(1. -2n)*residueweight_unnormalized(r1) ≈ residueweight_unnormalized(α.*r1)
    end

end

end
