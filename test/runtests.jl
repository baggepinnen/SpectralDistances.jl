@info "Running tests"
using SpectralDistances# Distributions
using Test, LinearAlgebra, Statistics, Random, ControlSystems, InteractiveUtils, SparseArrays # For subtypes
using DSP, Distances, DoubleFloats
import GLPK, Convex, JuMP

using SpectralDistances: ngradient, nhessian, njacobian, polyconv, hproots, rev


@testset "SpectralDistances.jl" begin


    @testset "Convolutional" begin
        @info "Testing Convolutional"
        include("test_convolutional.jl")
    end


    @testset "DTW" begin
        @info "Testing DTW"
        include("test_dtw.jl")
    end

    @testset "Model creation" begin
        @info "Testing Model creation"

        @testset "AR" begin
            @info "Testing AR"

            a = randn(3)
            m = AR(a)
            @test_nowarn show(m)
            @test_nowarn show([m, m])
            @test m.a == a
            @test length(m.p) == 2

            m = AR(Continuous(), a)
            @test m.ac == a
            @test length(m.pc) == 2

            r = ContinuousRoots(hproots(rev(a)))
            m = AR(r)
            @test m.pc == r

            r = DiscreteRoots(hproots(rev(a)))
            m = AR(r)
            @test m.p == r
        end


        @testset "ARMA" begin
            @info "Testing ARMA"

            a = randn(3)
            b = [1, 0.1]
            m = ARMA(b,a)
            @test s1(m.a) == s1(a)
            @test length(m.p) == 2
            @test m.b == b
            @test length(m.z) == 1

            m = ARMA(Continuous(), b,a)
            @test s1(m.ac) == s1(a)
            @test length(m.pc) == 2
            @test m.bc == b
            @test length(m.zc) == 1

            r = ContinuousRoots(hproots(rev(a)))
            z = ContinuousRoots(hproots(rev(b)))
            m = ARMA(z,r)
            @test m.pc == r
            @test m.zc == z

            r = DiscreteRoots(hproots(rev(a)))
            z = DiscreteRoots(hproots(rev(b)))
            m = ARMA(z,r)
            @test m.p == r
            @test m.z == z
        end

    end


    @testset "time" begin
        @info "Testing time"
        include("test_time.jl")
    end

    @testset "plot" begin
        @info "Testing plot"
        include("test_plotting.jl")
    end

    @testset "Solvers" begin
        @info "Testing Solvers"

        C = Float64.(Matrix(I(2)))
        a = [1., 0]
        b = [0, 1.]
        Γ,u,v = @inferred sinkhorn(C,a,b,β=0.01)
        @test Γ ≈ [0 1; 0 0]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = @inferred sinkhorn(C,a,b,β=0.1)
        @test Γ ≈ [0 0.5; 0 0.5]

        a = [1., 0]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = @inferred sinkhorn_log(C,a,b,β=0.01)
        @test Γ ≈ [0 1; 0 0]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = @inferred sinkhorn_log(C,a,b,β=0.01)
        @test Γ ≈ [0 0.5; 0 0.5]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = @inferred sinkhorn_log!(C,a,b,β=0.01)
        @test Γ ≈ [0 0.5; 0 0.5]

        a = [1., 0]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = @inferred IPOT(C,a,b,β=1)
        @test Γ ≈ [0 1; 0 0]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = @inferred IPOT(C,a,b)
        @test Γ ≈ [0 0.5; 0 0.5]

        a = [1., 0]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = @inferred sinkhorn_unbalanced(C,a,b,Balanced(),β=0.01)
        @test Γ ≈ [0 1; 0 0]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = @inferred sinkhorn_unbalanced(C,a,b,Balanced())
        @test Γ ≈ [0 0.5; 0 0.5]

        a = [1., 0]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = ot_jump(C,a,b)
        @test Γ ≈ [0 1; 0 0]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = ot_jump(C,a,b)
        @test Γ ≈ [0 0.5; 0 0.5]

        a = [1., 0]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = ot_convex(C,a,b)
        @test Γ ≈ [0 1; 0 0]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = ot_convex(C,a,b)
        @test Γ ≈ [0 0.5; 0 0.5]

        a = [0.5, 0.5]
        b = [0, 1.]
        C = Float64.(Matrix(I(2)))
        Γ,u,v = ot_convex(C,a,b)
        @test Γ ≈ [0 0.5; 0 0.5]

        models = examplemodels(2)
        Γ = transport_plan(OptimalTransportRootDistance(domain=Continuous(), weight=unitweight), models...; tol=1e-4)
        @test sum(Γ) ≈ 1 atol=0.001

    end


    @testset "gradients" begin
        @info "Testing gradients"
        include("test_diff.jl")
    end


    @testset "Energy" begin
        @info "testing Energy"
        for σ² = [0.1, 1., 2., 3]
            m = AR(ContinuousRoots([-1.]), σ²)
            e = @inferred spectralenergy(Continuous(),m)
            @test e ≈ σ²
            x = sqrt(σ²)randn(10000)
            m = LS(na=10)(x)
            @test spectralenergy(Continuous(),m) ≈ σ² atol=0.15
            m = TLS(na=10)(x)
            @test spectralenergy(Continuous(),m) ≈ σ² atol=0.15
        end
        # y = filt(numvec(Discrete(),m), denvec(Discrete(),m), x)
        # @test var(y) ≈ var(x)
    end

    @testset "Model estimation" begin
        y = sin.(0:0.1:100)
        fm = @inferred LS(na=2, λ=0)
        m = @inferred fm(y)
        @test fm(m) == m
        @test imag.(m.pc) ≈ [-0.1, 0.1] rtol=1e-4
        mc = @inferred SpectralDistances.change_precision(Float32, m)
        @test m ≈ mc
        @test eltype(mc.a) == Float32

        fm = @inferred TLS(na=2)
        m  = @inferred fm(y)
        @test imag.(m.pc) ≈ [-0.1, 0.1] rtol=1e-4

        fm = IRLS(na=2)
        m = fm(y)
        @test_broken imag.(m.pc) ≈ [-0.1, 0.1] rtol=1e-4

        y = sin.(0:0.1:1000) .+ 0.01 .*randn.()
        fm = PLR(na=2, nc=1)
        m = fm(y)
        @test imag.(log(m.p)) ≈ [-0.1, 0.1] rtol=1e-3


        y = sin.(0:0.1:100) .+ sin.(2 .* (0:0.1:100) .+ 0.3)
        fm = @inferred LS(na=4, λ=0)
        m = @inferred fm(y)
        @test imag.(m.pc) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=1e-1

        fm = TLS(na=4)
        m = fm(y)
        @test spectralenergy(Continuous(), m) ≈ var(y) rtol=1e-3
        @test imag.(m.pc) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=1e-4

        fm = IRLS(na=4)
        m = fm(y)
        @test @inferred(spectralenergy(Continuous(), m)) ≈ var(y) rtol=1e-3
        @test_broken imag.(m.pc) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=1e-4

        y = sin.(0:0.1:1000) .+ sin.(2 .* (0:0.1:1000)) .+ 0.001 .*randn.()
        fm = @inferred PLR(na=4, nc=1, λ=0.0)
        m = fm(y)
        @test imag.(m.pc) ≈ [-0.2, -0.1, 0.1, 0.2] rtol=0.6

    end




@testset "modeldistance" begin
    t = 1:300
    ϵ = 1e-7
    fitmethod = LS
    for fitmethod in [LS, TLS]
        @info "Testing fitmethod $(string(fitmethod))"
        ls_loss = ModelDistance(fitmethod(na=2), EuclideanRootDistance(domain=Discrete()))
        @test @inferred(ls_loss(sin.(t), sin.(t))) < ϵ
        @test ls_loss(sin.(t), -sin.(t)) < ϵ # phase flip invariance
        @test ls_loss(sin.(t), sin.(t .+ 1)) < ϵ # phase shift invariance
        @test ls_loss(sin.(t), sin.(t .+ 0.1)) < ϵ # phase shift invariance
        @test ls_loss(10sin.(t), sin.(t .+ 1)) < ϵ # amplitude invariance
        @test ls_loss(sin.(t), sin.(1.1 .* t)) < 0.2 # small frequency shifts gives small errors
        @test ls_loss(sin.(0.1t), sin.(1.1 .* 0.1t)) < 0.1 # small frequency shifts gives small errors

        @test ls_loss(sin.(t), sin.(1.1 .* t)) ≈ ls_loss(sin.(0.1t), sin.(0.2t)) rtol=1e-2  # frequency shifts of relative size should result in the same error, probably only true for p=1
        ls_loss = ModelDistance(fitmethod(na=10), OptimalTransportRootDistance(domain=Discrete()))
        @test @inferred(ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000)))) < 0.1 # Filtered through same filter, this test is very non-robust for TLS
        @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < ls_loss(filtfilt(ones(4),[4], randn(1000)), filtfilt(ones(10),[10], randn(1000))) # Filtered through different filters, this test is not robust
    end
    @testset "PLR" begin
        fitmethod = PLR
        @info "Testing fitmethod $(string(fitmethod))"
        t = 1:1000 .+ 0.01 .* randn.()
        ϵ = 0.01
        @info "Creating model distance"
        ls_loss = ModelDistance(fitmethod(na=2,nc=1), EuclideanRootDistance(domain=Discrete()))
        @info "Calc loss"
        @test ls_loss(sin.(t), sin.(t)) < ϵ
        @test ls_loss(sin.(t), -sin.(t)) < ϵ # phase flip invariance
        @test ls_loss(sin.(t), sin.(t .+ 1)) < ϵ # phase shift invariance
        @test ls_loss(sin.(t), sin.(t .+ 0.1)) < ϵ # phase shift invariance
        @info "Halfway"
        @test ls_loss(10sin.(t), sin.(t .+ 1)) < ϵ # amplitude invariance
        @test ls_loss(sin.(t), sin.(1.1 .* t)) < 0.2 # small frequency shifts gives small errors
        @test ls_loss(sin.(0.1t), sin.(1.1 .* 0.1t)) < 0.1 # small frequency shifts gives small errors

        @test ls_loss(sin.(t), sin.(1.1 .* t)) ≈ ls_loss(sin.(0.1t), sin.(0.2t)) rtol=1e-3  # frequency shifts of relative size should result in the same error, probably only true for p=1
        @info "Almost there"
        # ls_loss = ModelDistance(fitmethod(na=10,nc=2), OptimalTransportRootDistance(domain=Discrete())) # This test segfaulted sometimes, couldn't figure out the problem
        # @test ls_loss(randn(1000), randn(1000)) > 0.05
        # @info "Done"
        # @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < 1 # Filtered through same filter, this test is very non-robust for TLS
        # @test ls_loss(filtfilt(ones(10),[10], randn(1000)), filtfilt(ones(10),[10], randn(1000))) < ls_loss(filtfilt(ones(4),[4], randn(1000)), filtfilt(ones(10),[10], randn(1000))) # Filtered through different filters, this test is not robust
    end
end


@testset "discrete_grid_transportplan" begin
    x = [1.,0,0]
    y = [0,0.5,0.5]

    g = @inferred discrete_grid_transportplan(x,y)
    @test sum(g,dims=1)[:] == y
    @test sum(g,dims=2)[:] == x

    # test robustness for long vectors
    x = s1(rand(1000))
    y = s1(rand(1000))
    D = [abs2(abs(i-j)/1000) for i in eachindex(x), j in eachindex(y)]
    g = discrete_grid_transportplan(x,y)
    @test sum(g,dims=1)[:] ≈ y
    @test sum(g,dims=2)[:] ≈ x
    @test dot(g, D) ≈ discrete_grid_transportcost(x,y) rtol=0.01

    g = discrete_grid_transportplan(y,x)
    @test sum(g,dims=1)[:] ≈ x
    @test sum(g,dims=2)[:] ≈ y
    @test dot(g, D) ≈ discrete_grid_transportcost(y,x) rtol=0.01

    # test exception for unequal masses
    x = s1(rand(Float32,1000))
    y = rand(Float32,1000)
    @test_throws ErrorException discrete_grid_transportplan(x,y)
    @test_throws ErrorException discrete_grid_transportplan(y,x)

    @test_throws ErrorException discrete_grid_transportcost(x,y)
    @test_throws ErrorException discrete_grid_transportcost(y,x)

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
    @test @inferred(SpectralDistances.reflectd(2)) ≈ 0.5 + 0.0im
    @test @inferred(SpectralDistances.reflectd(complex(0,2))) ≈ 0 + 0.5im
    @test @inferred(SpectralDistances.reflectd(complex(0,-2))) ≈ 0 - 0.5im

    e = @inferred SpectralDistances.hproots(randn(7))
    ed = @inferred DiscreteRoots(e)
    @test real(e) == real.(e)
    @test real(ed) == real.(ed)
    @test issorted(ed.r, by=angle)
    @test all(<(1) ∘ abs, ed)
    ec = log(ed)
    @test issorted(ec, by=SpectralDistances.imageigsortby)
    @test all(<(0) ∘ real, ec)
    @test @inferred(domain_transform(Continuous(), ed)) isa ContinuousRoots
    @test domain_transform(Continuous(), ed) ≈ SpectralDistances.eigsort(SpectralDistances.reflectc.(log.(ed))) ≈ ContinuousRoots(ed)
    @test @inferred(domain_transform(Discrete(), ed)) == ed
    @test domain(ed) isa Discrete
    @test domain(ContinuousRoots(ed)) isa Continuous
    @test log(ed) isa ContinuousRoots

    @test all(<(1) ∘ abs, reflect(ed))
    @test all(<(0) ∘ real, reflect(ec))

    E = embedding(ec)
    @test length(E) == 12
    @test E == embedding(Vector, ec)
    E = embedding(Matrix, ec)
    @test size(E) == (6,2)
    E = embedding(Matrix, ec, false)
    @test size(E) == (3,2)

    @test_logs (:error, r"real poles") embedding(ContinuousRoots([0,1]), false)



    @test SpectralDistances.determine_domain(0.1randn(10)) isa Discrete
    @test SpectralDistances.determine_domain(randn(10).-4) isa Continuous
    @test_throws Exception SpectralDistances.determine_domain(0.1randn(10).-0.3)


    @testset "weight functions" begin
        @info "Testing weight functions"
        r = @inferred ContinuousRoots(randn(ComplexF64, 4))
        m = @inferred AR(r, 2.0)
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
    @info "Testing ControlSystems interoperability"
    m = @inferred AR(ContinuousRoots([-1]))
    g = tf(1,[1.,1])
    @test tf(m) == g
    @test_broken tf(m,1) == c2d(g,1)
    @test m*g == g*g
    @test ARMA(g).ac == [1,1]
    @test ARMA(g).bc == [1]
    @test @inferred(denvec(Continuous(), m)) == denvec(g)[1]
    @test numvec(Continuous(), m) == numvec(g)[1]
    @test @inferred(pole(Continuous(), m)) == pole(g)
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
    @test @inferred(polyconv(a,b)) ≈ DSP.conv(a,b)

    a = randn(5)
    b = randn(10)
    @test polyconv(a,b) ≈ DSP.conv(a,b)
    @test polyconv(b,a) ≈ DSP.conv(b,a)

    a[1] = 1
    @test @inferred(roots2poly(roots(reverse(a)))) ≈ a


end

@testset "preprocess roots with residue weight" begin
    rd  = @inferred EuclideanRootDistance(domain=Continuous(), weight=residueweight)
    m = @inferred AR([1., -0.1])
    @test @inferred(rd(m,m)) < eps()
    @test @inferred(SpectralDistances.preprocess_roots(rd, m))[] ≈ -2.3025850929940455
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
    r   = @inferred SpectralDistances.hproots(reverse(a))
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

    res = @inferred residues(a,1)
    @test sum(eachindex(r)) do i
        res[i]/(w-r[i])
    end ≈ F

    @test abs2.(res) ≈ SpectralDistances.abs2residues!(zeros(length(a)-1),a,1)

    @test prod(eachindex(r)) do i
        1/(w-r[i])
    end ≈ F

    n = 4
    a1 = randn(n+1); a1[1] = 1
    r1 = SpectralDistances.reflectc.(hproots(reverse(a1)))
    r1 = complex.(0.01real.(r1), imag.(r1))
    r1 = @inferred SpectralDistances.normalize_energy(ContinuousRoots(r1))

    a2 = randn(n+1); a2[1] = 1
    r2 = SpectralDistances.reflectc.(hproots(reverse(a2)))
    r2 = complex.(0.01real.(r2), imag.(r2))
    r2o = @inferred SpectralDistances.normalize_energy(ContinuousRoots(r2))

    m1,m2 = AR(a1), AR(a2)

    dist2 = @inferred RationalOptimalTransportDistance(domain=Continuous(), p=2, interval=(-20.,20.))
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
        d = @inferred D(domain=Continuous())
        println(D)
        @test @inferred(d(m,m)) < eps() + 0.001*(d isa OptimalTransportRootDistance)
        d isa Union{RationalOptimalTransportDistance, RationalCramerDistance} && continue
        d = D(domain=Discrete())
        println(D)
        @test d(m,m) < eps() + 0.001*(d isa OptimalTransportRootDistance)
    end
end

@testset "d(m,m̃)" begin
    a1 = [1,-0.1,0.8]
    m1 = @inferred AR(a1)
    a2 = [1,-0.1,0.801]
    m2 = AR(a2)
    for D in [  subtypes(SpectralDistances.AbstractDistance);
                subtypes(SpectralDistances.AbstractRootDistance);
                subtypes(SpectralDistances.AbstractCoefficientDistance)]
        (!isempty(methods(D)) && (:domain ∈ fieldnames(D))) || continue
        d = D(domain=Continuous())
        println(D)
        # @show d(m1,m2)
        @test @inferred(d(m1,m2)) > 1e-10
        d isa Union{RationalOptimalTransportDistance, RationalCramerDistance} && continue
        d = D(domain=Discrete())
        println(D)
        # @show d(m1,m2)
        @test d(m1,m2) > 1e-10
    end
end

@testset "distmat" begin
    e = complex.(randn(3), randn(3))
    D = @inferred SpectralDistances.distmat(SqEuclidean(), e)
    @test issymmetric(D)
    @test tr(D) == 0
    @test @inferred(SpectralDistances.distmat_euclidean(e,e)) ≈ D

    A = randn(10,10)
    A[diagind(A)] .= rand(10);
    A = A + A';
    B = SparseMatrixCSC(A)

    symmetrize!(A)
    symmetrize!(B)
    @test A == B
    @test issymmetric(A)
end


@testset "Welch" begin
    x1 = @inferred SpectralDistances.bp_filter(randn(3000), (0.01,0.1))
    x2 = SpectralDistances.bp_filter(randn(3000), (0.01,0.12))
    x3 = SpectralDistances.bp_filter(randn(3000), (0.01,0.3))
    dist = @inferred WelchOptimalTransportDistance(p=1)
    @test @inferred(dist(x1,x2)) < dist(x1,x3)
    dist = WelchOptimalTransportDistance(p=2)
    @test dist(x1,x2) < dist(x1,x3)

    dist = @inferred WelchLPDistance(p=1)
    @test @inferred(dist(x1,x2)) < dist(x1,x3)
    dist = WelchLPDistance(p=2)
    @test dist(x1,x2) < dist(x1,x3)

    dist = @inferred EnergyDistance()
    @test @inferred(dist(x1,x2)) < dist(x1,x3)
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
    dist = @inferred KernelWassersteinRootDistance(domain=Continuous())
    @test @inferred(dist(x1,x2)) < dist(x1,x3)
end

@testset "ClosedForm" begin
    x1 = SpectralDistances.bp_filter(randn(3000), (0.01,0.1))
    x2 = SpectralDistances.bp_filter(randn(3000), (0.01,0.12))
    x3 = SpectralDistances.bp_filter(randn(3000), (0.01,0.3))
    fm = LS(na=4)
    dist = ModelDistance(fm, RationalOptimalTransportDistance(domain=Continuous(), p=1, interval=(-15., 15)))
    @test dist(x1,x2) < dist(x1,x3)
    dist = RationalOptimalTransportDistance(domain=Continuous(), p=1, interval=(0., 15.))
    @test dist(fm(x1),welch_pgram(x2)) < dist(fm(x1),welch_pgram(x3))
    @test dist(fm(x1),welch_pgram(x2)) < dist(fm(x1),welch_pgram(x3))

    dist = ModelDistance(fm, RationalOptimalTransportDistance(domain=Continuous(), p=2, interval=(-15., 15)))
    @test dist(x1,x2) < dist(x1,x3)
    dist = RationalOptimalTransportDistance(domain=Continuous(), p=2, interval=(0., 15.))
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
end

@testset "barycenter" begin
    @info "Testing barycenter"
    include("test_barycenter.jl")
end

@testset "Slerp" begin
    a = n1([1,2,3])
    b = n1([4,4,3])
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



@testset "Unbalanced transport" begin
    @info "Testing Unbalanced transport"

    fm = LS(na = 10)
    m1 = fm(filtfilt(ones(10), [10], randn(1000)))
    m2 = fm(filtfilt(ones(5), [5], randn(1000)))
    dist = OptimalTransportRootDistance(domain = Continuous(), p=1, divergence=Balanced())
    d1 = evaluate(dist,m1,m2)
    dist = OptimalTransportRootDistance(domain = Continuous(), p=1, divergence=KL(1.0))
    d2 = evaluate(dist,m1,m2)
    dist = OptimalTransportRootDistance(domain = Continuous(), p=1, divergence=KL(10.0))
    d3 = evaluate(dist,m1,m2)
    dist = OptimalTransportRootDistance(domain = Continuous(), p=1, divergence=KL(0.01))
    d4 = evaluate(dist,m1,m2)

    @test d1 > d3 > d2 > d4
end

end

##
