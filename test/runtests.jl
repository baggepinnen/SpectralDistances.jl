@info "Running tests"
using SpectralDistances, Distributions
using Test, LinearAlgebra, Statistics, Random
using ForwardDiff, DSP


Random.seed!(0)

@testset "SpectralDistances.jl" begin


    y = param(randn(10))
    u = param(randn(10))
    using ForwardDiff
    @test Flux.jacobian((y)->vec(getARXregressor(y,u,2,2)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,2,2)[2]), y.data)
    @test Flux.jacobian((y)->vec(getARXregressor(y,u,2,2)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,2,2)[1]), y.data)
    @test Flux.jacobian((y)->vec(getARXregressor(y,u,5,2)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,5,2)[2]), y.data)
    @test Flux.jacobian((y)->vec(getARXregressor(y,u,5,2)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,5,2)[1]), y.data)

    @test Flux.jacobian((y)->vec(getARXregressor(y,u,2,1)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,2,1)[2]), y.data)
    @test Flux.jacobian((y)->vec(getARXregressor(y,u,2,1)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,2,1)[1]), y.data)
    @test Flux.jacobian((y)->vec(getARXregressor(y,u,5,1)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,5,1)[2]), y.data)
    @test Flux.jacobian((y)->vec(getARXregressor(y,u,5,1)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARXregressor(y,u.data,5,1)[1]), y.data)



    y = param(randn(10))
    @test Flux.jacobian((y)->vec(getARregressor(y,2)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARregressor(y,2)[2]), y.data)
    @test Flux.jacobian((y)->vec(getARregressor(y,2)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARregressor(y,2)[1]), y.data)
    @test Flux.jacobian((y)->vec(getARregressor(y,5)[2]), y.data) == ForwardDiff.jacobian((y)->vec(getARregressor(y,5)[2]), y.data)
    @test Flux.jacobian((y)->vec(getARregressor(y,5)[1]), y.data) == ForwardDiff.jacobian((y)->vec(getARregressor(y,5)[1]), y.data)



    let Gc = tf(1,[1,1,1,1])
        w = c2d(Gc,1).matrix[1] |> ControlSystems.denvec
        @test d2c(w) ≈ pole(Gc)
    end

    y = randn(5000)
    plr(y,40,2)
    @test Flux.gradient(y->sum(plr(y,4,2)[1]), y)[1][1].data ≈ ForwardDiff.gradient(y->sum(plr(y,4,2)[1]), y)
    @test Flux.gradient(y->sum(plr(y,4,2)[2]), y)[1][1].data ≈ ForwardDiff.gradient(y->sum(plr(y,4,2)[2]), y)


    p = [1.,1,1]
    # @btime riroots(p)
    fd = central_fdm(3, 1)
    @test Flux.gradient(p) do p
        r = riroots(p)
        sum([r[1]; r[2]])
    end[1][1].data ≈ FiniteDifferences.grad(fd, p->begin
        r = roots(p)
        sum([real(r); imag(r)])
    end, p)





    e1 = toreim(roots([1.,1,1]))
    e2 = toreim(roots([1.,1.2,1]))
    @test eigval_dist_wass(e1,e1) < 0.01
    @test eigval_dist_wass(e1,e2) == eigval_dist_wass(e2,e1)
    @test eigval_dist_wass(e1,e1) < eigval_dist_wass(e1,e2)
    @test eigval_dist_wass(e2,e2) < eigval_dist_wass(e1,e2)
    @test eigval_dist_wass(e1,e2,1) < eigval_dist_wass(e1,e2,10)



    X = param(randn(100))
    Xh = param(0.5copy(-X.data))
    Xh = param(randn(100))
    ls_loss(X,Xh,10)
    @test sum(abs, Flux.gradient(Xh->batch_loss(10, (X,Xh)->ls_loss(X,Xh,2), X, Xh), Xh)[1][1].data - ForwardDiff.gradient(Xh->batch_loss(10, (X,Xh)->ls_loss(X.data,Xh,2), X, Xh), Xh.data)) < 1e-6

    @test Flux.gradient((Xh)->ls_loss(X,Xh,2), Xh)[1][1] ≈ ForwardDiff.gradient((Xh)->ls_loss(X.data,Xh,2), Xh.data)
    @test Flux.gradient((Xh)->ls_loss_angle(X,Xh,2), Xh)[1][1] ≈ ForwardDiff.gradient((Xh)->ls_loss_angle(X.data,Xh,2), Xh.data)
    @test Flux.gradient((Xh)->ls_loss_eigvals_disc(X,Xh,2), Xh)[1][1] ≈ ForwardDiff.gradient((Xh)->ls_loss_eigvals_disc(X.data,Xh,2), Xh.data)
    # Flux.gradient((Xh)->ls_loss_eigvals_disc(X,Xh,2), Xh)


    X = [filt(ones(10),[10], randn(10000)); filt(ones(2),[2], randn(10000))]
    Xh = (filt(ones(2),[2], randn(20000)))
    ls_loss(X,Xh,10)
    batch_loss(10000, (X,Xh)->ls_loss(X,Xh,10), X, Xh)
    # gs = gradient(()->sum(ls_loss(X,Xh,10)), params(X,Xh))  #src



    let t = 1:100
        ϵ = 1e-7
        @test ls_loss(randn(100), randn(100),10) > 0.05
        @test ls_loss(sin.(t), sin.(t),2) < ϵ
        @test ls_loss(sin.(t), -sin.(t),2) < ϵ # phase flip invariance
        @test ls_loss(sin.(t), sin.(t .+ 1), 2) < ϵ # phase shift invariance
        @test ls_loss(sin.(t), sin.(t .+ 0.1), 2) < ϵ # phase shift invariance
        @test ls_loss(10sin.(t), sin.(t .+ 1), 2) < ϵ # amplitude invariance
        @test ls_loss(sin.(t), sin.(1.1 .* t), 2) < 0.2 # small frequency shifts gives small errors
        @test ls_loss(sin.(0.1t), sin.(1.1 .* 0.1t), 2) < 0.1 # small frequency shifts gives small errors

        @test_broken ls_loss(sin.(t), sin.(1.1 .* t), 2) ≈ ls_loss(sin.(0.1t), sin.(1.1 .* 0.1t), 2) < 0.1 # frequency shifts of relative size should result in the same error

        @test ls_loss(filt(ones(10),[10], randn(1000)), filt(ones(10),[10], randn(1000)),10) < 0.2 # Filtered through same filter
        @test ls_loss(filt(ones(10),[10], randn(1000)), filt(ones(10),[10], randn(1000)),10) < ls_loss(filt(ones(9),[9], randn(1000)), filt(ones(10),[10], randn(1000)),10) # Filtered through different filters
    end

    x = [1.,0,0]
    y = [0,0.5,0.5]

    g = SpectralDistances.trivial_transport(x,y)
    @test sum(g,dims=1)[:] == y
    @test sum(g,dims=2)[:] == x



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

@test SpectralDistances.reflectd(2) ≈ 0.5 + 0.0im
@test SpectralDistances.reflectd(complex(0,2)) ≈ 0 + 0.5im
@test SpectralDistances.reflectd(complex(0,-2)) ≈ 0 - 0.5im




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




end

end



a1,a2 = [1,1,0.8], [1,1,0.8]
@test SpectralDistances.closed_form_wass(a1,a2) == 0
@test SpectralDistances.closed_form_log_wass(a1,a2) == 0
