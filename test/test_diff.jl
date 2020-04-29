using SpectralDistances, Zygote
using SpectralDistances:
    ngradient, nhessian, njacobian, polyconv, eigsort, hproots, rev, companion

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




# true && get(ENV, "TRAVIS_BRANCH", nothing) == nothing && @testset "gradients" begin

using FiniteDifferences
using Zygote
using SpectralDistances: getARXregressor, getARregressor
y = (randn(10))
u = (randn(10))
@test ngradient((y) -> sum(getARXregressor(y, u, 2, 2)[2]), y) ==
      Zygote.gradient((y) -> sum(getARXregressor(y, u, 2, 2)[2]), y)[1]
@test ngradient((y) -> sum(getARXregressor(y, u, 2, 2)[1]), y) ==
      Zygote.gradient((y) -> sum(getARXregressor(y, u, 2, 2)[1]), y)[1]
@test ngradient((y) -> sum(getARXregressor(y, u, 5, 2)[2]), y) ==
      Zygote.gradient((y) -> sum(getARXregressor(y, u, 5, 2)[2]), y)[1]
@test ngradient((y) -> sum(getARXregressor(y, u, 5, 2)[1]), y) ==
      Zygote.gradient((y) -> sum(getARXregressor(y, u, 5, 2)[1]), y)[1]

@test ngradient((y) -> sum(getARXregressor(y, u, 2, 1)[2]), y) ==
      Zygote.gradient((y) -> sum(getARXregressor(y, u, 2, 1)[2]), y)[1]
@test ngradient((y) -> sum(getARXregressor(y, u, 2, 1)[1]), y) ==
      Zygote.gradient((y) -> sum(getARXregressor(y, u, 2, 1)[1]), y)[1]
@test ngradient((y) -> sum(getARXregressor(y, u, 5, 1)[2]), y) ==
      Zygote.gradient((y) -> sum(getARXregressor(y, u, 5, 1)[2]), y)[1]
@test ngradient((y) -> sum(getARXregressor(y, u, 5, 1)[1]), y) ==
      Zygote.gradient((y) -> sum(getARXregressor(y, u, 5, 1)[1]), y)[1]



y = (randn(10))
@test ngradient((y) -> sum(getARregressor(y, 2)[2]), y) ==
      Zygote.gradient((y) -> sum(getARregressor(y, 2)[2]), y)[1]
@test ngradient((y) -> sum(getARregressor(y, 2)[1]), y) ==
      Zygote.gradient((y) -> sum(getARregressor(y, 2)[1]), y)[1]
@test ngradient((y) -> sum(getARregressor(y, 5)[2]), y) ==
      Zygote.gradient((y) -> sum(getARregressor(y, 5)[2]), y)[1]
@test ngradient((y) -> sum(getARregressor(y, 5)[1]), y) ==
      Zygote.gradient((y) -> sum(getARregressor(y, 5)[1]), y)[1]

a, b = randn(3), randn(4)
@test ngradient(a -> sum(SpectralDistances.polyconv(a, b)), a) ≈
      Zygote.gradient(a -> sum(SpectralDistances.polyconv(a, b)), a)[1] rtol = 1e-3
@test ngradient(b -> sum(SpectralDistances.polyconv(a, b)), b) ≈
      Zygote.gradient(b -> sum(SpectralDistances.polyconv(a, b)), b)[1] rtol = 1e-3


@test ngradient(a -> sum(SpectralDistances.polyvec(a)), a) ≈
      Zygote.gradient(a -> sum(SpectralDistances.polyvec(a)), a)[1] rtol = 1e-3




@test_skip let Gc = tf(1, [1, 1, 1, 1])
    w = c2d(Gc, 1).matrix[1] |> ControlSystems.denvec
    @test d2c(w) ≈ pole(Gc)
end

# y = randn(5000)
# fm = PLR(na=40, nc=2)
# @test_skip Zygote.gradient(y->sum(fm(y)[1]), y)[1] ≈ ForwardDiff.gradient(y->sum(fm(y)[1]), y)
# @test_skip Zygote.gradient(y->sum(fm(y)[2]), y)[1] ≈ ForwardDiff.gradient(y->sum(fm(y)[2]), y)


p = [1.0, 1, 1]
# @btime riroots(p)
fd = central_fdm(3, 1)
@test Zygote.gradient(p) do p
    r = roots(p)
    sum(abs2, r)
end[1][1:end-1] ≈ FiniteDifferences.grad(fd, p -> begin
    r = roots(p)
    sum(abs2, r)
end, p)[1][1:end-1]

fm = TLS(na = 4)
y = randn(50)
sum(abs2, fm(y).pc)
@test_broken Zygote.gradient(y) do y
    sum(abs2, fitmodel(fm, y, true).pc)
end



fdm = central_fdm(5, 1)

a = randn(5)
a[end] = 1
f = x -> sum(abs2(r) for r in (hproots(x)))

G = Zygote.gradient(f, a)[1]
@test G[1:end-1] ≈ FiniteDifferences.grad(fdm, f, a)[1][1:end-1]


fd = central_fdm(5, 1)
a = randn(30)
a[1] = 1
r = roots(reverse(a)) |> ContinuousRoots
@inferred residues(a, 1, r)
f = a -> sum(abs2, residues(a, 1, r))
g = a -> real(residues(complex.(a), 1, r)[2])
@test sum(abs, f'(complex.(a))[2:end] - grad(fd, f, a)[1][2:end]) < sqrt(eps())
@test sum(abs, g'((a))[2:end] - grad(fd, g, a)[1][2:end]) < sqrt(eps())

fd = central_fdm(7, 1)
a = randn(30)
a[1] = 1
residues(a, 1)
f = a -> sum(abs2, residues(a, 1))
g = a -> real(residues((a), 1)[2])
@test sum(abs, f'((a))[2:end] - grad(fd, f, a)[1][2:end]) < 1e-5
@test_skip sum(abs, g'((a))[2:end] - grad(fd, g, a)[1][2:end]) < sqrt(eps()) # Not robust

@testset "Numerical curvature" begin
    @info "Testing Numerical curvature"
    a = randn(11)
    a[1] = 1
    using DoubleFloats
    m = AR(ContinuousRoots([-Double64(0.1) + im, -Double64(0.1) - im]))
    a = Vector(m.ac)
    for dist in [
        EuclideanRootDistance(domain = Continuous(), p = 1, weight = residueweight),
        OptimalTransportRootDistance(domain = Continuous(), p = 2, β = 0.01),
    ]
        H = SpectralDistances.curvature(dist, a)
        @test all(>(0) ∘ real, eigvals(H[2:end, 2:end]))
        @test all(==(0) ∘ imag, eigvals(H[2:end, 2:end]))
    end

end

# Zygote segfaults here
# @testset "Sinkhorn zygote" begin
#     @info "testing sinkhorn zygote"
#     function sinkdist(D,a,b)
#         ai = s1(a)
#         bi = s1(a+b)
#         P,u,v = sinkhorn(D,ai,bi, iters=1000, β=0.1)
#         dot(P, D)
#     end
#     a,b = abs.(randn(6)),abs.(randn(6))
#     D = SpectralDistances.distmat_euclidean(1:length(a), 1:length(a))
#     dD, da, db = Zygote.gradient(sinkdist, D,a,b)
#     @test n1(ForwardDiff.gradient(a->sinkdist(D,a,b), a))'n1(da) > 0.9
# end

a = -5:-1
@test SpectralDistances.roots2poly(a) ≈ SpectralDistances.roots2poly_zygote(a)


##
cosdist(x1, x2) = x1'x2 / norm(x1) / norm(x2)
using SpectralDistances, Zygote
@testset "More roots diff tests" begin
    @info "Testing More roots diff tests"

    rootfun = eigsort ∘ hproots ∘ rev
    rootfunnr = eigsort ∘ hproots
    # Testing roots carefully
    a = polyconv([1.0, 1], [1.1, 1])
    b = polyconv([1.0, 1], [2, 1])

    @test SpectralDistances.roots(a) ≈ [-1.1, -1]
    @test SpectralDistances.roots(b) ≈ [-2, -1]

    @test hproots(a) ≈ eigsort(Complex.([-1.1, -1]))
    @test hproots(b) ≈ eigsort(Complex.([-2, -1]))

    g1 = Zygote.gradient(a -> sum(abs, rootfunnr(a)), a)[1]
    g2 = ngradient(a -> sum(abs, rootfunnr(a)), a)
    @test real(g1)[1:end-1] ≈ real(g2)[1:end-1]
    g1 = Zygote.gradient(b -> sum(abs, rootfunnr(b)), b)[1]
    g2 = ngradient(b -> sum(abs, rootfunnr(b)), b)
    @test real(g1)[1:end-1] ≈ real(g2)[1:end-1]

    a = [1, 0.1, 1]
    g1 = Zygote.gradient(a -> sum(abs, rootfunnr(a)), a)[1]
    g2 = ngradient(a -> sum(abs, rootfunnr(a)), a)
    @test real(g1)[1:end-1] ≈ real(g2)[1:end-1]

    a = [1, 0.1, 1]
    g1 = Zygote.gradient(a -> sum(abs2, rootfunnr(a)), a)[1]
    g2 = ngradient(a -> sum(abs2, rootfunnr(a)), a)
    @test real(g1)[1:end-1] ≈ real(g2)[1:end-1]

    a = [1, 0.1, 1]
    g1 = Zygote.gradient(a -> real(rootfunnr(a)[2]), a)[1]
    g2 = ngradient(a -> real(rootfunnr(a)[2]), a)
    @test real(g1)[1:end-1] ≈ real(g2)[1:end-1]

    g1 = Zygote.gradient(a -> imag(rootfunnr(a)[2]), a)[1]
    g2 = ngradient(a -> imag(rootfunnr(a)[2]), a)
    @test real(g1)[1:end-1] ≈ real(g2)[1:end-1]

    function getpoly(n, γ = 0.1)
        skew(x) = x - x' + 0.1 * (x'x)
        r = eigsort(eigvals(skew(γ .* randn(n, n))))
        a = Vector(SpectralDistances.roots2poly(r))
        a, r
    end
    a, r = getpoly(9)

    r1 = @inferred rootfun(a)
    r2 = @inferred eigsort(eigen(companion(rev(a))).values)
    @test cosdist(real(r1), real(r2)) > 0.99
    @test cosdist(real(r1), real(r)) > 0.99


    ##
    a = [1.0, 1]
    for i = 2:10
        isinteractive() && @show i
        # global a
        a = SpectralDistances.polyconv(a, [1, -i])
        isinteractive() && @show a
        # a[1] = 1
        # a[end] = 1

        g1 = Zygote.gradient(a -> sum(abs2, rootfun(a)), a)[1]
        g2 = ngradient(a -> sum(abs2, rootfun(a)), a)
        @test real(g1)[2:end] ≈ real(g2)[2:end] atol=1e-3

        g1 = Zygote.gradient(a -> real(rootfun(a)[2]), a)[1]
        g2 = ngradient(a -> real(rootfun(a)[2]), a)
        @test (g1)[2:end] ≈ (g2)[2:end] atol = 1e-4 atol=1e-3
        #
        g1 = Zygote.gradient(a -> imag(rootfun(a)[2]), a)[1]
        g2 = ngradient(a -> imag(rootfun(a)[2]), a)
        @test real(g1)[2:end] ≈ real(g2)[2:end] atol=1e-3
    end

    # NOTE: eigsort is improtant
    g1 = Zygote.gradient(a -> sum(abs2, rootfun((a))), (a))[1]
    g2 = ngradient(a -> sum(abs2, rootfun((a))), a)
    @test cosdist(real(g1)[2:end], real(g2)[2:end]) > 0.9
    @test real(g1)[2:end] ≈ real(g2)[2:end] rtol = 0.3

    g1 = Zygote.gradient(a -> real(rootfun(a)[2]), a)[1]
    g2 = ngradient(a -> real(rootfun(a)[2]), a)
    @test cosdist(g1[2:end], g2[2:end]) > 0.9
    @test g1[2:end] ≈ g2[2:end] rtol = 1e-3

    g1 = Zygote.gradient(a -> imag(rootfun(a)[2]), a)[1]
    g2 = ngradient(a -> imag(rootfun(a)[2]), a)
    @test real(g1)[2:end] ≈ real(g2)[2:end]





    ##
    # using FiniteDifferences
    # fdm = central_fdm(5, 1)
    t = 0:100
    x1 = sin.(0.70 .* t) .+ sin.(0.50 .* t) .+ 0.1 .* randn.()
    x2 = sin.(0.50 .* t) .+ sin.(0.30 .* t) .+ 0.1 .* randn.()
    fm = LS(na = 10)
    m1, m2 = fm.((x1, x2))

    testfun1(x1) = sum(abs2, fm(x1).a)
    @test Zygote.gradient(testfun1, x1)[1] ≈ grad(fdm, testfun1, x1)[1] rtol = 1e-3
    testfun2(x1) = sum(abs2, (fm(x1).p))
    g1 = Zygote.gradient(testfun2, x1)[1]
    g2 = ngradient(testfun2, x1, δ = 1e-4)
    # g3 = grad(fdm, testfun2, x1)[1]
    @test cosdist(g1, g2) > 0.95
    # cosdist(g3, g2)

    testfun3(x1) = sum(abs2, fm(x1).pc)
    g1 = Zygote.gradient(testfun3, x1)[1]
    g2 = ngradient(testfun3, x1)
    # g3 = grad(fdm, testfun3, x1)[1]
    @test cosdist(g1, g2) > 0.95
    # cosdist(g3, g2)

    dist = ModelDistance(fm, OptimalTransportRootDistance(domain = Continuous(), β = 0.1))
    Zygote.@nograd rand
    # Zygote.refresh()

    df = x -> real(evaluate(dist, x, x2))
    df(x1)
    g = Zygote.gradient(df, x1)[1]
    gn = ngradient(x -> evaluate(dist, x, x2, iters = 50, solver = sinkhorn_log), x1)

    @test cosdist(g, gn) > 0.9
    isinteractive() && plot([gn g], layout = 2)


end
# @btime Zygote.gradient(x->real(evaluate(dist,x,$x2)), $x1)
# @profiler Zygote.gradient(x->real(evaluate(dist,x,x2)), x1)
# using DSP
# function callback(x)
#     dump(x)
#     plot(welch_pgram(x))
#     display(plot!(welch_pgram(x2)))
#     false
# end
#
# using Optim
# res = Optim.optimize(
#     df,
#     df',
#     x1,
#     GradientDescent(),
#     Optim.Options(
#         store_trace = true,
#         show_trace = true,
#         show_every = 1,
#         iterations = 40,
#         allow_f_increases = true,
#         time_limit = 100,
#         x_tol = 0,
#         f_tol = 0,
#         g_tol = 1e-8,
#         f_calls_limit = 0,
#         g_calls_limit = 0,
#     ),
#     inplace = false,
# )
#
#
#
# # # Optimize using ADAM
# using Flux
# using Flux: Params, params#, gradient
# X = filt(ones(10), [10], randn(500)) |> v1
# Xp = randn(500) |> v1
#
# opt = ADAM(0.3)
# losses = Float64[]
# P = welch_pgram(X)
# shift1(x) = x .+ x[2]
# # plot3d(fill(0,length(P.power)), log10.(shift1(P.freq)), log10.(P.power), colorbar=false)
#
# lossobj = (
#     ModelDistance(
#         LS(na = 10),
#         OptimalTransportRootDistance(
#             domain = Continuous(),
#             β = 0.1,
#             p = 1,
#             weight = simplex_residueweight,
#         ),
#         # EuclideanRootDistance(domain = Continuous()),
#     ),
#     EnergyDistance(),
# )
#
# using Distances
# # lossobj = (
# #     ModelDistance(
# #         LS(na = 10),
# #         CoefficientDistance(domain = Continuous(), distance=Euclidean()),
# #     ),
# #     EnergyDistance(),
# # )
#
# # lossobj =
# #     ModelDistance(LS(na = 10), OptimalTransportRootDistance(domain = Continuous(), β = 0.1))
#
# loss = (X, Xp) -> evaluate(lossobj, X, Xp)
# gs = gradient(Xp -> real(loss(X, Xp)), Xp)
# loss(X, Xp)
# lossobj[1](X, Xp)
# lossobj[2](X, Xp)
# f1 = plot(P)
# iters = 200
# for i = 1:iters
#     gs = gradient(Xp -> real(loss(X, Xp)), Xp)[1]
#     l = loss(X, Xp)
#     push!(losses, l)
#     @show l
#     Flux.Optimise.update!(opt, Xp, gs)
#     if i % 20 == 0
#         P = welch_pgram(Xp)
#         plot!(f1, P, lab = "", line_z = i) |> display
#     end
# end
# display(current())
# #-
# plot(losses, yscale = :log10)


# gradcheck(f, xs...) =
#   all(isapprox.(ngradient(f, xs...),
#                 Zygote.gradient(f, xs...), rtol = 1e-5, atol = 1e-5))
#
# gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
# gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)
# i = 1

# correct = [
#       [2,3,1],
#       [1, 2, 3],
#       [1,2,3],
#       [2,1,3]
# ]
# for i = 1:3
#       @show i
#       @test Zygote.gradient(v->sort(v)[i], [3.,1,2])[1][correct[1][i]] == 1
#       @test Zygote.gradient(v->sort(v)[i], [1.,2,3])[1][correct[2][i]] == 1
#       @test Zygote.gradient(v->sort(v,by=x->x%10)[i], [11,2,99])[1][correct[3][i]] == 1
#       @test Zygote.gradient(v->sort(v,by=x->x%10)[i], [2,11,99])[1][correct[4][i]] == 1
# end




# a = randn(5)
# p = sortperm(a)
# as = a[p]
# a2 = as[invperm(p)]
# @test a == a2
