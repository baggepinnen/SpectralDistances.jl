import Distances.evaluate

abstract type AbstractDistance <: Distances.Metric end
abstract type AbstractModelDistance <: AbstractDistance end
abstract type AbstractRootDistance <: AbstractDistance end
abstract type AbstractCoefficientDistance <: AbstractDistance end

const DistanceCollection = Union{Tuple, Vector{<:AbstractDistance}}

evaluate(d::DistanceCollection,x,y) = sum(evaluate(d,x,y) for d in d)

struct EuclideanCoefficientDistance <: AbstractCoefficientDistance
end
struct InnerProductCoefficientDistance <: AbstractCoefficientDistance
end

struct ModelDistance <: AbstractModelDistance
    fitmethod::FitMethod
    distance::AbstractDistance
end

@kwdef struct EuclideanRootDistance{D,A,F} <: AbstractRootDistance
    domain::D
    assignment::A = SortAssignement(imag)
    transform::F = identity
end

struct HungarianRootDistance{D} <: AbstractRootDistance
    domain::D
end

@kwdef struct KernelWassersteinRootDistance{D,F} <: AbstractRootDistance
    domain::D
    λ::Float64   = 1.
    transform::F = identity
end

struct OptimalTransportModelDistance{WT,DT} <: AbstractDistance
    w::WT
    distmat::DT
end

struct OptimalTransportSpectralDistance{DT} <: AbstractDistance
    distmat::DT
end

struct OptimalTransportHistogramDistance{DT} <: AbstractDistance
end

struct EnergyDistance <: AbstractDistance
end



domain(d::AbstractRootDistance) = d.domain
domain(d::AbstractModelDistance) = domain(d.distance)
domain(d) = throw(ArgumentError("This type does not define domain"))

transform(d::AbstractRootDistance) = d.transform
transform(d::AbstractRootDistance, x) = d.transform(x)


distmat_euclidean(e1::AbstractArray,e2::AbstractArray) = abs2.(e1 .- transpose(e2))
distmat_euclidean(e1,e2) = (n = length(e1[1]); [(e1[1][i]-e2[1][j])^2 + (e1[2][i]-e2[2][j])^2 for i = 1:n, j=1:n])

distmat_logmag(e1::AbstractArray,e2::AbstractArray) = distmat_euclidean(logmag.(e1), logmag.(e2))

evaluate(d::AbstractRootDistance,w1::AbstractModel,w2::AbstractModel) = evaluate(d, roots(w1), roots(w2))
evaluate(d::AbstractRootDistance,w1::ARMA,w2::ARMA) = evaluate(d, pole(w1), pole(w2)) + evaluate(d, tzero(w1), tzero(w2))

function evaluate(d::HungarianRootDistance, e1::AbstractVector{<:Complex},e2::AbstractVector{<:Complex})
    e1,e2 = domain_transform(d,e1), domain_transform(d,e2)
    dm    = distmat_euclidean(e1,e2)
    hungarian(dm)[2]
end


function evaluate(d::HungarianRootDistance, e1::AbstractRoots, e2::AbstractRoots)
    e1,e2 = domain_transform(d,e1), domain_transform(d,e2)
    e1,e2 = toreim(e1), toreim(e2)
    n = length(e1[1])
    dm = [(e1[1][i]-e2[1][j])^2 + (e1[2][i]-e2[2][j])^2 for i = 1:n, j=1:n]
    c = hungarian(dm)[2]
    # P,c = hungarian(Flux.data.(dm))
    # mean([(e1[1][i]-e2[1][j])^2 + (e1[2][i]-e2[2][j])^2 for i = 1:n, j=P])
end

function kernelsum(e1,e2,λ)
    s = 0.
    λ = -λ
    @inbounds for i in eachindex(e1)
        s += exp(λ*abs(e1[i]-e2[i])^2)
        for j in i+1:length(e2)
            s += 2exp(λ*abs(e1[i]-e2[j])^2)
        end
    end
    s / length(e1)^2
end

function logkernelsum(e1,e2,λ)
    s = 0.
    λ = -λ
    le2 = logreim.(e2)
    @inbounds for i in eachindex(e1)
        le1 = logreim(e1[i])
        s += exp(λ*abs2(le1-le2[i]))
        for j in i+1:length(e2)
            s += 2exp(λ*abs2(le1-le2[j]))
        end
    end
    s / length(e1)^2
end

function evaluate(d::EuclideanRootDistance, e1::AbstractRoots,e2::AbstractRoots)
    e1,e2 = domain_transform(d,e1), domain_transform(d,e2)
    e1    = d.transform.(e1)
    e2    = d.transform.(e2)
    I1,I2 = d.assignment(e1, e2)
    sum(abs, e1[I1]-e2[I2])
end

function eigval_dist_wass_logmag_defective(d::KernelWassersteinRootDistance, e1::AbstractRoots,e2::AbstractRoots)
    e1,e2 = domain_transform(d,e1), domain_transform(d,e2)
    λ     = d.λ
    e1    = d.transform.(e1)
    e2    = d.transform.(e2)
    error("this does not yield symmetric results")
    # e1 = [complex((logmag(magangle(e1)))...) for e1 in e1]
    # e2 = [complex((logmag(magangle(e2)))...) for e2 in e2]
    # e1    = logreim.(e1)
    # e2    = logreim.(e2)
    # e1    = sqrtreim.(e1)
    # e2    = sqrtreim.(e2)
    dm1   = logkernelsum(e1,e1,λ)
    dm2   = logkernelsum(e2,e2,λ)
    dm12  = logkernelsum(e1,e2,λ)
    dm1 - 2dm12 + dm2
end
function evaluate(d::KernelWassersteinRootDistance, e1::AbstractRoots,e2::AbstractRoots)
    e1,e2 = domain_transform(d,e1), domain_transform(d,e2)
    λ     = d.λ
    e1    = d.transform.(e1)
    e2    = d.transform.(e2)
    dm1   = exp.(.- λ .* distmat_euclidean(e1,e1))
    dm2   = exp.(.- λ .* distmat_euclidean(e2,e2))
    dm12  = exp.(.- λ .* distmat_euclidean(e1,e2))
    mean(dm1) - 2mean(dm12) + mean(dm2)
end

evaluate(d::AbstractCoefficientDistance,w1::AbstractModel,w2::AbstractModel) = evaluate(d, coefficients(w1), coefficients(w2))

function evaluate(d::EuclideanCoefficientDistance,w1::AbstractArray,w2::AbstractArray)
    sqrt(mean(abs2,w1-w2))
end

function evaluate(d::ModelDistance,X,Xh)
    w = fitmodel(d.fitmethod, X)
    wh = fitmodel(d.fitmethod, Xh)
    evaluate(d.distance, w, wh)
end


function evaluate(d::InnerProductCoefficientDistance,w1::AbstractArray,w2::AbstractArray)
    (1-w1'w2/norm(w1)/norm(w2))
end

# function ls_loss_eigvals_disc(d,X,Xh,order)
#     r = ar(X, order) |> polyroots
#     rh = ar(Xh, order) |> polyroots
#     eigval_dist_wass(d,r,rh)
# end
#
#
# function ls_loss_eigvals_cont(d,X,Xh,order)
#     r = ar(X, order) |> polyroots .|> log
#     rh = ar(Xh, order) |> polyroots .|> log
#     # weighted_eigval_dist_hungarian(r,rh)
#     eigval_dist_wass(d,r,rh)
# end
#
# function ls_loss_eigvals_cont_logmag(d,X,Xh,order)
#     r = ar(X, order) |> polyroots .|> log
#     rh = ar(Xh, order) |> polyroots .|> log
#     # weighted_eigval_dist_hungarian(r,rh)
#     eigval_dist_wass_logmag(d,r,rh)
# end


function batch_loss(bs::Int, loss, X, Xh)
    l = zero(eltype(Xh))
    lx = length(X)
    n_batches = length(X)÷bs
    inds = 1:bs
    # TODO: introduce overlap for smoother transitions  #src
    for i = 1:n_batches
        l += loss(X[inds],Xh[inds])
        inds = inds .+ bs
    end
    l *= bs
    residual_inds = inds[1]:lx
    lr = length(residual_inds)
    lr > 0 && (l += loss(X[residual_inds],Xh[residual_inds])*lr)
    l /= length(X)
    l / n_batches
end

evaluate(d::EnergyDistance,X::AbstractArray,Xh::AbstractArray) = (std(X)-std(Xh))^2 + (mean(X)-mean(Xh))^2




function OptimalTransportModelDistance(w = exp10.(LinRange(-3, 0.5, 300)))
    distmat = zeros(length(w), length(w))
    for i=1:n
        for j=1:n
            distmat[i, j] = abs((i/n)-(j/n))
        end
    end
    OptimalTransportModelDistance(w,distmat)
end

function OptimalTransportSpectralDistance(w)
    distmat = zeros(length(w), length(w))
    for i=1:n
        for j=1:n
            distmat[i, j] = abs((i/n)-(j/n))
        end
    end
    OptimalTransportSpectralDistance(distmat)
end


function evaluate(d::OptimalTransportModelDistance, m1, m2)
    noise_model = tf(m1)
    b1,_,_ = bode(noise_model, w.*2pi) .|> vec
    b1 .= log.(b1)
    b1 .-= minimum(b1)
    b1 ./= sum(b1)
    noise_model = tf(m2)
    b2,_,_ = bode(noise_model, w.*2pi) .|> vec
    b2 .= log.(b2)
    b2 .-= minimum(b2)
    b2 ./= sum(b2)
    plan = sinkhorn_plan(d.distmat, b1, b2; ϵ=1/10, rounds=300)
    # plan = sinkhorn_plan_log(distmat, b1, b2; ϵ=1/10, rounds=300)
    cost = sum(plan .* d.distmat)
end

function evaluate(d::OptimalTransportSpectralDistance, w1, w2)
    plan = sinkhorn_plan(d.distmat, w1, w2; ϵ=1/10, rounds=300)
    # plan = sinkhorn_plan_log(distmat, b1, b2; ϵ=1/10, rounds=300)
    cost = sum(plan .* d.distmat)
end


centers(x) = 0.5*(x[1:end-1] + x[2:end])
function evaluate(d::OptimalTransportHistogramDistance, x1, x2)
    h1 = fit(Histogram,x1)
    h2 = fit(Histogram,x2)
    distmat = [abs(e1-e2) for e1 in centers(h1.edges[1]), e2 in centers(h2.edges[1])]
    plan = sinkhorn_plan(distmat, s1(h1.weights), s1(h2.weights); ϵ=1\10, rounds=300)
    # plan = sinkhorn_plan_log(distmat, b1, b2; ϵ=1/10, rounds=300)
    cost = sum(plan .* distmat)
end



# for dt in [ EuclideanCoefficientDistance,
#             InnerProductCoefficientDistance,
#             ModelDistance,
#             EuclideanRootDistance,
#             HungarianRootDistance,
#             KernelWassersteinRootDistance,
#             OptimalTransportModelDistance,
#             OptimalTransportSpectralDistance,
#             EnergyDistance,
#             DistanceCollection]
#     @eval (d::$dt)(x,y) = evaluate(d, x, y)
# end

(d::AbstractDistance)(x,y) = evaluate(d, x, y)
(d::DistanceCollection)(x,y) = evaluate(d, x, y)



function trivial_transport(x,y)
    x = copy(x)
    yf = zero(y)
    n = length(x)
    g = zeros(n,n)
    i = j = 1
    while j <= n && i <= n
        needed = y[j] - yf[j]
        available = x[i]
        if available >= needed
            g[i,j] += needed
            x[i] -= needed
            j += 1
        else
            i += 1
        end

    end
    g
end

Base.inv(f::Function) = x->finv(f, x)
# Base.inv(f::Function,fp) = x->finv(f, fp, x)

function finv(f, z)
    w = Roots.fzero(w->f(w)-z, 0, 2π)
end

# function finv(f, fp, z)
#     w = Roots.fzero(w->f(w)-z, fp, π)
# end

∫(f,a,b) = quadgk(f,a,b; atol=1e-10, rtol=1e-7)[1]::Float64
∫0(f,b) = quadgk(f,0,b; atol=1e-10, rtol=1e-7)[1]::Float64
function c∫(f,a,b)
    fi    = (u,p,t) -> f(t)
    tspan = (a,b)
    prob  = ODEProblem(fi,0.,tspan)
    sol   = solve(prob,Tsit5(),reltol=1e-7,abstol=1e-8)
end


function closed_form_wass(a1,a2,p=1)
    n     = length(a1)
    f1    = w -> abs2(1/sum(j->a1[j]*exp(im*w*(n-j)), 1:n))
    f2    = w -> abs2(1/sum(j->a2[j]*exp(im*w*(n-j)), 1:n))
    σ1    = ∫0(f1,2π)
    σ2    = ∫0(f2,2π)
    f1    = w -> abs2(1/sum(j->a1[j]*exp(im*w*(n-j)), 1:n)) / σ1
    f2    = w -> abs2(1/sum(j->a2[j]*exp(im*w*(n-j)), 1:n)) / σ2
    F1(w) = ∫0(f1,w)
    F2(w) = ∫0(f2,w)
    ∫0(z->abs(inv(F1)(z) - inv(F2)(z))^p, 1)
end

function functionbarrier(sol1::T1,sol2::T2,p) where {T1,T2}
    σ1    = sol1(2pi)
    σ2    = sol2(2pi)
    F1(w) = sol1(w)/σ1
    F2(w) = sol2(w)/σ2
    ∫0(2π) do w
        F1w = F1(w)
        F2w = F2(w)
        abs(F1w-F2w)^p
    end
end
function closed_form_wass_noinverse(a1,a2,p=1)
    n     = length(a1)
    f1    = w -> abs2(1/sum(j->a1[j]*exp(im*w*(n-j)), 1:n))
    f2    = w -> abs2(1/sum(j->a2[j]*exp(im*w*(n-j)), 1:n))
    sol1  = c∫(f1,0,2π)
    sol2  = c∫(f2,0,2π)
    functionbarrier(sol1,sol2,p)
end

# This implementation has quadratic runtime due to integration inside integration
function closed_form_wass_noinverse_slow(a1,a2,p=1)
    n     = length(a1)
    f1    = w -> abs2(1/sum(j->a1[j]*exp(im*w*(n-j)), 1:n))
    f2    = w -> abs2(1/sum(j->a2[j]*exp(im*w*(n-j)), 1:n))
    σ1    = ∫0(f1,2π)
    σ2    = ∫0(f2,2π)
    f1    = w -> abs2(1/sum(j->a1[j]*exp(im*w*(n-j)), 1:n)) / σ1
    f2    = w -> abs2(1/sum(j->a2[j]*exp(im*w*(n-j)), 1:n)) / σ2
    F1(w) = ∫0(f1,w)
    F2(w) = ∫0(f2,w)
    ∫0(2π) do w
        F1w = F1(w)
        F2w = F2(w)
        abs(F1w-F2w)^p
    end
end

function closed_form_log_wass(a1,a2,p=1)
    n     = length(a1)
    f1    = w -> -log(abs2(sum(j->a1[j]*exp(im*w*(n-j)), 1:n))) + 1/(2π)
    f2    = w -> -log(abs2(sum(j->a2[j]*exp(im*w*(n-j)), 1:n))) + 1/(2π)
    endpoint = min(∫0(f1,2π), ∫0(f2,2π))
    F1(w) = ∫0(f1,w)
    F2(w) = ∫0(f2,w)
    ∫0(z->abs(inv(F1)(z) - inv(F2)(z))^p, endpoint)
end
