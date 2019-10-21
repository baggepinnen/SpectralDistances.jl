import Distances.evaluate

"The top level distance type"
abstract type AbstractDistance <: Distances.Metric end
"All subtypes of this type operates on rational transfer functions"
abstract type AbstractModelDistance <: AbstractDistance end
"All subtypes of this type operates on signals"
abstract type AbstractSignalDistance <: AbstractDistance end
"All subtypes of this type operates on the roots of rational transfer functions"
abstract type AbstractRootDistance <: AbstractModelDistance end
"All subtypes of this type operates on the coefficients of rational transfer functions"
abstract type AbstractCoefficientDistance <: AbstractModelDistance end

"A Union that represents a collection of distances"
const DistanceCollection = Union{Tuple, Vector{<:AbstractDistance}}

struct Identity end
struct Log end
magnitude(d) = Identity()


evaluate(d::DistanceCollection,x,y) = sum(evaluate(d,x,y) for d in d)
Base.:(+)(d::AbstractDistance...) = d

(d::AbstractDistance)(x,y) = evaluate(d, x, y)
(d::DistanceCollection)(x,y) = evaluate(d, x, y)

"""
    CoefficientDistance{D, ID} <: AbstractCoefficientDistance

Distance metric based on model coefficients

#Arguments:
- `domain::D`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `distance::ID = SqEuclidean()`: Inner distance between coeffs
"""
CoefficientDistance
@kwdef struct CoefficientDistance{D,ID} <: AbstractCoefficientDistance
    domain::D
    distance::ID = SqEuclidean()
end

"""
    ModelDistance{D <: AbstractDistance} <: AbstractSignalDistance

A model distance operates on signals and works by fitting an LTI model to the signals before calculating the distance. The distance between the LTI models is defined by the field `distance`. This is essentially a wrapper around the inner distance that handles the fitting of a model to the signals. How the model is fit is determined by `fitmethod`.

#Arguments:
- `fitmethod::FitMethod`: `LS` or `PLR`
- `distance::D`: The inner distance between the models
"""
struct ModelDistance{D <: AbstractDistance} <: AbstractSignalDistance
    fitmethod::FitMethod
    distance::D
end

"""
    EuclideanRootDistance{D, A, F1, F2} <: AbstractRootDistance

Simple euclidean distance between roots of transfer functions

#Arguments:
- `domain::D`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `assignment::A =` [`SortAssignement`](@ref)`(imag)`: Determines how roots are assigned. An alternative is `HungarianAssignement`
- `transform::F1 = identity`: DESCRIPTION
- `weight` : A function used to calculate weights for the induvidual root distances. A good option is [`residueweight`](@ref)
"""
EuclideanRootDistance
@kwdef struct EuclideanRootDistance{D,A,F1,F2} <: AbstractRootDistance
    domain::D
    assignment::A = SortAssignement(imag)
    transform::F1 = identity
    weight::F2 = e->ones(length(e))
end

"""
    SinkhornRootDistance{D, F1, F2} <: AbstractRootDistance

The Sinkhorn distance between roots. The weights are provided by `weight`, which defaults to [`residueweight`](@ref).

#Arguments:
- `domain::D`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `transform::F1 = identity`: Probably not needed.
- `weight::F2 = `[`residueweight`](@ref): A function used to calculate weights for the induvidual root distances.
- `β::Float64 = 0.01`: Amount of entropy regularization
- `iters::Int = 10000`: Number of iterations of the Sinkhorn algorithm.
"""
SinkhornRootDistance
@kwdef struct SinkhornRootDistance{D,F1,F2} <: AbstractRootDistance
    domain::D
    transform::F1 = identity
    weight::F2 = residueweight
    β::Float64 = 0.01
    iters::Int = 10000
end

# TODO: Merge the two above
@kwdef struct ManhattanRootDistance{D,A,F} <: AbstractRootDistance
    domain::D
    assignment::A = SortAssignement(imag)
    transform::F = identity
end

"""
    HungarianRootDistance{D, ID <: Distances.PreMetric, F} <: AbstractRootDistance

DOCSTRING

#Arguments:
- `domain::D`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `distance::ID = SqEuclidean()`: Inner distance
- `transform::F = identity`: If provided, this Function transforms all roots before the distance is calculated
"""
HungarianRootDistance
@kwdef struct HungarianRootDistance{D,ID <: Distances.PreMetric,F} <: AbstractRootDistance
    domain::D
    distance::ID = SqEuclidean()
    transform::F = identity
end

"""
    KernelWassersteinRootDistance{D, F, DI} <: AbstractRootDistance

DOCSTRING

#Arguments:
- `domain::D`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `λ::Float64 = 1.0`: reg param
- `transform::F = identity`: If provided, this Function transforms all roots before the distance is calculated
- `distance::DI = SqEuclidean()`: Inner distance
"""
KernelWassersteinRootDistance
@kwdef struct KernelWassersteinRootDistance{D,F,DI} <: AbstractRootDistance
    domain::D
    λ::Float64   = 1.
    transform::F = identity
    distance::DI = SqEuclidean()
end

"""
    OptimalTransportModelDistance{WT, DT} <: AbstractModelDistance

DOCSTRING

#Arguments:
- `w::WT = LinRange(0.01, 0.5, 300)`: Frequency set
- `distmat::DT = distmat_euclidean(w, w)`: DESCRIPTION
"""
OptimalTransportModelDistance
@kwdef struct OptimalTransportModelDistance{WT,DT} <: AbstractModelDistance
    w::WT = LinRange(0.01, 0.5, 300)
    distmat::DT = distmat_euclidean(w,w)
end

"""
    WelchOptimalTransportDistance{DT, AT <: Tuple, KWT <: NamedTuple} <: AbstractSignalDistance

Calculates the Wasserstein distance between two signals by estimating a Welch periodogram of each.

#Arguments:
- `distmat::DT`: you may provide a matrix array for this
- `args::AT = ()`: Options to the Welch function
- `kwargs::KWT = NamedTuple()`: Options to the Welch function
"""
WelchOptimalTransportDistance
@kwdef struct WelchOptimalTransportDistance{DT,AT <: Tuple, KWT <: NamedTuple} <: AbstractSignalDistance
    distmat::DT
    β::Float64 = 0.01
    iters::Int = 10000
    args::AT = ()
    kwargs::KWT = NamedTuple()
end

"""
    OptimalTransportHistogramDistance{DT} <: AbstractDistance

What it sounds like

#Arguments:
- `p::Int = 1`: order
"""
OptimalTransportHistogramDistance
@kwdef struct OptimalTransportHistogramDistance{DT} <: AbstractDistance
    p::Int = 1
end

"""
    ClosedFormSpectralDistance{DT, MT} <: AbstractModelDistance

calculates the Wasserstein distance using the closed-form solution based on integrals and inverse cumulative functions.

#Arguments:
- `domain::DT`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `p::Int = 1`: order
- `magnitude::MT = Identity()`:
- `interval = (-(float(π)), float(π))`: Integration interval
"""
ClosedFormSpectralDistance
@kwdef struct ClosedFormSpectralDistance{DT,MT} <: AbstractModelDistance
    domain::DT
    p::Int = 1
    magnitude::MT = Identity()
    interval = (-float(π,),float(π))
end
magnitude(d::ClosedFormSpectralDistance) = d.magnitude

"""
    CramerSpectralDistance{DT} <: AbstractModelDistance

Similar to `ClosedFormSpectralDistance` but does not use inverse functions.

#Arguments:
- `domain::DT`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `p::Int = 2`: order
- `interval = (-(float(π)), float(π))`: Integration interval
"""
CramerSpectralDistance
@kwdef struct CramerSpectralDistance{DT} <: AbstractModelDistance
    domain::DT
    p::Int = 2
    interval = (-float(π,),float(π))
end

"""
    BuresDistance <: AbstractDistance

Distance between pos.def. matrices
"""
struct BuresDistance <: AbstractDistance
end

"""
    EnergyDistance <: AbstractSignalDistance

`std(x1) - std(x2)`
"""
struct EnergyDistance <: AbstractSignalDistance
end


"""
    domain(d::AbstractDistance)

Return the domain of the distance
"""
domain(d) = d.domain
domain(d::ModelDistance) = domain(d.distance)


"""
    domain_transform(d::AbstractDistance, e)

Change domain of roots
"""
domain_transform(d::AbstractDistance, e) = domain_transform(domain(d), e)


transform(d::AbstractRootDistance) = d.transform
transform(d::AbstractRootDistance, x) = d.transform(x)


distmat_euclidean(e1::AbstractArray,e2::AbstractArray) = abs2.(e1 .- transpose(e2))
distmat_euclidean(e1,e2) = (n = length(e1[1]); [(e1[1][i]-e2[1][j])^2 + (e1[2][i]-e2[2][j])^2 for i = 1:n, j=1:n])

distmat(dist,e1,e2) = (n = length(e1[1]); [dist(e1[i],e2[j]) for i = 1:n, j=1:n])

distmat_logmag(e1::AbstractArray,e2::AbstractArray) = distmat_euclidean(logmag.(e1), logmag.(e2))

function preprocess_roots(d::AbstractRootDistance, e::AbstractRoots)
    e2 = domain_transform(d,e)
    T = similar_type(e2)
    T(d.transform.(e2))
end

function preprocess_roots(d::AbstractRootDistance, m::AbstractModel)
    e = roots(m)
    preprocess_roots(d, e)
end
function evaluate(d::AbstractRootDistance,w1::AbstractModel,w2::AbstractModel)
    evaluate(d, preprocess_roots(d,w1), preprocess_roots(d,w2))
end
function evaluate(d::AbstractRootDistance,w1::ARMA,w2::ARMA)
    d1 = evaluate(d, preprocess_roots(d,pole(w1)), preprocess_roots(d,pole(w2)))
    d2 = evaluate(d, preprocess_roots(d,tzero(w1)), preprocess_roots(d,tzero(w2)))
    d1 + d2
end


function evaluate(d::HungarianRootDistance, e1::AbstractRoots, e2::AbstractRoots)
    e1,e2 = toreim(e1), toreim(e2)
    n     = length(e1[1])
    dist  = d.distance
    dm    = [dist(e1[1][i],e2[1][j]) + dist(e1[2][i],e2[2][j]) for i = 1:n, j=1:n]
    c     = hungarian(dm)[2]
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
    I1,I2 = d.assignment(e1, e2)
    w1,w2 = d.weight(e1), d.weight(e2)
    sum(eachindex(e1)) do i
        i1 = I1[i]
        i2 = I2[i]
        (w1[i1]*w2[i2])abs2((e1[i1]-e2[i2]))
    end
end

function evaluate(d::SinkhornRootDistance, e1::AbstractRoots,e2::AbstractRoots)
    D     = distmat_euclidean(e1,e2)
    w1    = d.weight(e1)
    w2    = d.weight(e2)
    C     = sinkhorn(D,SVector{length(w1)}(w1),SVector{length(w2)}(w2),β=d.β, iters=d.iters)[1]
    if any(isnan, C)
        @info "Nan in SinkhornRootDistance, increasing precision"
        C     = sinkhorn(big.(D),SVector{length(w1)}(big.(w1)),SVector{length(w2)}(big.(w2)),β=d.β, iters=d.iters)[1]
        any(isnan, C) && @error "Sinkhorn failed, consider increasing β"
        eltype(D).(C)
    end
    sum(C.*D)
end

manhattan(c) = abs(c.re) + abs(c.im)
function evaluate(d::ManhattanRootDistance, e1::AbstractRoots,e2::AbstractRoots)
    I1,I2 = d.assignment(e1, e2)
    sum(manhattan, e1[I1]-e2[I2])
end

function eigval_dist_wass_logmag_defective(d::KernelWassersteinRootDistance, e1::AbstractRoots,e2::AbstractRoots)
    λ     = d.λ
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
    λ     = d.λ
    # dm1   = exp.(.- λ .* distmat(d.distance, e1,e1))
    # dm2   = exp.(.- λ .* distmat(d.distance, e2,e2))
    # dm12  = exp.(.- λ .* distmat(d.distance, e1,e2))
    dm1   = exp.(.- λ .* distmat_euclidean(e1,e1))
    dm2   = exp.(.- λ .* distmat_euclidean(e2,e2))
    dm12  = exp.(.- λ .* distmat_euclidean(e1,e2))
    mean(dm1) - 2mean(dm12) + mean(dm2)
end

evaluate(d::AbstractCoefficientDistance,w1::AbstractModel,w2::AbstractModel) = evaluate(d, coefficients(domain(d),w1), coefficients(domain(d),w2))

function evaluate(d::CoefficientDistance,w1::AbstractArray,w2::AbstractArray)
    evaluate(d.distance,w1,w2)
end

function evaluate(d::ModelDistance,X,Xh)
    w = fitmodel(d.fitmethod, X)
    wh = fitmodel(d.fitmethod, Xh)
    evaluate(d.distance, w, wh)
end

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

"""
    precompute(d::AbstractDistance, As, threads=true)

Perform computations that only need to be donce once when several pairwise distances are to be computed

#Arguments:
- `As`: A vector of models
- `threads`: Us multithreading? (true)
"""
function precompute(d::OptimalTransportModelDistance, As::AbstractArray{<:AbstractModel}, threads=true)
    mapfun = threads ? tmap : map
    mapfun(As) do A1
        w = d.w
        noise_model = tf(m1)
        b1 = bode(noise_model, w.*2π)[1] |> vec
        b1 .= abs2.(b1)
        b1 ./= sum(b1)
        b1
    end
end

function evaluate(d::OptimalTransportModelDistance, m1::AbstractModel, m2::AbstractModel)
    w = d.w
    noise_model = tf(m1)
    b1,_,_ = bode(noise_model, w.*2π) .|> vec
    b1 .= abs2.(b1)
    # b1 .-= (minimum(b1) - 1e-9) # reg to ensure no value is exactly 0
    b1 ./= sum(b1)
    noise_model = tf(m2)
    b2,_,_ = bode(noise_model, w.*2π) .|> vec
    b2 .= abs2.(b2)
    # b2 .-= (minimum(b2) - 1e-9)
    b2 ./= sum(b2)
    evaluate(d, b1, b2)
end

function evaluate(d::OptimalTransportModelDistance, b1, b2)
    plan = trivial_transport(b1, b2)
    cost = sum(plan .* d.distmat)
end

function evaluate(d::WelchOptimalTransportDistance, w1::DSP.Periodograms.TFR, w2::DSP.Periodograms.TFR)
    D = d.distmat == nothing ? distmat_euclidean(w1.freq, w2.freq) : d.distmat
    if issorted(w1.freq) && issorted(w2.freq)
        C = trivial_transport(s1(w1.power),s1(w2.power), 1e-3)
    else
        C = sinkhorn(D,s1(w1.power),s1(w2.power),β=d.β, iters=d.iters)[1]
    end
    cost = sum(C .* D)
end

function evaluate(d::WelchOptimalTransportDistance, x1, x2)
    evaluate(d, welch_pgram(x1, d.args...; d.kwargs...), welch_pgram(x2, d.args...; d.kwargs...))
end

centers(x) = 0.5*(x[1:end-1] + x[2:end])
function evaluate(d::OptimalTransportHistogramDistance, x1, x2)
    p  = d.p
    h1 = fit(Histogram,x1)
    h2 = fit(Histogram,x2)
    distmat = [abs(e1-e2)^p for e1 in centers(h1.edges[1]), e2 in centers(h2.edges[1])]
    plan = IPOT(distmat, s1(h1.weights), s1(h2.weights))[1]
    # plan = sinkhorn_plan_log(distmat, b1, b2; ϵ=1/10, rounds=300)
    cost = sum(plan .* distmat)
end

"""
    trivial_transport(x::AbstractVector{T}, y::AbstractVector{T}, tol=sqrt(eps(T))) where T

Calculate the optimal-transport plan between two vectors that are assumed to have the same support, with sorted support points.
"""
function trivial_transport(x::AbstractVector{T},y::AbstractVector{T},tol=sqrt(eps(T))) where T
    x  = copy(x)
    yf = zero(T)
    n  = length(x)
    @assert length(y) == n
    g = zeros(n,n)
    i = j = 1
    @inbounds while j <= n && i <= n
        needed = y[j] - yf
        available = x[i]
        if available >= needed
            g[i,j] += needed
            x[i] -= needed
            yf = zero(T)
            j += 1
        else
            g[i,j] += available
            yf += available
            i += 1
        end
    end
    if j < n || i < n
        takenfromy = yf#min(needed,available)
        sumleft = (sum(x[i:end]),sum(y[j:end])-takenfromy)
        if sumleft[1] > tol || sumleft[2] > tol
            error("Not all indices were covered (n,i,j) = $((n,i,j)), sum left (x,y) = $(sumleft)")
        end
    end
    g
end

function Base.inv(f::AbstractVector)
    n = length(f)
    r = LinRange(0, f[end], n)
    map(r) do r
        findfirst(>=(r), f)
    end
end

Base.inv(f::Function, interval) = x->finv(f, x, interval)
# Base.inv(f::Function,fp) = x->finv(f, fp, x)

function finv(f, z, interval)
    w = Roots.fzero(w->f(w)-z, interval...)
end

# function finv(f, fp, z)
#     w = Roots.fzero(w->f(w)-z, fp, π)
# end

"""
    ∫(f, a, b) = begin

Integrate `f` between `a` and `b`
"""
∫(f,a,b) = quadgk(f,a,b; atol=1e-12, rtol=1e-7)[1]::Float64

"""
    c∫(f, a, b; kwargs...)

Cumulative integration of `f` between `a` and `b`

#Arguments:
- `kwargs`: are sent to the DiffEq solver
"""
function c∫(f,a,b;kwargs...)
    fi    = (u,p,t) -> f(t)
    tspan = (a,b)
    prob  = ODEProblem(fi,0.,tspan)
    retry = false
    try
        sol   = solve(prob,Tsit5();reltol=1e-12,abstol=1e-45,kwargs...)
    catch
        retry = true
    end
    retry = retry || length(sol) <= 1 || abs(sol[end]-1) > 0.1
    if retry # In this case the solver failed due to numerical issues
        @warn "Spectral integration failed, increasing precision"
        prob  = ODEProblem(fi,big(0.),big.(tspan))
        sol   = solve(prob,Vern9();reltol=1e-16,abstol=1e-45,kwargs...)
    end
    sol
end

@inline ControlSystems.evalfr(::Discrete, m::Identity, w, a::AbstractArray, scale::Number=1) =
        (n=length(a);abs2(scale/sum(j->a[j]*exp(im*w*(n-j)), 1:n)))
@inline ControlSystems.evalfr(::Continuous, m::Identity, w, a::AbstractArray, scale::Number=1) =
        (n=length(a);abs2(scale/sum(j->a[j]*(im*w)^(n-j), 1:n)))
@inline ControlSystems.evalfr(::Discrete, m::Log, w, a::AbstractArray, scale::Number=1) =
        (n=length(a);-log(abs2(sum(j->a[j]*exp(im*w*(n-j)), 1:n))) + scale/(2π))
@inline ControlSystems.evalfr(::Continuous, m::Log, w, a::AbstractArray, scale::Number=1) =
        (n=length(a);-log(abs2(sum(j->a[j]*(im*w)^(n-j), 1:n))) + scale/(2π))

@inline ControlSystems.evalfr(d, m, w, a::AR, scale=1) = evalfr(d,m,w,denvec(d, a), scale)
@inline ControlSystems.evalfr(r::AbstractRoots, m, w, scale=1) = evalfr(domain(r),m,w,roots2poly(r), scale)


function invfunctionbarrier(sol1::T1,sol2::T2,p,interval) where {T1,T2}
    σ1    = sol1(interval[2]) # The total energy in the spectrum
    σ2    = sol2(interval[2]) # The total energy in the spectrum
    abs(σ1-1) > 0.05 && @warn "Cumulative spectral energy not equal to 1, σ1 = $(σ1)"
    abs(σ2-1) > 0.05 && @warn "Cumulative spectral energy not equal to 1, σ2 = $(σ2)"
    F1(w) = sol1(w)/σ1
    F2(w) = sol2(w)/σ2
    ∫(z->abs(inv(F1, interval)(z) - inv(F2, interval)(z))^p, 0, 1)
end

function closed_form_wass(d::ClosedFormSpectralDistance,sol1,sol2)
    p,interval = d.p, d.interval
    invfunctionbarrier(sol1,sol2,p,interval)
end

function functionbarrier(sol1::T1,sol2::T2,p,interval) where {T1,T2}
    σ1    = sol1(interval[2]) # The total energy in the spectrum
    σ2    = sol2(interval[2]) # The total energy in the spectrum
    abs(σ1-1) > 0.05 && @warn "Cumulative spectral energy not equal to 1, σ1 = $(σ1)"
    abs(σ2-1) > 0.05 && @warn "Cumulative spectral energy not equal to 1, σ2 = $(σ2)"
    F1(w) = sol1(w)/σ1
    F2(w) = sol2(w)/σ2
    ∫(interval...) do w
        abs(F1(w)-F2(w))^p
    end
end

function evaluate(d::Union{ClosedFormSpectralDistance, CramerSpectralDistance}, A1::AbstractModel, A2::AbstractModel)
    e1   = 1/sqrt(spectralenergy(domain(d), A1))
    f1   = w -> evalfr(domain(d), magnitude(d), w, A1, e1)
    e2   = 1/sqrt(spectralenergy(domain(d), A2))
    f2   = w -> evalfr(domain(d), magnitude(d), w, A2, e2)
    sol1 = c∫(f1,d.interval...)
    sol2 = c∫(f2,d.interval...)
    d isa ClosedFormSpectralDistance && d.p > 1 && (return closed_form_wass(d,sol1,sol2))
    evaluate(d, sol1, sol2)
end

function evaluate(d::Union{ClosedFormSpectralDistance, CramerSpectralDistance}, sol1, sol2)
    d isa ClosedFormSpectralDistance && d.p > 1 && (return closed_form_wass(d,sol1,sol2))
    functionbarrier(sol1,sol2,d.p,d.interval)
end

function evaluate(d::Union{ClosedFormSpectralDistance, CramerSpectralDistance}, A1::AbstractModel, P::DSP.Periodograms.TFR)
    p    = d.p
    e    = 1/sqrt(spectralenergy(domain(d), A1))
    f    = w -> evalfr(domain(d), magnitude(d), w, A1, e)
    w    = P.freq .* (2π)
    sol  = c∫(f,d.interval...; saveat=w)
    cP   = cumsum(sqrt.(P.power))
    cP ./= cP[end]
    plan = trivial_transport(sol.u./sol.u[end], cP)
    c    = 0.
    for i in axes(plan,1), j in axes(plan,2)
        c += abs(w[i]-w[j])^p * plan[i,j]
    end
    @assert c >= 0 "Distance turned out to be negative :/ c=$c"
    c
end

function precompute(d::Union{ClosedFormSpectralDistance, CramerSpectralDistance}, As::AbstractArray{<:AbstractModel}, threads=true)
    mapfun = threads ? tmap : map
    mapfun(As) do A1
        e1 = 1/sqrt(spectralenergy(domain(d), A1))
        f1   = w -> evalfr(domain(d), magnitude(d), w, A1, e1)
        sol1 = c∫(f1,d.interval...)
    end
end

# function closed_form_log_wass(a1,a2,p=1)
#     n     = length(a1)
    # f1    = w -> -log(abs(evalfr(Discrete(), w, a1))) + 1/(2π)
#     f2    = w -> -log(abs(evalfr(Discrete(), w, a2))) + 1/(2π)
#     endpoint = min(∫(f1,0,2π), ∫(f2,0,2π))
#     F1(w) = ∫(f1,0,w)
#     F2(w) = ∫(f2,0,w)
#     ∫(z->abs(inv(F1)(z) - inv(F2)(z))^p, 1e-9, endpoint-1e-9)
# end

"""
    stabilize(sys)

takes a statespace system and reflects all eigenvalues to be in the left half plane
"""
function stabilize(sys)
    A = sys.A
    e = eigen(A)
    e.values .= reflect(ContinuousRoots(e.values))
    sys.A .= real((e.vectors) * Diagonal(e.values) * inv(e.vectors))
    sys
end

function evaluate(d::BuresDistance, A1::AbstractModel, A2::AbstractModel)
    evaluate(d,tf(A1),tf(A2))
end

function evaluate(d::BuresDistance, A1::LTISystem, A2::LTISystem)
    sys1 = (ss((A1))) # stabilize
    sys2 = (ss((A2)))
    X1   = gram(sys1,:c)
    X2   = gram(sys2,:c)
    evaluate(d,(sys1.C*X1*sys1.C'),(sys2.C*X2*sys2.C'))
    # evaluate(d,inv(X1),inv(X2))
    # 2π*tr(sys.C*X*sys.C')
end

function evaluate(d::BuresDistance, A::AbstractMatrix, B::AbstractMatrix)
    sA = sqrt(A)
    tr(A+B - 2sqrt(sA*B*sA))
end
