import Distances.evaluate

"The top level distance type"
abstract type AbstractDistance <: Distances.Metric end
"All subtypes of this type operates on rational transfer functions"
abstract type AbstractRationalDistance <: AbstractDistance end
"All subtypes of this type operates on signals"
abstract type AbstractSignalDistance <: AbstractDistance end
"All subtypes of this type operates on the roots of rational transfer functions"
abstract type AbstractRootDistance <: AbstractRationalDistance end
"All subtypes of this type operates on the coefficients of rational transfer functions"
abstract type AbstractCoefficientDistance <: AbstractRationalDistance end
abstract type AbstractWelchDistance <: AbstractSignalDistance end

Base.Broadcast.broadcastable(p::AbstractDistance) = Ref(p)

"A Union that represents a collection of distances"
const DistanceCollection = Union{Tuple, Vector{<:AbstractDistance}}

struct Identity end
struct Log end
Base.Broadcast.broadcastable(p::Identity) = Ref(p)
Base.Broadcast.broadcastable(p::Log) = Ref(p)
magnitude(d) = Identity()

Distances.result_type(::AbstractDistance, a, b) = promote_type(eltype(a), eltype(b))


evaluate(d::DistanceCollection,x,y;kwargs...) = sum(evaluate(d,x,y;kwargs...) for d in d)
Base.:(+)(d::AbstractDistance...) = d

(d::AbstractDistance)(x,y,args...;kwargs...) = evaluate(d, x, y,args...;kwargs...)
(d::DistanceCollection)(x,y,args...;kwargs...) = evaluate(d, x, y,args...;kwargs...)

"""
    SymmetricDistance{D} <: AbstractDistance

Evaluates `d(x,y) - 0.5(d(x,x) + d(y,y))`
"""
struct SymmetricDistance{D} <: AbstractDistance
    d::D
end

function evaluate(d::SymmetricDistance, x, y, args...; kwargs...)
    evaluate(d.d, x, y, args...; kwargs...) -
    0.5 * (
        evaluate(d.d, x, x, args...; kwargs...) +
        evaluate(d.d, y, y, args...; kwargs...)
    )
end

"""
    CoefficientDistance{D, ID} <: AbstractCoefficientDistance

Distance metric based on model coefficients

# Arguments:
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

# Arguments:
- `fitmethod::`[`FitMethod`](@ref): [`LS`](@ref), [`TLS`](@ref) or [`PLR`](@ref)
- `distance::D`: The inner distance between the models

# Example:
```julia
using SpectralDistances
innerdistance = OptimalTransportRootDistance(domain=Continuous(), β=0.005, p=2)
dist = ModelDistance(TLS(na=30), innerdistance)
```
"""
struct ModelDistance{D <: AbstractDistance, FM <: FitMethod} <: AbstractSignalDistance
    fitmethod::FM
    distance::D
end

"""
    EuclideanRootDistance{D, A, F1, F2} <: AbstractRootDistance

Simple euclidean distance between roots of transfer functions

# Arguments:
- `domain::D`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `assignment::A =` [`SortAssignement`](@ref)`(imag)`: Determines how roots are assigned. An alternative is `HungarianAssignement`
- `transform::F1 = identity`: DESCRIPTION
- `weight` : A function used to calculate weights for the induvidual root distances. A good option is [`residueweight`](@ref)
- `p::Int = 2` : Order of the distance
"""
EuclideanRootDistance
@kwdef struct EuclideanRootDistance{D,A,F1,F2} <: AbstractRootDistance
    domain::D = Continuous()
    assignment::A = SortAssignement(imag)
    transform::F1 = identity
    weight::F2 = unitweight
    p::Int = 2
end

"""
    OptimalTransportRootDistance{D, F1, F2, S} <: AbstractRootDistance

The Sinkhorn distance between roots. The weights are provided by `weight`, which defaults to [`residueweight`](@ref).

# Arguments:
- `domain::D = Continuous()`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `transform::F1 = identity`: Probably not needed.
- `weight::F2 = `[`simplex_residueweight`](@ref): A function used to calculate weights for the induvidual root distances.
- `β::Float64 = 0.01`: Amount of entropy regularization
- `p::Int = 2` : Order of the distance
- `divergence::S = nothing`: A divergence that penalizes creation and destruction of mass.
"""
OptimalTransportRootDistance
@kwdef struct OptimalTransportRootDistance{
    D,
    F1,
    F2,
    S<:Union{UnbalancedOptimalTransport.AbstractDivergence,Nothing},
} <: AbstractRootDistance
    domain::D = Continuous()
    transform::F1 = identity
    weight::F2 = simplex_residueweight
    β::Float64 = 0.01
    p::Int = 2
    divergence::S = nothing
end

@deprecate SinkhornRootDistance(args...;kwargs...) OptimalTransportRootDistance(args...;kwargs...)

"""
    HungarianRootDistance{D, ID <: Distances.PreMetric, F} <: AbstractRootDistance

Similar to [`EuclideanRootDistance`](@ref) but does the pole assignment using the Hungarian method.

# Arguments:
- `domain::D = Continuous()`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `distance::ID = SqEuclidean()`: Inner distance
- `transform::F = identity`: If provided, this Function transforms all roots before the distance is calculated
"""
HungarianRootDistance
@kwdef struct HungarianRootDistance{D,ID <: Distances.PreMetric,F} <: AbstractRootDistance
    domain::D = Continuous()
    distance::ID = SqEuclidean()
    transform::F = identity
end

"""
    KernelWassersteinRootDistance{D, F, DI} <: AbstractRootDistance

A kernel version of the root distance

# Arguments:
- `domain::D = Continuous()`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `λ::Float64 = 1.0`: Kernel precision, lower value means wider kernel.
- `transform::F = identity`: If provided, this Function transforms all roots before the distance is calculated
"""
KernelWassersteinRootDistance
@kwdef struct KernelWassersteinRootDistance{D,F,DI} <: AbstractRootDistance
    domain::D = Continuous()
    λ::Float64   = 1.
    transform::F = identity
    distance::DI = SqEuclidean()
end

"""
    DiscretizedRationalDistance{WT, DT} <: AbstractRationalDistance

This distance discretizes the spectrum before performing the calculations.

# Arguments:
- `w::WT = LinRange(0.01, 0.5, 300)`: Frequency set
- `distmat::DT = distmat_euclidean(w, w)`: DESCRIPTION
"""
DiscretizedRationalDistance
@kwdef struct DiscretizedRationalDistance{WT,DT} <: AbstractRationalDistance
    w::WT = LinRange(0.01, 0.5, 300)
    distmat::DT = distmat_euclidean(w,w)
end

"""
    WelchOptimalTransportDistance{DT, AT <: Tuple, KWT <: NamedTuple} <: AbstractWelchDistance

Calculates the Wasserstein distance between two signals by estimating a Welch periodogram of each.

# Arguments:
- `distmat::DT`: you may provide a matrix array for this
- `args::AT = ()`: Options to the Welch function
- `kwargs::KWT = NamedTuple()`: Options to the Welch function
- `p::Int = 2` : Order of the distance
"""
WelchOptimalTransportDistance
@kwdef struct WelchOptimalTransportDistance{DT,AT <: Tuple, KWT <: NamedTuple} <: AbstractWelchDistance
    distmat::DT = nothing
    args::AT = ()
    kwargs::KWT = NamedTuple()
    p::Int = 2
end

"""
    WelchLPDistance{AT <: Tuple, KWT <: NamedTuple, F} <: AbstractWelchDistance

Lᵖ distance between welch spectra, `mean(abs(x1-x2)^p)`.

#Arguments:
- `args::AT = ()`: These are sent to `welch_pgram`
- `kwargs::KWT = NamedTuple()`: These are sent to `welch_pgram`
- `p::Int = 2`: Order of the distance
- `normalized::Bool = true`: Normlize spectrum to sum to 1 (recommended)
- `transform::F = identity`: Optional function to apply to the spectrum, example `log1p` or `sqrt`. Must not produce negative values, so `log` is not a good idea. The function is applied like this: `transform.(x1)`.
"""
WelchLPDistance
@kwdef struct WelchLPDistance{AT <: Tuple, KWT <: NamedTuple, F} <: AbstractWelchDistance
    args::AT = ()
    kwargs::KWT = NamedTuple()
    p::Int = 2
    normalized::Bool = true
    transform::F = identity
end

"""
    OptimalTransportHistogramDistance{DT} <: AbstractDistance

Optimal transport between two histograms. If you pass two vectors to this distance two histograms will be computed automatically (`fit(Histogram,x)`). Pass two histograms to get better manual control.

# Arguments:
- `p::Int = 1`: order
"""
OptimalTransportHistogramDistance
@kwdef struct OptimalTransportHistogramDistance <: AbstractDistance
    p::Int = 1
end

"""
    DiscreteGridTransportDistance{DT} <: AbstractDistance

Optimal transport between two measures on a common discrete grid.
"""
DiscreteGridTransportDistance
struct DiscreteGridTransportDistance{D,TD,T,V} <: AbstractDistance
    inner::D
    distmat::Matrix{TD}
    C::Matrix{T}
    c1::V
    c2::V
end

"""
    DiscreteGridTransportDistance(Cityblock(), Float32, n, n)

The constructor takes an inner distance, the numeric type and the length of the two measures.
"""
function DiscreteGridTransportDistance(inner::Distances.PreMetric, T, n, m)
    distmat = inner.(one(T):n, (one(T):m)')
    C = zeros(T, n,m)
    c1 = zeros(T, n)
    c2 = zeros(T, m)
    DiscreteGridTransportDistance(inner,distmat,C,c1,c2)
end

"""
    CationalOptimalTransportDistance{DT, MT} <: AbstractRationalDistanCe

calculates the Wasserstein distance using the closed-form sold*c for (d,c) in zip(D,C)s and inverse cumulative functions.

C ArgumentC:
- `domain::DT = Continuous()`: [`Discrete`](@ref) or [`Continuous`](@red*c for (d,c) in zip(D,C)r
- `magnitude::MT = Identity()`:
- `interval = (-(float(π)), float(π))`: Integration interval
"""
RationalOptimalTransportDistance
@kwdef struct RationalOptimalTransportDistance{DT,MT} <: AbstractRationalDistance
    domain::DT = Continuous()
    p::Int = 1
    magnitude::MT = Identity()
    interval = (-float(π,),float(π))
end
magnitude(d::RationalOptimalTransportDistance) = d.magnitude

"""
    RationalCramerDistance{DT} <: AbstractRationalDistance

Similar to `RationalOptimalTransportDistance` but does not use inverse functions.

# Arguments:
- `domain::DT = Continuous()`: [`Discrete`](@ref) or [`Continuous`](@ref)
- `p::Int = 2`: order
- `interval = (-(float(π)), float(π))`: Integration interval
"""
RationalCramerDistance
@kwdef struct RationalCramerDistance{DT} <: AbstractRationalDistance
    domain::DT = Continuous()
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
This distance can be added to a loss function to ensure that the energy in the two signals is the same. Some of the optimal transport-based distances are invariant to the energy in the signal, requiring this extra cost if that invariance is not desired. Combining distances is done by putting two or more in a tuple.
    Usage: `combined_loss = (primary_distance, EnergyDistance())`
"""
struct EnergyDistance <: AbstractSignalDistance
end


"""
    domain(d::AbstractDistance)
    $(TYPEDSIGNATURES)

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


"""
    distmat_euclidean(e1::AbstractVector, e2::AbstractVector, p = 2)

The euclidean distance matrix between two vectors of complex numbers
"""
distmat_euclidean(e1::AbstractVector,e2::AbstractVector,p=2) = @fastmath abs.(e1 .- transpose(e2)).^p # Do not replace with call to mutating version, Zygote limitation

"""
    distmat_euclidean!(D, e1::AbstractVector, e2::AbstractVector, p = 2) = begin

In-place version
"""
distmat_euclidean!(D, e1::AbstractVector,e2::AbstractVector,p=2) = @fastmath D .= abs.(e1 .- transpose(e2)).^p

# distmat(dist,e1,e2) = (n = length(e1[1]); [dist(e1[i],e2[j]) for i = 1:n, j=1:n])

"""
    $(TYPEDSIGNATURES)

Compute the symmetric, pairwise distance matrix using the specified distance.

- `normalize`: set to true to normalize distances such that the diagonal is zero. This is useful for distances that are not true distances due to `d(x,y) ≠ 0` such as the [`OptimalTransportRootDistance`](@ref)

This function uses multiple threads if available and copies the distance object to each thread in case it stores an internal cache.
"""
function distmat(dist,e::AbstractVector; normalize=false, kwargs...)
    n = length(e)
    T = typeof(evaluate(dist,e[1],e[1];kwargs...))
    D = zeros(T,n,n)
    dists = [deepcopy(dist) for _ in 1:nthreads()]
    @showprogress 1 "distmat" for i = 1:n
        D[i,i] = evaluate(dist,e[i],e[i];kwargs...) # Note we do calc this distance since it's nonzero for regularized distances.
        @threads for j=i+1:n
            D[i,j] = evaluate(dists[threadid()], e[i], e[j]; kwargs...)
            (D[j,i] = D[i,j])
        end
    end
    normalize && symmetrize!(D)
    Symmetric(D)
end


function symmetrize!(D)
    d = diag(D)
    for i = 1:size(D,1)
        x = 0.5d[i]
        D[:,i] .-= x
        D[i,:] .-= x
    end
    D
end

function symmetrize!(D::SparseMatrixCSC)
    d = deepcopy(diag(D))
    N = length(d)
    for k = 1:N
        x = 0.5d[k]
        for i = 1:N
            D[i,k] == 0 && continue
            D[i,k] -= x
            D[k,i] -= x
        end
    end
    D
end

# distmat_logmag(e1::AbstractArray,e2::AbstractArray) = distmat_euclidean(logmag.(e1), logmag.(e2))

function preprocess_roots(d::AbstractRootDistance, e::AbstractRoots)
    e2 = domain_transform(d,e)
    T = similar_type(e2)
    T(d.transform.(e2))
end

function preprocess_roots(d::AbstractRootDistance, m::AbstractModel)
    e = roots(domain(d), m)
    preprocess_roots(d, e)
end

function evaluate(d::AbstractRootDistance,w1::AbstractModel,w2::AbstractModel; kwargs...)
    # if isderiving()
    #     # w1,w2 = change_precision(Float64, w1), change_precision(Float64, w2)
    #     return evaluate(d, preprocess_roots(d,w1), preprocess_roots(d,w2); solver=sinkhorn_log, tol=0.01) # workaround for https://github.com/FluxML/Zygote.jl/issues/584, when that has been closed, this branch can be removed
    # end
    r1 = preprocess_roots(d,w1)
    r2 = preprocess_roots(d,w2)
    evaluate(d, r1, r2; kwargs...)
end
function evaluate(d::AbstractRootDistance,w1::ARMA,w2::ARMA; kwargs...)
    d1 = evaluate(d, preprocess_roots(d,poles(domain(d),w1)), preprocess_roots(d,poles(domain(d),w2)))
    d2 = evaluate(d, preprocess_roots(d,tzeros(domain(d),w1)), preprocess_roots(d,tzeros(domain(d),w2)))
    d1 + d2
end

function evaluate(d::HungarianRootDistance, e1::AbstractRoots{CT}, e2::AbstractRoots{CT}; kwargs...)::real(CT) where CT
    e1,e2 = toreim(e1), toreim(e2)
    n     = length(e1[1])
    dist  = d.distance
    dm    = [dist(e1[1][i],e2[1][j]) + dist(e1[2][i],e2[2][j]) for i = 1:n, j=1:n]
    _,c   = hungarian(dm)
    c
end

# function kernelsum(e1,e2,λ)
#     s = 0.
#     λ = -λ
#     @inbounds for i in eachindex(e1)
#         s += exp(λ*abs(e1[i]-e2[i])^2)
#         for j in i+1:length(e2)
#             s += 2exp(λ*abs(e1[i]-e2[j])^2)
#         end
#     end
#     s / length(e1)^2
# end
#
# function logkernelsum(e1,e2,λ)
#     s = 0.
#     λ = -λ
#     le2 = logreim.(e2)
#     @inbounds for i in eachindex(e1)
#         le1 = logreim(e1[i])
#         s += exp(λ*abs2(le1-le2[i]))
#         for j in i+1:length(e2)
#             s += 2exp(λ*abs2(le1-le2[j]))
#         end
#     end
#     s / length(e1)^2
# end

function evaluate(d::EuclideanRootDistance, e1::AbstractRoots,e2::AbstractRoots; kwargs...)
    length(e1) == 0 && return zero(real(eltype(e1)))
    w1,w2 = d.weight(e1), d.weight(e2)
    # sum(eachindex(e1)) do i
    #     i1 = I1[i]
    #     i2 = I2[i]
    #     (w1[i1]*w2[i2])^β*abs(w1[i1]*e1[i1]-w2[i2]*e2[i2])^p
    # end # below is workaround for zygote #314
    I1,I2 = d.assignment(e1, e2)
    evaluate(d,e1[I1],e2[I2],w1[I1],w2[I2])
end

function evaluate(d::EuclideanRootDistance, e1,e2, w1, w2; kwargs...)
    p = d.p
    β = (1-p)/2
    l = zero(real(eltype(e1)))
    @inbounds @fastmath for i in 1:length(e1)
        l += (w1[i]*w2[i])^β*abs(w1[i]*e1[i]-w2[i]*e2[i])^p
    end
    real(l) # Workaround for expanding to complex for Zygote support
end

function evaluate(
    d::OptimalTransportRootDistance,
    e1::AbstractRoots,
    e2::AbstractRoots;
    solver = sinkhorn_log!,
    kwargs...,
)
    D = distmat_euclidean(e1, e2, d.p)
    w1 = d.weight(e1)
    w2 = d.weight(e2)
    # C     = solver(D,SVector{length(w1)}(w1),SVector{length(w2)}(w2); β=d.β, kwargs...)[1]

    if d.divergence !== nothing
        return OT!(
            d.divergence,
            D,
            DiscreteMeasure(w1),
            DiscreteMeasure(w2),
            d.β;
            max_iters = get(kwargs, :iters, 100_000),
            tol = get(kwargs, :tol, 1e-8),
            warn = true
        )
    end

    C = solver(D, w1, w2; β = d.β, kwargs...)[1]
    if any(isnan, C) && !isderiving()
        println("Nan in OptimalTransportRootDistance, increasing precision")
        C = solver(big.(D), big.(w1), big.(w2); β = d.β, kwargs...)[1]
        any(isnan, C) &&
        error("No solution found by solver $(solver), check your input and consider increasing β ($(d.β)).")
        C = eltype(D).(C)
    end
    dot(C,D)
end

function evaluate(d::AbstractRootDistance, a1::AbstractVector{<: Real},a2::AbstractVector{<: Real}; kwargs...)
    evaluate(d, AutoRoots(domain(d), hproots(rev(a1))), AutoRoots(domain(d), hproots(rev(a2))); kwargs...)
end

# function eigval_dist_wass_logmag_defective(d::KernelWassersteinRootDistance, e1::AbstractRoots,e2::AbstractRoots)
#     λ     = d.λ
#     error("this does not yield symmetric results")
#     # e1 = [complex((logmag(magangle(e1)))...) for e1 in e1]
#     # e2 = [complex((logmag(magangle(e2)))...) for e2 in e2]
#     # e1    = logreim.(e1)
#     # e2    = logreim.(e2)
#     # e1    = sqrtreim.(e1)
#     # e2    = sqrtreim.(e2)
#     dm1   = logkernelsum(e1,e1,λ)
#     dm2   = logkernelsum(e2,e2,λ)
#     dm12  = logkernelsum(e1,e2,λ)
#     dm1 - 2dm12 + dm2
# end
function evaluate(d::KernelWassersteinRootDistance, e1::AbstractRoots,e2::AbstractRoots; kwargs...)
    λ     = d.λ
    # dm1   = exp.(.- λ .* distmat(d.distance, e1,e1))
    # dm2   = exp.(.- λ .* distmat(d.distance, e2,e2))
    # dm12  = exp.(.- λ .* distmat(d.distance, e1,e2))

    D = distmat_euclidean(e1,e1)
    @avx D .= exp.(.- λ .* D)
    c = mean(D)
    distmat_euclidean!(D,e2,e2)
    @avx D .= exp.(.- λ .* D)
    c += mean(D)
    distmat_euclidean!(D,e1,e2)
    @avx D .= exp.(.- λ .* D)
    c -= 2mean(D)
    c
end

evaluate(d::AbstractCoefficientDistance,w1::AbstractModel,w2::AbstractModel; kwargs...) = evaluate(d, coefficients(domain(d),w1), coefficients(domain(d),w2); kwargs...)

function evaluate(d::CoefficientDistance,w1::AbstractArray,w2::AbstractArray; kwargs...)
    evaluate(d.distance,w1,w2; kwargs...)
end

function evaluate(d::ModelDistance,X,Xh; kwargs...)
    w = fitmodel(d.fitmethod, X)
    wh = fitmodel(d.fitmethod, Xh)
    evaluate(d.distance, w, wh; kwargs...)
end

# function batch_loss(bs::Int, loss, X, Xh; kwargs...)
#     l = zero(eltype(Xh))
#     lx = length(X)
#     n_batches = length(X)÷bs
#     inds = 1:bs
#     # TODO: introduce overlap for smoother transitions  #src
#     for i = 1:n_batches
#         l += loss(X[inds],Xh[inds]; kwargs...)
#         inds = inds .+ bs
#     end
#     l *= bs
#     residual_inds = inds[1]:lx
#     lr = length(residual_inds)
#     lr > 0 && (l += loss(X[residual_inds],Xh[residual_inds]; kwargs...)*lr)
#     l /= length(X)
#     l / n_batches
# end

evaluate(d::EnergyDistance,X::AbstractArray,Xh::AbstractArray; kwargs...) = (std(X)-std(Xh))^2 + (mean(X)-mean(Xh))^2

"""
    precompute(d::AbstractDistance, As, threads=true)

Perform computations that only need to be donce once when several pairwise distances are to be computed

# Arguments:
- `As`: A vector of models
- `threads`: Us multithreading? (true)
"""
function precompute(d::DiscretizedRationalDistance, As::AbstractArray{<:AbstractModel}, threads=true)
    mapfun = threads ? tmap : map
    mapfun(As) do A1
        w = d.w
        tm = tf(m1)
        b1 = bode(tm, w.*2π)[1] |> vec
        b1 .= abs2.(b1)
        b1 ./= sum(b1)
        b1
    end
end

function evaluate(d::DiscretizedRationalDistance, m1::AbstractModel, m2::AbstractModel; kwargs...)
    w = d.w
    m1 = tf(m1)
    b1,_,_ = bode(m1, w.*2π) .|> vec
    b1 .= abs2.(b1)
    # b1 .-= (minimum(b1) - 1e-9) # reg to ensure no value is exactly 0
    b1 ./= sum(b1)
    m2 = tf(m2)
    b2,_,_ = bode(m2, w.*2π) .|> vec
    b2 .= abs2.(b2)
    # b2 .-= (minimum(b2) - 1e-9)
    b2 ./= sum(b2)
    evaluate(d, b1, b2; kwargs...)
end

function evaluate(d::DiscretizedRationalDistance, b1, b2; kwargs...)
    plan = discrete_grid_transportplan(b1, b2)
    cost = dot(plan, d.distmat)
end

function evaluate(
    d::WelchOptimalTransportDistance,
    w1::DSP.Periodograms.TFR,
    w2::DSP.Periodograms.TFR;
    kwargs...,
)
    if d.distmat === nothing
        return discrete_grid_transportcost(s1(w1.power), s1(w2.power), d.p, 1e-3; inplace=true)
    else
        D = d.distmat
        C = discrete_grid_transportplan(s1(w1.power), s1(w2.power), 1e-3; inplace=true)
        return dot(C, D)
    end
end

function evaluate(d::WelchLPDistance, w1::DSP.Periodograms.TFR, w2::DSP.Periodograms.TFR;  kwargs...)
    x1,x2 = d.normalized ? (s1(w1.power),s1(w2.power)) : (w1.power, w2.power)
    mean(abs.(d.transform.(x1)-d.transform.(x2)).^d.p)
end

function evaluate(d::AbstractWelchDistance, x1, x2; kwargs...)
    evaluate(d, welch_pgram(x1, d.args...; d.kwargs...), welch_pgram(x2, d.args...; d.kwargs...))
end

centers(x) = 0.5*(x[1:end-1] + x[2:end])
function evaluate(d::OptimalTransportHistogramDistance, x1, x2; solver=IPOT, kwargs...)
    p  = d.p
    h1 = fit(Histogram,x1)
    h2 = fit(Histogram,x2)
    distmat = [abs(e1-e2)^p for e1 in centers(h1.edges[1]), e2 in centers(h2.edges[1])]
    plan = solver(distmat, s1(h1.weights), s1(h2.weights); kwargs...)[1]
    # plan = sinkhorn_plan_log(distmat, b1, b2; ϵ=1/10, rounds=300)
    cost = dot(plan, distmat)
end

function evaluate(d::DiscreteGridTransportDistance, x1, x2; kwargs...)
    inner = d.inner
    D = d.distmat
    (length(x1),length(x2)) == size(D) || throw(ArgumentError("Sizes do not match, should be (length(x1),length(x2)) == size(D), but got $((length(x1),length(x2))), $(size(D))"))
    c1,c2 = d.c1, d.c2
    d.C .= 0
    @avx c1 .= x1 ./ sum(x1)
    @avx c2 .= x2 ./ sum(x2)
    if typeof(inner) ∈ (Cityblock, Euclidean)
        return discrete_grid_transportcost(c1, c2, 1, inplace=true)
    elseif inner isa SqEuclidean
        return discrete_grid_transportcost(c1, c2, 2, inplace=true)
    end
    C = discrete_grid_transportplan(c1, c2, g=d.C, inplace=true)
    dot(d.C, D)
end

"""
    discrete_grid_transportplan(x::AbstractVector{T}, y::AbstractVector{T}, tol=sqrt(eps(T)); inplace=false) where T

Calculate the optimal-transport plan between two vectors that are assumed to have the same support, with sorted support points.
- ´inplace` if true, `x` is overwritten.
- It's possible to supply keyword argument `g` to provide preallocated memory for the transport plan. Make sure it's all zeros if you do.
"""
function discrete_grid_transportplan(x::AbstractVector{T},y::AbstractVector{T},tol=sqrt(eps(T)); g = zeros(T,length(x), length(x)), inplace=false) where T
    n  = length(x)
    length(y) == n || throw(ArgumentError("Inputs must have the same length"))
    if !inplace
        x  = copy(x)
    end
    yf = zero(T)
    i = j = 1
    @inbounds while j <= n && i <= n
        needed = y[j] - yf
        available = x[i]
        if available >= needed
            g[i,j] = needed
            x[i] -= needed
            yf = zero(T)
            j += 1
        else
            g[i,j] = available
            yf += available
            i += 1
        end
    end
    if j < n || i < n
        takenfromy = yf#min(needed,available)
        @views sumleft = (sum(x[i:end]),sum(y[j:end])-takenfromy)
        if sumleft[1] > tol || sumleft[2] > tol
            error("Not all indices were covered (n,i,j) = $((n,i,j)), sum left (x,y) = $(sumleft)")
        end
    end
    g
end

"""
    discrete_grid_transportcost(x::AbstractVector{T}, y::AbstractVector{T}, p = 2, tol=sqrt(eps(T)); inplace=false) where T

Calculate the optimal-transport cost between two vectors that are assumed to have the same support, with sorted and equdistant support points.

The calculated cost corresponds to the following
```julia
D = [abs2((i-j)/(n-1)) for i in 1:n, j in 1:n] # note the 1/(n-1)
Γ = discrete_grid_transportplan(x, y)
dot(Γ, D)
```
- ´p´ is the power of the distance
- ´inplace` if true, `x` is overwritten.
"""
function discrete_grid_transportcost(x::AbstractVector{T},y::AbstractVector{T}, p = 2, tol=sqrt(eps(T)); inplace=false) where T
    n  = length(x)
    length(y) == n || throw(ArgumentError("Inputs must have the same length"))
    if !inplace
        x  = copy(x)
    end
    yf = zero(T)
    i = j = 1
    cost = zero(T)
    @inbounds while j <= n && i <= n
        needed = y[j] - yf
        available = x[i]
        if available >= needed
            cost += (abs(i-j)/(n-1))^p*needed # the -1 is because the largest difference is n-1
            x[i] -= needed
            yf = zero(T)
            j += 1
        else
            cost += (abs(i-j)/(n-1))^p*available # the -1 is because the largest difference is n-1
            yf += available
            i += 1
        end
    end
    if j < n || i < n
        takenfromy = yf#min(needed,available)
        @views sumleft = (sum(x[i:end]),sum(y[j:end])-takenfromy)
        if sumleft[1] > tol || sumleft[2] > tol
            error("Not all indices were covered (n,i,j) = $((n,i,j)), sum left (x,y) = $(sumleft)")
        end
    end
    cost
end

# function Base.inv(f::AbstractVector)
#     n = length(f)
#     r = LinRange(0, f[end], n)
#     map(r) do r
#         findfirst(>=(r), f)
#     end
# end

Base.inv(f::Function, interval) = x->finv(f, x, interval)
# Base.inv(f::Function,fp) = x->finv(f, fp, x)

function finv(f, z, interval)
    f(interval[1])-z > 0 && return interval[1]
    f(interval[2])-z < 0 && return interval[2]
    w = Roots.fzero(w->f(w)-z, interval...)
end

# function finv(f, fp, z)
#     w = Roots.fzero(w->f(w)-z, fp, π)
# end

"""
    ∫(f, a, b) = begin

Integrate `f` between `a` and `b`
"""
∫(f,a,b)::Float64 = quadgk(f,a,b; atol=1e-12, rtol=1e-7)[1]

"""
    c∫(f, a, b; kwargs...)

Cumulative integration of `f` between `a` and `b`

# Arguments:
- `kwargs`: are sent to the DiffEq solver
"""
function c∫(f,a,b;kwargs...)
    fi    = (u,p,t) -> f(t)
    tspan = (a,b)
    prob  = ODEProblem(fi,0.,tspan)
    retry = false
    local sol
    try
        sol   = solve(prob,AutoTsit5(Rosenbrock23());reltol=1e-12,abstol=1e-45,kwargs...)
    catch
        retry = true
    end
    retry = retry || length(sol) <= 1 || abs(sol[end]-1) > 0.1
    if retry # In this case the solver failed due to numerical issues
        @warn "Spectral integration failed, increasing precision"
        prob  = ODEProblem(fi,big(0.),big.(tspan))
        sol   = solve(prob,AutoVern9(Rodas5());reltol=1e-16,abstol=1e-45,kwargs...)
    end
    sol
end

@inline ControlSystemsBase.evalfr(::Discrete, m::Identity, w, a::AbstractArray, b::AbstractVector) =
        (n=length(a);m=length(b);abs2(sum(j->b[j]*cis(w*(m-j)), 1:m)/sum(j->a[j]*cis(w*(n-j)), 1:n)))
@inline ControlSystemsBase.evalfr(::Continuous, m::Identity, w, a::AbstractArray, b::AbstractVector) =
        (n=length(a);m=length(b);abs2(sum(j->b[j]*(im*w)^(m-j), 1:m)/sum(j->a[j]*(im*w)^(n-j), 1:n)))

@inline ControlSystemsBase.evalfr(::Discrete, m::Identity, w, a::AbstractArray, scale::Number=1) =
        (n=length(a);abs2(scale/sum(j->a[j]*cis(w*(n-j)), 1:n)))
@inline ControlSystemsBase.evalfr(::Continuous, m::Identity, w, a::AbstractArray, scale::Number=1) =
        (n=length(a);abs2(scale/sum(j->a[j]*(im*w)^(n-j), 1:n)))
@inline ControlSystemsBase.evalfr(::Discrete, m::Log, w, a::AbstractArray, scale::Number=1) =
        (n=length(a);-log(abs2(sum(j->a[j]*cis(w*(n-j)), 1:n))) + scale/(2π))
@inline ControlSystemsBase.evalfr(::Continuous, m::Log, w, a::AbstractArray, scale::Number=1) =
        (n=length(a);-log(abs2(sum(j->a[j]*(im*w)^(n-j), 1:n))) + scale/(2π))

@inline ControlSystemsBase.evalfr(d, m, w, a::ARMA, scale=1) = evalfr(d,m,w,denvec(d, a), scale*denvec(d,a))
@inline ControlSystemsBase.evalfr(d, m, w, a::AR, scale=1) = evalfr(d,m,w,denvec(d, a), scale)
@inline ControlSystemsBase.evalfr(r::AbstractRoots, m, w, scale=1) = evalfr(domain(r),m,w,roots2poly(r), scale)


function invfunctionbarrier(sol1::T1,sol2::T2,p,interval) where {T1,T2}
    σ1    = sol1(interval[2]) # The total energy in the spectrum
    σ2    = sol2(interval[2]) # The total energy in the spectrum
    abs(σ1-1) > 0.05 && @warn "Cumulative spectral energy not equal to 1, σ1 = $(σ1)"
    abs(σ2-1) > 0.05 && @warn "Cumulative spectral energy not equal to 1, σ2 = $(σ2)"
    F1(w) = sol1(w)/σ1
    F2(w) = sol2(w)/σ2
    ∫(z->abs(inv(F1, interval)(z) - inv(F2, interval)(z))^p, 0, 1)
end

function closed_form_wass(d::RationalOptimalTransportDistance,sol1,sol2)
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

function evaluate(d::Union{RationalOptimalTransportDistance, RationalCramerDistance}, A1::AbstractModel, A2::AbstractModel)
    sc   = d.interval[1] == 0 ? sqrt(2) : 1
    e1   = sc*sqrt(A1.b/spectralenergy(domain(d), A1))
    f1   = w -> evalfr(domain(d), magnitude(d), w, A1, e1)
    e2   = sc*sqrt(A2.b/spectralenergy(domain(d), A2))
    f2   = w -> evalfr(domain(d), magnitude(d), w, A2, e2)
    sol1 = c∫(f1,d.interval...)
    sol2 = c∫(f2,d.interval...)
    d isa RationalOptimalTransportDistance && d.p > 1 && (return closed_form_wass(d,sol1,sol2))
    evaluate(d, sol1, sol2)
end

function evaluate(d::Union{RationalOptimalTransportDistance, RationalCramerDistance}, sol1, sol2)
    d isa RationalOptimalTransportDistance && d.p > 1 && (return closed_form_wass(d,sol1,sol2))
    functionbarrier(sol1,sol2,d.p,d.interval)
end

function evaluate(d::Union{RationalOptimalTransportDistance, RationalCramerDistance}, A1::AbstractModel, P::DSP.Periodograms.TFR)
    @assert d.interval[1] == 0 "Integration interval must start at 0 to compare against a periodogram"
    p    = d.p
    e    = sqrt(2)sqrt(A1.b/spectralenergy(domain(d), A1))
    f    = w -> evalfr(domain(d), magnitude(d), w, A1, e)
    w    = P.freq .* (2π)
    plan = discrete_grid_transportplan(s1(f.(w)), s1(sqrt.(P.power)))
    c    = 0.
    for i in axes(plan,1), j in axes(plan,2)
        c += abs(w[i]-w[j])^p * plan[i,j]
    end
    @assert c >= 0 "Distance turned out to be negative :/ c=$c"
    c
end

function precompute(d::Union{RationalOptimalTransportDistance, RationalCramerDistance}, As::AbstractArray{<:AbstractModel}, threads=true)
    mapfun = threads ? tmap : map
    sc   = d.interval[1] == 0 ? sqrt(2) : 1
    mapfun(As) do A1
        e1 = sc/sqrt(spectralenergy(domain(d), A1))
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
