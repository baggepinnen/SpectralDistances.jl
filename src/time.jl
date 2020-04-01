"""
This is a time-aware distance. It contains an inner distance (currently only [`OptimalTransportRootDistance`](@ref) supported), and some parameters that are specific to the time dimension, example:
```
dist = TimeDistance(inner=OptimalTransportRootDistance(domain=Continuous(), p=2, weight=s1∘residueweight), tp=2, c=0.1)
```
`tp` is the same as `p` but for the time dimension, and `c` trades off the distance along the time axis with the distance along the frequency axis. A smaller `c` makes it cheaper to transport mass across time. The frequency axis spans `[-π,π]` and the time axis is the non-negative integers, which should give you an idea for how to make this trade-off.
"""
@kwdef struct TimeDistance{D<:AbstractDistance,T} <: AbstractDistance
    inner::D
    tp::Int = 2
    c::T = 0.1
end

struct TimeVaryingRoots{T<:AbstractRoots} <: AbstractModel
    roots::Vector{T}
end

@inline Base.length(m::TimeVaryingRoots) = sum(length, m.roots)
@inline Base.size(m::TimeVaryingRoots) = (length(m.roots[1]), length(m.roots))
@inline Base.eachindex(m::TimeVaryingRoots) = 1:length(m)

@inline function Base.getindex(m::TimeVaryingRoots, i, j)
    m.roots[j][i]
end

@inline function mi2ij(m,i)
    r,c = size(m)
    j = (i-1) ÷ r + 1
    i = (i-1) % r + 1
    i,j
end


@inline function Base.getindex(m::TimeVaryingRoots, i)
    i,j = mi2ij(m,i)
    m[i,j]
end


PolynomialRoots.roots(::Continuous, m::TimeVaryingRoots) = m.roots


"""
We define a custom fit method for fitting time varying spectra, [`TimeWindow`](@ref). It takes as arguments an inner fitmethod, the number of points that form a time window, and the number of points that overlap between two consecutive time windows:
# Example
```
fitmethod = TimeWindow(TLS(na=2), 1000, 500)
y = sin.(0:0.1:100);
model = fitmethod(y)
```
"""
@kwdef struct TimeWindow{T<:FitMethod} <: FitMethod
    inner::T
    n::Int
    noverlap::Int = n÷2
end

function fitmodel(fm::TimeWindow, x)
    n,noverlap = fm.n, fm.noverlap
    models = map(arraysplit(x,n,noverlap)) do slice
        fitmodel(fm.inner, slice)
    end
    # ModelSpectrogram(models)
    TimeVaryingRoots(roots.(Continuous(), models))
end

preprocess_roots(d, e::Vector{<:AbstractRoots}) = e

distmat_euclidean(m1::AbstractModel,m2::AbstractModel,p=2, tp=2, c=0.1) = distmat_euclidean!(zeros(length(m1),length(m2)), m1, m2, p, tp, c)

function distmat_euclidean!(D, m1::TimeVaryingRoots,m2::TimeVaryingRoots,p=2, tp=2, c=0.1)
    for i in 1:length(m1), j in 1:length(m2)
        _,t1 = mi2ij(m1,i)
        _,t2 = mi2ij(m2,j)
        D[i,j] = abs(m1[i]-m2[j])^p + c*abs(t1-t2)^tp
    end
    D
end


function evaluate(od::TimeDistance, m1::TimeVaryingRoots,m2::TimeVaryingRoots; solver=sinkhorn_log!, kwargs...)
    d     = od.inner
    D     = distmat_euclidean(m1,m2,d.p, od.tp, od.c)
    w1    = s1(reduce(vcat,d.weight.(m1.roots)))
    w2    = s1(reduce(vcat,d.weight.(m2.roots)))
    C     = solver(D,SVector{length(w1)}(w1),SVector{length(w2)}(w2); β=d.β, kwargs...)[1]
    if any(isnan, C)
        println("Nan in OptimalTransportRootDistance, increasing precision")
        C     = solver(big.(D),SVector{length(w1)}(big.(w1)),SVector{length(w2)}(big.(w2)); β=d.β, kwargs...)[1]
        any(isnan, C) && error("No solution found by solver $(solver), check your input and consider increasing β ($(d.β)).")
        eltype(D).(C)
    end
    sum(C.*D)
end
