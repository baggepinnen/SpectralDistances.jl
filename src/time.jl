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

distmat_euclidean(m1::AbstractModel,m2::AbstractModel,p=2) = distmat_euclidean!(zeros(length(m1),length(m2)), m1,m2,p)

function distmat_euclidean!(D, m1::TimeVaryingRoots,m2::TimeVaryingRoots,p=2)
    for i in 1:length(m1), j in 1:length(m2)
        _,t1 = mi2ij(m1,i)
        _,t2 = mi2ij(m2,j)
        D[i,j] = abs(m1[i]-m2[j])^p + abs(t1-t2)^p
    end
    D
end


function evaluate(d::OptimalTransportRootDistance, m1::TimeVaryingRoots,m2::TimeVaryingRoots; solver=sinkhorn_log!, kwargs...)
    D     = distmat_euclidean(m1,m2,d.p)
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
