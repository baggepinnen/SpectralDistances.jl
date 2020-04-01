struct ModelSpectrogram{T<:AbstractModel} <: AbstractModel
    models::Vector{T}
end

struct TimeVaryingRoots{T<:AbstractRoots} <: AbstractModel
    roots::Vector{T}
end

Base.length(m::TimeVaryingRoots) = sum(length, roots)
Base.size(m::TimeVaryingRoots) = (length(m.roots[1]), length(m.roots))


function Base.getindex(m::TimeVaryingRoots, i, j)
    m.roots[j][i]
end

function Base.getindex(m::TimeVaryingRoots, i)
    j =
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

distmat_euclidean(m1::AbstractModel,m2::AbstractModel,p=2) = distmat_euclidean!(zeros(length(m1),length(m2)), m1,m2,p)

function distmat_euclidean!(D, m1::TimeVaryingRoots,m2::TimeVaryingRoots,p=2)
    for i in 1:length(m1), j in 1:length(m2)
        D[i,j] = abs(m1[i]-m2[j])^p
    end
    D
end


y = randn(10000)
fm = TimeWindow(LS(na=2), 1000, 500)
m = fm(y)

distmat_euclidean(m,m)
