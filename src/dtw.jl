
struct DTWHelper{D,T <: PrimitiveModel,S} <: AbstractDistance
    q::Vector{T}
    y::Vector{T}
    d::D
    storage::S
end

DTWHelper(d::AbstractDistance, q::TimeVaryingAR, y::TimeVaryingAR) = DTWHelper(
    map(m -> PrimitiveModel(d, m), q.models),
    map(m -> PrimitiveModel(d, m), y.models),
    d,
    nothing,
)

function DTWHelper(d::OptimalTransportRootDistance, q::TimeVaryingAR, y::TimeVaryingAR)
    Q = map(m -> PrimitiveModel(d, m), q.models)
    Y = map(m -> PrimitiveModel(d, m), y.models)
    n = length(Q[1].weights)
    T = eltype(Q[1].weights)
    C = zeros(T, n, n)
    DTWHelper(Q, Y, d, (SinkhornLogWorkspace(T, n, n), C))
end


function evaluate(d::DTWHelper{<:EuclideanRootDistance}, m1::PrimitiveModel, m2::PrimitiveModel; kwargs...)
    evaluate(d.d, m1.roots, m2.roots, m1.weights, m2.weights)
end

function evaluate(d::DTWHelper{<:OptimalTransportRootDistance}, m1::PrimitiveModel, m2::PrimitiveModel; kwargs...)
    d.d.divergence === nothing ||
        throw(ArgumentError("Divergences are not supported for efficient DTW"))
    w,C = d.storage
    distmat_euclidean!(C, m1.roots, m2.roots, d.d.p)
    Γ = sinkhorn_log!(w, C, m1.weights, m2.weights; β = d.d.β, kwargs...)[1]
    dot(Γ, C)
end

function DynamicAxisWarping.dtw_cost(d::DTWHelper, q, y, args...; kwargs...)
    DynamicAxisWarping.dtw_cost(q, y, d.d, args...; kwargs...)
end


"""
    DynamicAxisWarping.dtwnn(q::TimeVaryingAR, y::TimeVaryingAR, dist, rad::Int, args...; kwargs...)

Wrapper for `dtwnn`. To save allocations between multiple calls to `dtwnn`, you may manually create a helper object and call
    DynamicAxisWarping.dtwnn(d::DTWHelper, rad::Int, args...; kwargs...)
"""
function DynamicAxisWarping.dtwnn(q::TimeVaryingAR, y::TimeVaryingAR, dist, rad::Int, args...; kwargs...)
    h = DTWHelper(dist, q, y)
    DynamicAxisWarping.dtwnn(h, rad, args...; kwargs...)
end

function DynamicAxisWarping.dtwnn(d::DTWHelper, rad::Int, args...; kwargs...)
    w = DynamicAxisWarping.DTWWorkspace(d.q, d, rad)
    DynamicAxisWarping.dtwnn(w, d.y, args...; kwargs...)
end
