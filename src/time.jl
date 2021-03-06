"""
This is a time-aware distance. It contains an inner distance (currently only [`OptimalTransportRootDistance`](@ref) and [`KernelWassersteinRootDistance`](@ref)supported), and some parameters that are specific to the time dimension, example:
```
dist = TimeDistance(
    inner = OptimalTransportRootDistance(
        domain = Continuous(),
        p      = 2,
        weight = simplex_residueweight,
    ),
    tp = 2,
    c  = 0.1,
)
```
`tp` is the same as `p` but for the time dimension, and `c` trades off the distance along the time axis with the distance along the frequency axis. A smaller `c` makes it cheaper to transport mass across time. The frequency axis spans `[-π,π]` and the time axis is the non-negative integers, which should give you an idea for how to make this trade-off.
"""
TimeDistance
@kwdef struct TimeDistance{D,T} <: AbstractDistance
    inner::D
    tp::Int = 2
    c::T    = 0.1
end

"""
    TimeVaryingAR{T <: AbstractRoots} <: AbstractModel

This model represents a rational spectrogram, i.e., a rational spectrum that changes with time. See [`TimeWindow`](@ref) for the corresponding fit method and [`TimeDistance`](@ref) for a time-aware distance.

Internally, this model stores a vector of [`AR`](@ref).
"""
struct TimeVaryingAR{MT<:AR, VT <: AbstractVector{MT}} <: AbstractModel
    models::VT
end

@inline Base.length(m::TimeVaryingAR) = sum(length, m.models)-length(m.models) # the length of a model is 1 more than the number of poles due to the numerator
@inline Base.size(m::TimeVaryingAR) = (length(m.models[1])-1, length(m.models))
@inline Base.eachindex(m::TimeVaryingAR) = 1:length(m)

@inline function Base.getindex(m::TimeVaryingAR, i::Int, j::Int)
    m.models[j].pc[i]
end


@inline function divmod(a,b)
    c = a÷b
    d = a - c*b
    c,d
end

@inline function mi2ij(m,i)
    r,c = size(m)
    j, i = divmod((i-1), r)
    i+1, j+1
end

@inline function Base.getindex(m::TimeVaryingAR, i::Int)
    i,j = mi2ij(m,i)
    m[i,j]
end


PolynomialRoots.roots(::Continuous, m::TimeVaryingAR) = roots.(Continuous(), m.models)

function Base.isapprox(m1::TimeVaryingAR, m2::TimeVaryingAR, args...; kwargs...)
    all(isapprox(m1, m2, args...; kwargs...) for m1 in m1.models, m2 in m2.models)
end

change_precision(F, m::TimeVaryingAR) = TimeVaryingAR(change_precision.(F, m.models))

"""
We define a custom fit method for fitting time varying spectra, [`TimeWindow`](@ref). It takes as arguments an inner fitmethod, the number of points that form a time window, and the number of points that overlap between two consecutive time windows:
# Example
```
fitmethod = TimeWindow(TLS(na=2), 1000, 500)
y = sin.(0:0.1:100);
model = fitmethod(y)
```
"""
TimeWindow
@kwdef struct TimeWindow{T<:FitMethod} <: FitMethod
    inner::T
    n::Int
    noverlap::Int = n÷2
end

function fitmodel(fm::TimeWindow, x; showprogress = true)
    n,na,noverlap = fm.n, fm.inner.na, fm.noverlap
    if fm.inner.λ > 0
        AS = similar(x, n, na+1)
    else
        AS = similar(x, n-na, na+1)
    end
    if showprogress
        models = @showprogress 1 "Model estimation" map(arraysplit(x,n,noverlap)) do slice
            fitmodel!(AS, fm.inner, slice)
        end
    else
        models = map(arraysplit(x,n,noverlap)) do slice
            fitmodel!(AS, fm.inner, slice)
        end
    end
    TimeVaryingAR(models)
end

preprocess_roots(d, e::Vector{<:AbstractRoots}) = e

distmat_euclidean(m1::AbstractModel,m2::AbstractModel,p, tp, c) = distmat_euclidean!(zeros(length(m1),length(m2)), m1, m2, p, tp, c)

function distmat_euclidean!(D, m1::TimeVaryingAR, m2::TimeVaryingAR, p, tp, c)
    @assert size(D) == (length(m1), length(m2))
    _distmat_kernel!( D, m1, m2, c,
        p == 1 ? Base.FastMath.abs_fast :
        p == 2 ? Base.FastMath.abs2_fast : error("p must be 1 or 2"),
        tp == 1 ? Base.FastMath.abs_fast :
        tp == 2 ? Base.FastMath.abs2_fast : error("tp must be 1 or 2"),
    )
end

function _distmat_kernel!(D,m1,m2,c,f1::F1,f2::F2) where {F1,F2}
    @fastmath @inbounds for j in 1:length(m2)
        for i in 1:length(m1)
            _,t1 = mi2ij(m1,i)
            _,t2 = mi2ij(m2,j)
            D[i,j] = f1(m1[i]-m2[j]) + c*f2(t1-t2)
        end
    end
    D
end


function evaluate(od::TimeDistance{<:OptimalTransportRootDistance}, m1::TimeVaryingAR,m2::TimeVaryingAR; solver=sinkhorn_log!, kwargs...)
    d     = od.inner
    @assert d.domain isa Continuous "TimeDistance currently only works in continuous domain, open an issue with a motivation for why you require support for discrete domain and I might be able to add it."
    D     = distmat_euclidean(m1, m2, d.p, od.tp, od.c)
    w1    = s1(reduce(vcat,map(d.weight, m1.models)))
    w2    = s1(reduce(vcat,map(d.weight, m2.models)))
    C     = solver(D,w1,w2; β=d.β, kwargs...)[1]
    if any(isnan, C)
        @info("Nan in OptimalTransportRootDistance, increasing precision")
        C     = solver(big.(D),big.(w1),big.(w2); β=d.β, kwargs...)[1]
        any(isnan, C) && error("No solution found by solver $(solver), check your input and consider increasing β ($(d.β)).")
        eltype(D).(C)
    end
    dot(C, D)
end


@inline function Base.resize!(D::AbstractMatrix, s1::Int, s2::Int)
    l = s1 * s2
    Dv = resize!(vec(D), l)
    reshape(Dv, s1, s2)
end

function evaluate(
    od::TimeDistance{<:KernelWassersteinRootDistance},
    m1::TimeVaryingAR,
    m2::TimeVaryingAR;
    D = zeros(max(length(m1), length(m2)), max(length(m1), length(m2))),
    kwargs...,
)
    d = od.inner
    λ = d.λ
    @assert d.domain isa Continuous "TimeDistance currently only works in continuous domain, open an issue with a motivation for why you require support for discrete domain and I might be able to add it."

    D = resize!(D, length(m2), length(m2))
    distmat_euclidean!(D, m2, m2, 2, od.tp, od.c)
    @avx @. D = exp(-λ * D)
    c = mean(D)

    D = resize!(D, length(m1), length(m1))
    distmat_euclidean!(D, m1, m1, 2, od.tp, od.c)
    @avx @. D = exp(-λ * D)
    c += mean(D)

    D = resize!(D, length(m1), length(m2))
    distmat_euclidean!(D, m1, m2, 2, od.tp, od.c)
    @avx @. D = exp(-λ * D)
    c -= 2 * mean(D)

    c
end



"""
    distance_profile(od::TimeDistance, q::TimeVaryingAR, y::TimeVaryingAR; normalize_each_timestep = false, kwargs...)

Optimized method to compute the distance profile corresponding to sliding the short query `q` over the longer `y`.
"""
function SlidingDistancesBase.distance_profile(od::TimeDistance{<:OptimalTransportRootDistance}, q::TimeVaryingAR, y::TimeVaryingAR; normalize_each_timestep = false, kwargs...)
    d     = od.inner
    @assert d.domain isa Continuous "TimeDistance currently only works in continuous domain, open an issue with a motivation for why you require support for discrete domain and I might be able to add it."
    any(methods(d.weight).ms) do m
        m.nargs == 3
    end || throw(ArgumentError("distance_profile requires the weight function of the distance to support in-place update on the form `weightfun(weights, roots)`. See simplex_residueweight, residueweight, unitweight"))

    T  = eltype(q.models[1].a)
    N  = length(q)
    na = length(q.models[1].pc)
    nq = length(q.models)
    ny = length(y.models)
    w1 = s1(reduce(vcat,map(d.weight, q.models)))
    w2 = similar(w1)
    C  = Matrix{T}(undef, N, N)

    workspace = SinkhornLogWorkspace(T,N,N)

    @showprogress 1 "Distance profile" map(1:ny-nq) do i
        @views Y = TimeVaryingAR(y.models[i:i+nq-1])
        distmat_euclidean!(C, q, Y, d.p, od.tp, od.c)
        inds = (1:na)
        for i in eachindex(Y.models)
            d.weight(@view(w2[inds]), Y.models[i])
            if normalize_each_timestep
                w2[inds] ./= sum(@view(w2[inds]))
            end
            inds = inds .+ na
        end
        w2  ./= sum(w2)
        Γ = sinkhorn_log!(workspace,C,w1,w2; β=d.β, kwargs...)[1]
        dot(Γ,C)
    end
end


"""
    SlidingDistancesBase.distance_profile(d::ConvOptimalTransportDistance, q::AbstractMatrix, y::AbstractMatrix; stride=1, kwargs...)

Optimized method for [`ConvOptimalTransportDistance`](@ref). To get smooth distance profiles, a slightly higher β than for barycenters is recommended.  β around 0.01 should do fine.
- `stride`: allows you to reduce computations by setting `stride > 1`.

See also [`SlidingConvOptimalTransportDistance`](@ref)
"""


function SlidingDistancesBase.distance_profile(d::ConvOptimalTransportDistance, Q::AbstractMatrix, Y::AbstractMatrix; stride=1, kwargs...)

    T   = eltype(Q)
    m,n = size(Q)
    stride <= n || throw(ArgumentError("The stride can not be longer than the length of `q`"))
    N   = lastlength(Y)
    A   = copy(Q)
    B   = copy(getwindow(Y, n, 1))
    A ./= sum(A)
    workspace = SCWorkspace(A,B,d.β)
    U,V = workspace.U, workspace.V

    D = similar(Q, (N-n)÷stride+1)
    # Di = similar(Q, (N-n)÷stride+1)
    sB = sum(B) - sum(B[:,n-stride+1:n])
    iD = 0
    @views for i = 1:stride:N-n+1
        B .= getwindow(Y, n, i) # TODO: this is wasteful, only update one column and shift the rest
        sB += sum(B[:,n-stride+1:n])
        sB1 = sum(B[:,1:stride])
        @avx B ./= sB
        D[iD += 1] = sinkhorn_convolutional(workspace, A, B; β = d.β, initUV = false, kwargs...)[1]
        if d.invariant_axis != 0
            ia = d.invariant_axis
            sum_axis = ia == 1 ? 2 : 1
            v,u = vec(sum(A, dims=sum_axis)), vec(sum(B, dims=sum_axis))
            # v ./= sum(v); u ./= sum(u)
            invariant_cost = discrete_grid_transportcost(v, u; inplace=true)
            # Di[iD] = invariant_cost
            D[iD] -= invariant_cost
        end
        @avx U[:,1:end-stride] .= exp.(U[:,1+stride:end]) # Warm-start gives 3x imrovement in simple benchmark
        @avx V[:,1:end-stride] .= exp.(V[:,1+stride:end])
        @avx U[:,end-stride+1:end] .= 1#exp.(U[:,end]) # These make it slower
        @avx V[:,end-stride+1:end] .= 1#exp.(V[:,end])
        sB -= sB1
    end
    D#, Di
end

function SlidingDistancesBase.distance_profile(d::ConvOptimalTransportDistance, q::DSP.Periodograms.TFR, y::DSP.Periodograms.TFR; kwargs...)
    df  = d.dynamic_floor
    Q   = normalize_spectrogram(q, df)
    Y   = normalize_spectrogram(y, df)
    distance_profile(d, Q, Y; kwargs...)
end



# function distance_profile(d::AbstractDistance, q::TimeVaryingAR, y::TimeVaryingAR; normalize_each_timestep = false, kwargs...) where F
#     T  = eltype(q.models[1].a)
#     N  = length(q)
#     na = length(q.models[1].pc)
#     nq = length(q.models)
#     ny = length(y.models)
#     w1 = s1(reduce(vcat,map(d.weight, q.models)))
#     w2 = s1(reduce(vcat,map(d.weight, y.models)))
#     C  = Matrix{T}(undef, N, N)
#
#     any(methods(d.weight).ms) do m
#         m.nargs == 3
#     end || throw(ArgumentError("distance_profile requires the weight function of the distance to support in-place update on the form `weightfun(weights, roots)`. See simplex_residueweight, residueweight, unitweight"))
#
#     # workspace = SinkhornLogWorkspace(T,N,N)
#
#     @showprogress 1 "Distance profile"  map(1:ny-nq) do i
#         @views Y = TimeVaryingAR(y.models[i:i+nq-1])
#         inds = (1:na)
#         for i in eachindex(Y.models)
#             if normalize_each_timestep
#                 w2[inds] ./= sum(@view(w2[inds]))
#             end
#             inds = inds .+ na
#         end
#         w2  ./= sum(w2)
#         evaluate(d,e1,e2,w1,w2)
#     end
# end


struct PrimitiveModel{R <: AbstractVector,W <: AbstractVector} <: AbstractModel
    roots::R
    weights::W
end

function PrimitiveModel(d::AbstractDistance, m::AbstractModel)
    r = roots(domain(d), m)
    w = d.weight(r)
    PrimitiveModel(r.r,w)
end

function distmat(od::TimeDistance{<:OptimalTransportRootDistance}, models::Vector{<:TimeVaryingAR}; normalize=false, kwargs...)
    n = length(models)
    d = od.inner
    T = eltype(models[1].models[1].a)
    N = length(models[1])
    C = [Matrix{T}(undef, N, N) for _ = 1:nthreads()]

    weights = map(models) do q
        s1(reduce(vcat,map(d.weight, q.models)))
    end

    workspace = [SinkhornLogWorkspace(T, N, N) for _ = 1:nthreads()]
    dists = [deepcopy(d) for _ = 1:nthreads()]
    D = zeros(T, n, n)
    @showprogress 1 "distmat" for i = 1:n
        m1 = models[i]
        distmat_euclidean!(C[1], m1, m1, d.p, od.tp, od.c)
        Γ = sinkhorn_log!(workspace[1], C[1], weights[i], weights[i]; β = d.β, kwargs...)[1]
        D[i, i] = dot(Γ, C[1])
        Threads.@threads for j = i+1:n
            m2 = models[j]
            id = threadid()
            distmat_euclidean!(C[id], m1, m2, d.p, od.tp, od.c)
            Γ = sinkhorn_log!(
                workspace[id],
                C[id],
                weights[i],
                weights[j];
                β = d.β,
                kwargs...,
            )[1]
            D[i, j] = dot(Γ, C[id])
            (D[j, i] = D[i, j])
        end
    end
    normalize && symmetrize!(D)
    Symmetric(D)

end

function distmat(od::TimeDistance{<:KernelWassersteinRootDistance}, models::Vector{<:TimeVaryingAR}; normalize=false, kwargs...)
    n = length(models)
    d = od.inner
    λ = d.λ
    T = eltype(models[1].models[1].a)
    N = length(models[1])
    C = [Matrix{T}(undef, N, N) for _ = 1:nthreads()]
    dists = [deepcopy(d) for _ = 1:nthreads()]
    D = zeros(T, n, n)
    # error("need to take care of the size of C")
    @showprogress 1 "distmat" for i = 1:n
        m1 = models[i]
        Threads.@threads for j = i:n
            m2 = models[j]
            id = threadid()

            C[id] = resize!(C[id], length(m2), length(m2))
            distmat_euclidean!(C[id], m2, m2, 2, od.tp, od.c)
            @avx @. C[id] = exp(-λ * C[id])
            c = mean(C[id])

            C[id] = resize!(C[id], length(m1), length(m1))
            distmat_euclidean!(C[id], m1, m1, 2, od.tp, od.c)
            @avx @. C[id] = exp(-λ * C[id])
            c += mean(C[id])

            C[id] = resize!(C[id], length(m1), length(m2))
            distmat_euclidean!(C[id], m1, m2, 2, od.tp, od.c)
            @avx @. C[id] = exp(-λ * C[id])
            c -= 2 * mean(C[id])

            D[i, j] = c
            (D[j, i] = D[i, j])
        end
    end
    normalize && symmetrize!(D)
    Symmetric(D)

end
