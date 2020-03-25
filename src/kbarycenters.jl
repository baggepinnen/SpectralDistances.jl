

# Base.@kwdef struct KBOptions{T}
#     β::T = 0.5
#     solver::F = IPOT
#     tol::Float64 = 1e-6
#     iters::Int = 1000
# end

struct KBResult{T} <: Clustering.ClusteringResult
    assignments::Vector{Int}
    barycenters::T
    cost::Float64
end


Clustering.nclusters(R::KBResult) = length(R.barycenters)
Clustering.counts(R::KBResult) = [count(R.assigments .== i) for i in 1:nclusters(R)]
Clustering.assignments(R::KBResult) = R.assignments


"""
    kbarycenters(X, p, k; seed = :rand, kiters = 10, verbose = false, output = :best, kwargs...)

Clustering using K-barycenters. If you want to cluster spectra, consider the method that accepts models instead.

# Arguments:
- `X`: Support of input measures, Vector{Matrix} where each matrix is n_dims×n_atoms
- `p`: Weights of input measures Vector{Vector} where each matrix is of length n_atoms and should sum to 1.
- `k`: number of clusters
- `seed`: :rand or :eq
- `kiters`: number of iterations
- `verbose`: print stuff?
- `output`: output lowest cost clustering or :latest?
- `kwargs`: are sent to the inner solvers.
"""
function kbarycenters(X,p,k; seed=:rand, kiters=10, verbose=false, output=:best, kwargs...)
    N = length(X)
    @assert length(p) == N
    @assert k < N "number of clusters must be smaller than number of points"
    if seed === :rand
        Q = perturb!.(copy.(X[randperm(N)[1:k]]), Ref(X))
    elseif seed === :eq
        Q = perturb!.(copy.(X[1:N÷k:end]), Ref(X))
    else
        throw(ArgumentError("Unknown symbol for seed: $seed"))
    end
    C = distmat_euclidean(Q[1],Q[1])
    q = ones(size(X[1],2)) |> s1
    λ = ones(N) |> s1
    ass, cost = assignments(C,X,Q,p,q;kwargs...)
    ass_old = copy(ass)
    verbose && @info "kbarycenters: Iter: 0 cost: $(cost)"
    bestcost = cost
    bestass = ass
    for iter = 1:kiters
        # @show iter
        Q = barycenters(C,X,p,q,λ,ass,k;kwargs...)
        ass, cost = assignments(C,X,Q,p,q;kwargs...)
        # @show ass
        if cost < bestcost
            bestcost = cost
            bestass = ass
        end
        if ass == ass_old
            verbose && @info "kbarycenters: Iter: $iter converged"
            break
        end
        verbose && @info "kbarycenters: Iter: $iter cost: $(cost) num changes: $(count(ass .!= ass_old))"
        ass_old = ass
        yield()
    end
    if ass != bestass && output === :best
        verbose && @info "kbarycenters: Best cost: $bestcost"
        cost = bestcost
        Q = barycenters(C,X,p,q,λ,bestass,k;kwargs...)
    end
    KBResult(ass, Q, cost)
end

function barycenters(C,X,p,q,λ,ass,k;kwargs...)
    N = length(X)
    unnull!(ass,k)
    Q = tmap(1:k) do i
        inds = ass .== i
        if count(inds) == 0
            @warn "null cluster"
            inds = randperm(N)[1:2]
        end
        barycenter(X[inds],p[inds], s1(λ[inds]); kwargs...)[1] # TODO: not considering the weights
    end
end

function kwcostfun(C,X,Q,p,q; solver=sinkhorn_log!, kwargs...)
    C = distmat_euclidean!(C, X,Q)
    sum(solver(C, p, q; kwargs...)[1] .* C)
end


function assignments(C,X,Q,p,q;kwargs...)
    k = length(Q)
    c = 0.0
    ass = map(1:length(X)) do i
        dists = map(1:k) do j
            kwcostfun(C,X[i],Q[j],p[i],q;kwargs...)
        end
        ind = argmin(dists)
        c += dists[ind]
        ind
    end
    ass, c
end


function unnull!(ass,k)
    N = length(ass)
    change = true
    while change
        change = false
        for i = 1:k
            if i ∉ ass
                change = true
                ass[randperm(N)[1:1]] .= i
            end
        end
    end
end

"""
    kbarycenters(d::SinkhornRootDistance, models::Vector{<:AbstractModel}, k; normalize = true, kwargs...)

# Example:
```julia
clusterresult = kbarycenters(
    dist,
    models,
    n_classes, # number of clusters
    seed       = :rand,
    solver     = sinkhorn_log!,
    tol        = 2e-6,
    innertol   = 2e-6,
    iters      = 100000,
    inneriters = 100000,
    verbose    = true,
    output     = :best,
    uniform    = true,
    kiters     = 10
)
```

# Arguments:
- `models`: A vector of models
- `k`: number of clusters
- `normalize`: Whether or not to normalize the weight vectors (recommended)
- `kwargs`: are sent to inner solvers, (`solver,tol,iters`)
"""
function kbarycenters(d::SinkhornRootDistance, models::Vector{<:AbstractModel}, k; normalize=true, kwargs...)
    d.p == 2 || throw(ArgumentError("p must be 2"))
    X, w, realpoles = barycenter_matrices(d, models, normalize)
    res = kbarycenters(X, w, k; β=d.β, kwargs...)
    KBResult(res.assignments, bc2model.(res.barycenters,realpoles), res.cost)
end
