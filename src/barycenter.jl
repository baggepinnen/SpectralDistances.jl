function barycenter_matrices(d, models, normalize=true; allow_shortcut=true)
    r = roots.(SpectralDistances.Continuous(), models)
    w = [Float64.(d.weight(r)) for r in r]
    realpoles = any(any(iszero ∘ imag, r) for r in r)

    realpoles = realpoles || !allow_shortcut # if we don't allow the shortcut it's the  same as if there are real poles.

    if !realpoles
        r = [r[1:end÷2] for r in r]
        w = [2w[1:end÷2] for w in w]
    end
    X = [[Float64.(real(r)'); Float64.(imag(r)')] for r in r]
    if !all(sum.(w) .≈ 1)
        if normalize
            w = s1.(w)
        else
            @warn "sum.(w) ≠ 1" sum.(w)
        end
    end

    X, w, realpoles
end

function embedding(model::AR,args...)
    embedding(roots(SpectralDistances.Continuous(), model), args...)
end

"""
    embedding([::Type{Vector}], m, [full=true])

Returns a `Vector/Matrix` containing the roots of `m`.
`full` indicates whether or not to use all poles or only one half-plane.
"""
embedding(::Type{Vector}, model, args...) = embedding(model, args...)
embedding(::Type{Matrix}, model, args...) = reshape(embedding(model, args...), :, 2)

function embedding(r::AbstractRoots, full::Bool=true)
    if full
        [real.(r); imag.(r)]
    else
        realpoles = any(any(iszero ∘ imag, r) for r in r)
        realpoles && @error("Model contained real poles, this produces weird results when used in an embedding")
        @views [real.(r[1:end÷2]); imag.(r[1:end÷2])]
    end
end

"""
    barycenter(d::OptimalTransportRootDistance, models; normalize = true, kwargs...)
    $(SIGNATURES)

**Approximately** calculate the barycenter supported on the same number of atoms as the number of poles in the models.

The solver can be selected by providing a keword argument, example: `solver=IPOT`.

Uses the algorithms from ["Fast Computation of Wasserstein Barycenters"](https://arxiv.org/pdf/1310.4375.pdf)

# Example:
```julia
models = examplemodels(10)

d = OptimalTransportRootDistance(domain=SpectralDistances.Continuous(),p=2, weight=residueweight, β=0.01)
Xe = barycenter(d, models, solver=sinkhorn_log!, uniform=true)

plot()
pzmap!.(models)
pzmap!(tf(Xe), m=:c, title="Barycenter OptimalTransportRootDistance", lab="BC")
```

# Arguments:
- `models`: vector of AR models
- `normalize`: make sure weights sum to 1
- `kwargs`: are sent to the solver
"""
function barycenter(d::OptimalTransportRootDistance,models, λ=s1(ones(length(models))); normalize=true, uniform=true, solver=sinkhorn_log!, kwargs...)
    d.p == 2 || throw(ArgumentError("p must be 2"))
    X, w, realpoles = barycenter_matrices(d, models, normalize)
    if uniform
        bc = barycenter(X, λ; uniform=true, solver=solver, β=d.β, kwargs...)
    else
        bc,w = barycenter(X,w, λ; uniform=false, solver=solver, β=d.β, kwargs...)
    end
    bc2model(bc, realpoles)

end

function bc2model(bc, realpoles)
    r1 = complex.(bc[1,:],bc[2,:])
    if realpoles
        bcr = ContinuousRoots(r1)
    else
        @assert !any(iszero ∘ imag, r1) "A real root was found in barycenter even though inputs had no real roots"
        bcr = ContinuousRoots([r1; conj.(r1)])
    end
    AR(bcr)
end


"""
    barycenter(d::EuclideanRootDistance, models::AbstractVector, [λ])

# Example:
```julia
models = examplemodels(10)

Xe = barycenter(EuclideanRootDistance(domain=SpectralDistances.Continuous(),p=2), models)

G = tf.(models)
plot()
pzmap!.(G)
pzmap!(tf(Xe), m=:c, title="Barycenter EuclideanRootDistance")
current()
```
"""
function barycenter(d::EuclideanRootDistance,models::AbstractVector;kwargs...)
    r = roots.(SpectralDistances.Continuous(), models)
    w = map(d.weight, r)
    bc = map(1:length(r[1])) do pi
        sum(w[pi]*r[pi] for (w,r) in zip(w,r))/sum(w[pi] for w in w)
    end
    AR(ContinuousRoots(bc))
end

function barycenter(d::EuclideanRootDistance,models::AbstractVector, λ;kwargs...)
    N = length(models)
    r = roots.(SpectralDistances.Continuous(), models)
    w = map(d.weight, r)
    bc = map(1:length(r[1])) do pi
        sum(λ[i]*w[pi]*r[pi] for (i,w,r) in zip(1:N,w,r))/sum(λ[i]*w[pi] for (i,w) in enumerate(w))
    end
    AR(ContinuousRoots(bc))
end


function distmat_euclidean(X::AbstractMatrix,Y::AbstractMatrix)
    C = [mean(abs2, c1-c2) for c1 in eachcol(X), c2 in eachcol(Y)]
end

function distmat_euclidean!(C,X::AbstractMatrix,Y::AbstractMatrix)
    for (j,c2) in enumerate(eachcol(Y))
        for (i,c1) in enumerate(eachcol(X))
            C[i,j] = mean(((c1,c2),) -> abs2(c1-c2), zip(c1,c2))
        end
    end
    C
end


"""
    barycenter(X::Vector{<:AbstractArray}, λ)

Calculate the weighted barycenter for point clouds in `X`.
Each `X[i]` has the shame `n_dims × n_atoms`
`λ` is the weight vector that should sum to 1.
"""
function barycenter(X::Vector{<:AbstractArray}, λ; uniform=true, solver=sinkhorn_log, kwargs...)
    N = length(X)
    n = size(X[1],2)
    w = s1(ones(n))

    # X0 = mean(X)
    ind = rand(1:length(X))
    X0 = X[ind] .- mean(X[ind],dims=2) .+ mean(mean.(X, dims=2))
    perturb!(X0,X)
    bc = alg2(X0,X,w,fill(w,N); solver=solver, weights=λ, uniform=uniform, kwargs...)[1]
end

function perturb!(X0,X)
    m = 0.1minimum(std, X)
    X0 .+= m .* randn.()
end

function barycenter(X::Vector{<:AbstractArray}, p, λ; uniform=true, solver=sinkhorn_log!, kwargs...)
    N = length(X)
    n = size(X[1],2)
    w = s1(ones(n))
    m = 0.1minimum(std, X)
    # X0 = mean(X)
    ind = rand(1:length(X))
    X0 = X[ind] .- mean(X[ind],dims=2) .+ mean(mean.(X, dims=2))
    X0 .+= m .* randn.()
    alg2(X0,X,w,p; solver=solver, weights=λ, uniform=uniform, kwargs...)
end


barycentric_weighting(X,λ,sw) = sum(λ[i].*X[i][:,sw[i]] for i in eachindex(sw))

function softmax(x)
    e = exp.(x)
    e ./= sum(e)
end

using Random

"""
    λ = barycentric_coordinates(pl, ql, p, q; options, kwargs...)

Compute the barycentric coordinates `λ` such that
sum(λᵢ W(pᵢ,q) for i in eachindex(p)) is minimized.

This function works best with the `sinkhorn_log!` solver, a large β (around 1) and small tolerance. These are set using `kwargs...`.

# Arguments:
- `pl`: Atoms in measures `p`, vector, length `n_measures`, of matrices of size `n_dims × n_atoms`
- `ql`: Atoms in measure `q`
- `p`: Measures `p`, a matrix of weight vectors, size `n_atoms × n_measures` that sums to 1
- `q`: the veight vector for measure `q`, length is `n_atoms`
- `options`: For the Optim solver. Defaults are `options = Optim.Options(store_trace=false, show_trace=false, show_every=0, iterations=20, allow_f_increases=true, time_limit=100, x_tol=1e-5, f_tol=1e-6, g_tol=1e-6, f_calls_limit=0, g_calls_limit=0)`
- `solver`: = [`sinkhorn_log!`](@ref) solver
- `tol`:    = 1e-7 tolerance
- `β`:      = 0.1 entropy regularization. This function works best with rather large regularization, hence the large default value.
- `kwargs`: these are sent to the solver algorithm.

# Example:
```julia
using SpectralDistances, ControlSystems, Optim
models = examplemodels(10)

d = OptimalTransportRootDistance(
    domain = SpectralDistances.Continuous(),
    p      = 2,
    weight = residueweight,
    β      = 0.01,
)
Xe = barycenter(d, models, solver=sinkhorn_log!)

G = tf.(models)
plot()
pzmap!.(G)
pzmap!(tf(Xe), m=:c, title="Barycenter OptimalTransportRootDistance", lab="BC")

options = Optim.Options(store_trace       = true,
                        show_trace        = false,
                        show_every        = 1,
                        iterations        = 50,
                        allow_f_increases = true,
                        time_limit        = 100,
                        x_tol             = 1e-7,
                        f_tol             = 1e-7,
                        g_tol             = 1e-7,
                        f_calls_limit     = 0,
                        g_calls_limit     = 0)


method = LBFGS()
λ = barycentric_coordinates(d, models, Xe, method,
    options = options,
    solver  = sinkhorn_log!,
    robust  = true,
    uniform = true,
    tol     = 1e-6,
)
bar(λ, title="Barycentric coorinates")

G = tf.(models)
plot()
pzmap!.(G, lab="")
pzmap!(tf(Xe), m = :c, title = "Barycenter OptimalTransportRootDistance", lab = "BC")
# It's okay if the last system dot does not match the barycenter exactly, there are limited models to choose from.
pzmap!(G[argmax(λ)], m = :c, lab = "Largest bc coord", legend = true)
```
"""
function barycentric_coordinates(pl, ql, p, q::AbstractVector{T}, method=LBFGS();
    options = Optim.Options(store_trace       = false,
                            show_trace        = false,
                            show_every        = 1,
                            iterations        = 20,
                            allow_f_increases = true,
                            time_limit        = 100,
                            x_tol             = 1e-5,
                            f_tol             = 1e-6,
                            g_tol             = 1e-6,
                            f_calls_limit     = 0,
                            g_calls_limit     = 0),
    robust = true,
    solver = sinkhorn_log!,
    tol    = 1e-7,
    β      = 0.1,
    plot   = nothing,
    kwargs...) where T

    # C = [[mean(abs2, x1-x2) for x1 in eachcol(Xi), x2 in eachcol(ql)] for Xi in pl]

    S = length(pl)
    k = length(p[1])
    # λl = 1e-8randn(S)
    # dists = map(i->sum(C[i].*IPOT(C[i],p[i],q)[1]), eachindex(p))
    # λl = -v1(dists) # Initial guess based on distances between anchors and query point
    # λl .*= 0sqrt(length(λl)) # scale so that softmax(λ) is reasonably sparse.
    # randn!(λl)
    # return softmax(λl)
    # function fg!(F,G,λl)
    #     λ = softmax(λl) # optimization done in log domain
    #     local cost = 0.0
    #     if G !== nothing
    #         cost,P,∇ℰ = sinkhorn_diff(pl,ql,p,q,C,λ; L=L, kwargs...)
    #         G .= ∇ℰ
    #     end
    #     if F !== nothing
    #         if G === nothing
    #             cost = sinkhorn_cost(pl,ql,p,q,C,λ; L=L, kwargs...)
    #         end
    #         return cost
    #     end
    # end
    # if robust
    #     res = Optim.optimize(Optim.only_fg!(fg!), λl, NelderMead(), Optim.Options(iterations=60, store_trace=false))
    #     λl = res.minimizer
    # end
    # res = Optim.optimize(Optim.only_fg!(fg!), λl, method, options)

    λl = zeros(T,S)
    C = zeros(T,k,k)
    costfun = λ -> sinkhorn_cost(pl, ql, p, q, softmax(λ);
        solver = solver,
        tol    = tol,
        β      = β,
        kwargs...)

    if plot !== nothing
        @assert S == 2 "Can only plot for two anchor measures"
        cf = x -> costfun(log.([x,1-x]))
        plot(LinRange(1e-3, 1-1e-3, 100), cf) |> display
    end
    if robust
        res = Optim.optimize(costfun, λl, ParticleSwarm(), Optim.Options(iterations=100, store_trace=false))
        λl = res.minimizer
    end
    local λh
    try
        res = Optim.optimize(costfun, λl, method, options, autodiff=:forward)
        λh = softmax(res.minimizer)
    catch err
        @error("Barycentric coordinates: optimization failed, retrying with robust options: ", err)
        res = Optim.optimize(costfun, λl, NelderMead(), options)
        λh = softmax(res.minimizer)
    end

    # cost,P,∇ℰ = sinkhorn_diff(pl,ql,p,q,C,λh; kwargs...)
    λh
end

function barycentric_coordinates(d::OptimalTransportRootDistance,models, qmodel, method=BFGS(); kwargs...)

    d.p == 2 || throw(ArgumentError("p must be 2"))
    pl, p, realpolesp = barycenter_matrices(d, models)
    ql, q, realpolesq = barycenter_matrices(d, [qmodel])
    if realpolesp != realpolesq
        pl, p, realpolesp = barycenter_matrices(d, models, allow_shortcut=false)
        ql, q, realpolesq = barycenter_matrices(d, [qmodel], allow_shortcut=false)
    end

    for pi in p
        pi ./= sum(pi)
    end
    q = q[1] ./= sum(q[1])
    @assert sum(q) ≈ 1
    @assert all(sum(p) ≈ 1 for p in p)

    if d.divergence !== nothing
        if :solver ∈ keys(kwargs)
            throw(ArgumentError("Solver choice not available for unbalanced distances, solver `sinkhorn_unbalanced` will be used."))
        end
        solver, kwargs = unbalanced_solver_closure(d, kwargs...)
    else
        solver = get(kwargs, :solver, IPOT)
    end
    λ = barycentric_coordinates(pl, ql[1], p, vec(q), method; solver=solver, β=d.β, kwargs...)
    return λ
end


function barycentric_coordinates(d::EuclideanRootDistance, models, qmodel, args...; kwargs...)
    d.p == 2 || throw(ArgumentError("p must be 2"))
    N = length(models)
    pl = embedding.(models)
    ql = embedding(qmodel)
    if d.weight != unitweight
        return _wrd_barycentric_coordinates(d,models,pl,ql)
    end
    ⅅ  = reduce(hcat, pl) # Dictionary matrix
    TotalLeastSquares.sls(Float64.(ⅅ), Float64.(ql); kwargs...)
end

function _wrd_barycentric_coordinates(d,models,pl,ql)
    N = length(models)

    R = roots.(domain(d), models)
    w = map(d.weight, R)
    w2 = [Float64.(getindex.(w, i)) for i in 1:length(w[1])]
    w2 = [w2;w2]

    P = [Float64.(getindex.(pl, i)) for i in 1:length(pl[1])]
    n = length(ql)

    function scostfun(λ)
        λ = softmax(λ)
        sum(abs2(dot(λ,Diagonal(P[j]),w2[j])/dot(λ,w2[j]) - ql[j]) for j in 1:n)
    end

    res = Optim.optimize(scostfun, zeros(N), LBFGS(m=20), Optim.Options(store_trace=false, show_trace=false, show_every=1, iterations=100, allow_f_increases=false, time_limit=100, x_tol=0, f_tol=0, g_tol=1e-8, f_calls_limit=0, g_calls_limit=0), autodiff=:forward)

    softmax(res.minimizer)
end


##
"Sum over j≠i. Internal function."
function ∑jni(X,i,S,k)
    s = zero(X[1][:,1])
    d = size(X[1],1)
    @inbounds for j in eachindex(X)
        j == i && continue
        for l in 1:d
            s[l] += X[j][l,S[j][k]]
        end
    end
    s
end

"""
    ISA(X, w = nothing; iters = 100, printerval = typemax(Int))

Iterative swapping algorithm from "On the Computation of Wasserstein barycenters", Giovanni Puccetti et al.

# Arguments:
- `X`: vector of d×k matrices where d is dimension and k number of atoms
- `w`: weights. See the files `test_barycenter.jl` for different uses.
- `iters`: maximum number of iterations
- `printerval`: print this often
"""
function ISA(X, w=nothing; iters=100, printerval = typemax(Int))
    n = length(X)
    d,k = size(X[1])

    if w !== nothing
        # X = deepcopy(X)
        X = map(eachindex(X)) do i # This does both copy and handles weird input types
            X[i] .* w[i] # This should work for both w[i] scalar and vector
        end
    end

    fudgefactor = 1.0

    σ = [collect(1:k) for _ in 1:n] # let σᵢ = Id, 1 ≤ i ≤ n.
    σ′ = deepcopy(σ)
    @inbounds for iter = 1:iters
        swaps = 0
        for i = 1:n
            σᵢ = σ[i]
            σᵢ′ = σ′[i]
            for k₁ = 1:k-1, k₂ = k₁+1:k
                Xik1 = @view X[i][:,σᵢ[k₁]]
                Xik2 = @view X[i][:,σᵢ[k₂]]
                if dot(Xik1, ∑jni(X,i,σ,k₁)) + dot(Xik2, ∑jni(X,i,σ,k₂)) < (dot(Xik2, ∑jni(X,i,σ,k₁)) + dot(Xik1, ∑jni(X,i,σ,k₂)))*fudgefactor
                    σᵢ′[k₁],σᵢ′[k₂] = σᵢ[k₂],σᵢ[k₁]
                    swaps += 1
                end
            end
        end
        iter % printerval == 0 && @show iter, swaps
        swaps == 0 && (return σ)
        σ = deepcopy(σ′) # Update assignment
        fudgefactor *= 1-1/iters^3
    end
    σ
end



## ======================
"""
    alg1(X, Y, â, b; β = 1, printerval = typemax(Int), tol = 1.0e-5, iters = 10000, solver = IPOT)

Algorithm 1 from ["Fast Computation of Wasserstein Barycenters"](https://arxiv.org/pdf/1310.4375.pdf) Notation is the same as in the paper.

# Arguments:
- `X`: Initial guess for barycenter support points
- `Y`: Support points for measures to calc barycenter of
- `a`: initial guess of barycenter weights
- `b`: Weigts of measures in Y
- `β`: Reg param, higher is more reg (inverse of λ in paper)
- `printerval`: DESCRIPTION
- `tol`: DESCRIPTION
- `iters`: DESCRIPTION
- `solver`: any of [`IPOT`](@ref) (default), [`sinkhorn`](@ref), [`sinkhorn_log`](@ref)
"""
function alg1(X,Y,â::AbstractVector{T},b;β=1, printerval=typemax(Int), tol=1e-5, iters=10000, solver=IPOT, weights=nothing)::Vector{T} where T
    N = length(Y)
    â = copy(â)
    a = copy(â)
    ã = copy(â)
    t0 = 1
    t = 0
    𝛂 = similar(a, length(a), N)
    Mth = [distmat_euclidean(X,Y[1]) for i in 1:Threads.nthreads()]
    for outer t = 1:iters
        B = (t0+t)/2
        a .= (1-inv(B)).*â .+ inv(B).*ã
        @sync for i in 1:N
            Threads.@spawn begin
                M = distmat_euclidean!(Mth[Threads.threadid()], X,Y[i])
                ai = solver(M,a,b[i]; iters=50000, β=β, tol=tol)[2]
                if !all(isfinite, a)
                    @warn "Got nan in inner sinkhorn alg 1, increasing precision"
                    ai = solver(M,big.(â),big.(b[i]); iters=50000, β=β, tol=tol)[2]
                    ai = eltype(â).(ai)
                end
                scale!(ai, i, weights)
                𝛂[:,i] .= ai
            end
        end

        ã .= ã .* exp.((-t0*B).*vec(mean(𝛂, dims=2)))
        ã ./= sum(ã)
        aerr = sum(abs2,â-ã)
        t % printerval == 0 && @info "Sinkhorn alg1:  iter: $t, aerr: $aerr"
        â .= (1-inv(B)).*â .+ inv(B).*ã
        â ./ sum(â)
        if aerr < tol
            t > printerval && @info "Sinkhorn alg1 done at iter $t"
            return â
        end
    end
    t > printerval && @info "Sinkhorn alg1 maximum number of iterations reached: $iters"
    â
end



"""
    alg2(X, Y, a, b;
            β          = 1/10,
            θ          = 0.5,
            printerval = typemax(Int),
            tol        = 1.0e-6,
            innertol   = 1.0e-5,
            iters      = 500,
            inneriters = 1000,
            atol       = 1.0e-32,
            solver     = IPOT,
            γ          = 1,
        )

Algorithm 2 from ["Fast Computation of Wasserstein Barycenters"](https://arxiv.org/pdf/1310.4375.pdf) Notation is the same as in the paper.

# Arguments:
- `X`: Initial guess for barycenter support points
- `Y`: Support points for measures to calc barycenter of
- `a`: initial guess of barycenter weights
- `b`: Weigts of measures in Y
- `β`: Reg param, higher is more reg
- `θ`: step size ∈ [0,1]
- `printerval`: print this often
- `tol`: outer tolerance
- `innertol`: inner tolerance
- `solver`: any of [`IPOT`](@ref) (default), [`sinkhorn`](@ref), [`sinkhorn_log`](@ref)
- `γ`: Sparsity parameter, if <1, encourage a uniform weight vector, if >1, do the opposite. Kind of like the inverse of α in the Dirichlet distribution.
"""
function alg2(X,Y,a,b; β = 1/10, θ = 0.5, printerval=typemax(Int), tol=1e-6, innertol=1e-4, iters=500, inneriters=10000, atol=1e-32, solver=IPOT, γ=0.0, weights=nothing, uniform=false)
    uniform || @warn("This function is known to be buggy when not enforcing uniform weights", maxlog=10)
    N = length(Y)
    a = copy(a)
    ao = copy(a)
    X = copy(X)
    if weights !== nothing && eltype(weights) != eltype(X)
        X = convert(Matrix{eltype(weights)}, X)
    end
    Xo = copy(X)
    weights === nothing && (weights = fill(1/N, N))
    fill!(ao, 1/length(ao))
    for iter = 1:iters
        uniform || (a = alg1(X,Y,ao,b,β=β, printerval=printerval, tol=innertol, iters=inneriters, solver=solver, weights=weights))
        if γ > 0 && γ != 1
            a .= softmax(γ.*log.(a))
        else
            a = s1(a)
        end
        YT = sum(1:N) do i
            M = distmat_euclidean(X,Y[i])
            T,_ = solver(M,a,b[i]; iters=inneriters, β=β, tol=innertol)
            @assert !any(isnan, T) "Got nan in sinkhorn alg 2"
            scale!(Y[i]*T', i, weights)
        end

        X .= (1-θ).*X .+ θ.*(YT ./ (a' .+ eps()))
        aerr = mean(abs2, a-ao)
        xerr = mean(abs2, X-Xo)
        iter % printerval == 0 && @info "Sinkhorn alg2:  iter: $iter, aerr: $aerr, xerr: $xerr"
        if xerr < tol
            iter > printerval && @info "Sinkhorn alg2 done at iter $iter"
            return X,(a./=sum(a))
        end
        copyto!(ao,a)
        copyto!(Xo,X)
        ao ./= sum(ao)
        # θ *= 0.999
    end
    iters > printerval && @info "Sinkhorn alg2 maximum number of iterations reached: $iters"
    X,a
end

scale!(x,_,::Nothing) = x
scale!(x::AbstractArray{T},i,w::AbstractArray{T}) where T = (x .*= w[i])
scale!(x,i,w::AbstractArray) = (x * w[i]) # for dual numbers etc.




struct BCWorkspace{T,TK}
    K::TK
    KV::Array{T,3}
    U::Array{T,3}
    b::Array{T,2}
    bold::Array{T,2}
    S::Array{T,2}
end

function BCWorkspace(A, β)
    N    = length(A)
    m, n = size(A[1])
    T    = eltype(eltype(A))
    β    = T(β)

    b    = zeros(T, m, n)
    bold = zeros(T, m, n)
    U    = ones(T, m, n, N)
    KV   = ones(T, m, n, N)
    S1   = Matrix{T}(undef, m, n)
    S2   = Matrix{T}(undef, m, n)

    t    = LinRange(T(0), T(1), m)
    Y, X = meshgrid(t, t)
    xi1  = @. exp(-(X - Y)^2 / β)

    t    = LinRange(T(0), T(1), n)
    Y, X = meshgrid(t, t)
    xi2  = @. exp(-(X - Y)^2 / β)

    function K(u,x)
        # xi1 * x * xi2
        mul!(S2,x,xi2)
        mul!(u,xi1,S2)
    end
    BCWorkspace(K, KV, U, b, bold, S1)
end

"""
    barycenter_convolutional(A, [λ]; β = 0.01, iters = 1000, tol = 1e-9, ϵ = 1e-30)


´β` is the regularization and `λ` (optional) are the weights (barycentric coordinates). To reuse allocated space ebtween successive calls, use the "workspace" method from the example below.
```julia
a1      = zeros(Float32, 10,10)
a1[2,2] = 1
a2      = zeros(Float32, 10,10)
a2[6,6] = 1
A       = [a1,a2]
β       = 0.01
λ       = [0.5, 0.5] # Barycentric coordinates, must sum to 1
w       = BCWorkspace(A, β)
b       = barycenter_convolutional(w,A,λ)
plot(heatmap(a1), heatmap(a2), heatmap(b))
```

Ref: J. Solomon, F. de Goes, G. Peyré, M. Cuturi, A. Butscher, A. Nguyen, T. Du, L. Guibas. Convolutional Wasserstein Distances: Efficient Optimal Transportation on Geometric Domains.  2015 https://people.csail.mit.edu/jsolomon/assets/convolutional_w2.compressed.pdf
"""
function barycenter_convolutional(
    w::BCWorkspace{T,TK},
    A,
    λ = Fill(1 / length(models), length(models));
    iters = 1000,
    tol = 1e-6,
    ϵ = 1e-40,
) where {T,TK}

    sum(λ) ≈ 1 || throw(ArgumentError("sum of barycentric coordinates λ was $(sum(λ)) but should be 1"))
    N = length(A)
    K, KV, U, b, bold, S = w.K, w.KV, w.U, w.b, w.bold, w.S
    iter = 0
    err = one(T)

    while err > tol && iter < iters
        copyto!(bold, b)
        iter = iter + 1
        b .= 0
        for r = 1:N
            temp = @view KV[:, :, r]
            @views K(S, U[:, :, r])
            @avx S .= A[r] ./ max.(ϵ, S)
            K(temp, S)
            @views @avx @. b += λ[r] * log(max(ϵ, U[:, :, r] * KV[:, :, r]))
        end
        @avx b .= exp.(b)
        @avx b ./= sum(b)
        for r = 1:N
            @views @avx @. U[:, :, r] = b / max(ϵ, KV[:, :, r])
        end

        if iter % 10 == 1
            err = sum(abs(bold - b) for (bold, b) in zip(bold, b))
        end
    end
    b
end

function barycenter_convolutional(
    A,
    λ = Fill(1 / length(models), length(models));
    β = 0.001,
    kwargs...,
)
    w = BCWorkspace(A, β)
    barycenter_convolutional(w, A, λ; kwargs...)
end

function meshgrid(a,b)
    grid_a = [i for i in a, j in b]
    grid_b = [j for i in a, j in b]
    grid_a, grid_b
end


function barycenter_convolutional(
    models::Vector{<:DSP.Periodograms.TFR},
    λ = Fill(1 / length(models), length(models));
    kwargs...,
)
    ss = x -> sqrt.(x)
    A  = s1.(ss.(power.(models)))
    b  = barycenter_convolutional(A, λ; kwargs...)
    B  = deepcopy(models[1])
    Bp = power(B)
    Bp .= abs2.(b)
    B
end




# function barycenter(d::OptimalTransportRootDistance,models; normalize=true, kwargs...)
#     X, w, realpoles = barycenter_matrices(d, models, normalize)
#     S = ISA(X; kwargs...)
#     bc = barycentric_weighting(X,w,S)
#
#     r1 = complex.(bc[1,:],bc[2,:])
#     if realpoles
#         bcr = ContinuousRoots(r1)
#     else
#         @assert !any(iszero ∘ imag, r1) "A real root was found in barycenter even though inputs had no real roots"
#         bcr = ContinuousRoots([r1; conj.(r1)])
#     end
#     AR(bcr)
# end
# function barycenter(d::OptimalTransportRootDistance,models,w; kwargs...)
#     r = roots.(SpectralDistances.Continuous(), models)
#     realpoles = any(any(iszero ∘ imag, r) for r in r)
#
#     if !realpoles
#         r = [r[1:end÷2] for r in r]
#     end
#     X = [[real(r)'; imag(r)'] for r in r]
#
#     S = ISA(X, w; kwargs...)
#     X̂ = [X[i][:,S[i]] for i in eachindex(X)]
#     bc = sum(w.*X̂)
#     r1 = complex.(bc[1,:],bc[2,:])
#     if realpoles
#         bcr = ContinuousRoots(r1)
#     else
#         @assert !any(iszero ∘ imag, r1) "A real root was found in barycenter even though inputs had no real roots"
#         bcr = ContinuousRoots([r1; conj.(r1)])
#     end
#
#     AR(bcr)
# end
# the method below does not work very well due to likely bug in ISA
# function barycenter(X::Vector{<:AbstractArray}, λ; iters=100, kwargs...)
#     sw = ISA(X, λ; iters=iters, kwargs...)
#     barycentric_weighting(X,λ,sw)
# end
# function barycentric_weighting2(X,λ,sw)
#     sum(λ[sw[i]]'.*X[i][:,sw[i]] for i in eachindex(sw))
# end # This one was definitely bad
## ==========================================================
# function simplex_ls(ⅅ, ql; iters=2000, verbose=false, kwargs...)
#
#     N = length(ql)
#     α = 1.0
#     ε = 0.1/N
#     @show λ = max.(ⅅ\ql .+ 0.01 .* randn.(), 0)
#     λ ./= sum(λ)
#     λo = copy(λ)
#     g = similar(λ)
#     local ng, err
#     err = 0.0
#     for iter = 1:iters # Do a few projected gradient iterations with soft thresholding
#         g .= ⅅ'*(ql-ⅅ*λ) # This is the negative gradient so + below
#         ng = norm(g)
#         ng < 1e-10 && break
#         λ .+= α.*g
#         λ .-= minimum(λ)
#         λ .= soft_th.(λ, ε)
#         λ ./= sum(λ)
#         α *= 0.999
#         err = norm(λ-λo)
#         verbose && @info "Iter $iter norm(g): $ng norm(λ-λo): $err"
#         err < 1e-5 && break
#         λo .= λ
#     end
#     verbose &&  @info "Converged norm(g): $ng norm(λ-λo): $err"
#     λ
# end
