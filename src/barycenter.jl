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

    # w = transpose.(w)
    # W = reduce(vcat,w)
    # W ./= sum(W,dims=1)
    # w2 = [Float64.(W[i,:]) for i in 1:length(X)]

    X, w, realpoles
end

"""
    barycenter(d::SinkhornRootDistance, models; normalize = true, kwargs...)
    $(SIGNATURES)

**Approximately** calculate the barycenter supported on the same number of atoms as the number of poles in the models.

#Arguments:
- `models`: vector of AR models
- `normalize`: make sure weights sum to 1
- `kwargs`: are sent to [`ISA`](@ref)
"""
function barycenter(d::SinkhornRootDistance,models, λ=s1(ones(length(models))); normalize=true, uniform=true, solver=IPOT, kwargs...)
    d.p == 2 || throw(ArgumentError("p must be 2"))
    X, w, realpoles = barycenter_matrices(d, models, normalize)
    if uniform
        bc = barycenter(X, λ; uniform=uniform, solver=solver, β=d.β, kwargs...)
    else
        bc = barycenter(X,w, λ; uniform=uniform, solver=solver, β=d.β, kwargs...)
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

# function barycenter(d::SinkhornRootDistance,models; normalize=true, kwargs...)
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

# function barycenter(d::SinkhornRootDistance,models,w; kwargs...)
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


function barycenter(d::EuclideanRootDistance,models::AbstractVector)
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    bc = map(1:length(r[1])) do pi
        sum(w[pi]*r[pi] for (w,r) in zip(w,r))/sum(w[pi] for w in w)
    end
    AR(ContinuousRoots(bc))
end


function distmat_euclidean(X::AbstractMatrix,Y::AbstractMatrix)
    C = [mean(abs2, c1-c2) for c1 in eachcol(X), c2 in eachcol(Y)]
    # C ./ median(C)
end

function distmat_euclidean!(C,X::AbstractMatrix,Y::AbstractMatrix)
    for (j,c2) in enumerate(eachcol(Y))
        for (i,c1) in enumerate(eachcol(X))
            C[i,j] = mean(((c1,c2),) -> abs2(c1-c2), zip(c1,c2))
        end
    end
    C
    # C ./ median(C)
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
    m = 0.1minimum(x->std(x), X)
    X0 .+= m .* randn.()
end

function barycenter(X::Vector{<:AbstractArray}, p, λ; uniform=true, solver=sinkhorn_log!, kwargs...)
    N = length(X)
    n = size(X[1],2)
    w = s1(ones(n))
    m = 0.1minimum(x->std(x), X)
    # X0 = mean(X)
    ind = rand(1:length(X))
    X0 = X[ind] .- mean(X[ind],dims=2) .+ mean(mean.(X, dims=2))
    X0 .+= m .* randn.()
    alg2(X0,X,w,p; solver=solver, weights=λ, uniform=uniform, kwargs...)
end

# the method below does not work very well due to likely bug in ISA
# function barycenter(X::Vector{<:AbstractArray}, λ; iters=100, kwargs...)
#     sw = ISA(X, λ; iters=iters, kwargs...)
#     barycentric_weighting(X,λ,sw)
# end

barycentric_weighting(X,λ,sw) = sum(λ[i].*X[i][:,sw[i]] for i in eachindex(sw))

# function barycentric_weighting2(X,λ,sw)
#     sum(λ[sw[i]]'.*X[i][:,sw[i]] for i in eachindex(sw))
# end # This one was definitely bad

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

#Arguments:
- `pl`: Atoms in measures `p`, vector, length `n_measures`, of matrices of size `n_dims × n_atoms`
- `ql`: Atoms in measure `q`
- `p`: Measures `p`, a matrix of weight vectors, size `n_atoms × n_measures` that sums to 1
- `q`: the veight vector for measure `q`, length is `n_atoms`
- `options`: For the Optim solver. Defaults are `options = Optim.Options(store_trace=false, show_trace=false, show_every=0, iterations=20, allow_f_increases=true, time_limit=100, x_tol=1e-5, f_tol=1e-6, g_tol=1e-6, f_calls_limit=0, g_calls_limit=0)`
- `solver`: = [`sinkhorn_log!`](@ref) solver
- `tol`:    = 1e-7 tolerance
- `β`:      = 1 entropy regularization. This function works best with rather large regularization, hence the large default value.
- `kwargs`: these are sent to the solver algorithm.
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
    β      = 1,
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
        solver = sinkhorn_log!,
        tol    = 1e-7,
        β      = 1,
        kwargs...)
    if robust
        res = Optim.optimize(costfun, λl, ParticleSwarm(), Optim.Options(iterations=100, store_trace=false))
        λl = res.minimizer
    end
    local λh
    # try
        res = Optim.optimize(costfun, λl, method, options, autodiff=:forward)
        λh = softmax(res.minimizer)
    # catch err
        # @error("Barycentric coordinates: optimization failed: ", err)
        # λh = softmax(λl)
    # end

    # cost,P,∇ℰ = sinkhorn_diff(pl,ql,p,q,C,λh; kwargs...)
    λh
end

function barycentric_coordinates(d::SinkhornRootDistance,models, qmodel, method=BFGS(); kwargs...)

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

    λ = barycentric_coordinates(pl, ql[1], p, vec(q), method; β=d.β, kwargs...)
    return λ
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

#Arguments:
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


## Measures with nonuniform weights, the bc should be pulled to atom 1 and 4 in the first measure. The trick seems to be to run ISA without weights and then use the weights to form the estimate

# d = 2
# k = 4
# X0 = [1 1 2 2; 1 2 1 2]
# X = [X0 .+ 0.3rand(d,k) .+ 0.020rand(d) for _ in 1:6]
# w = [ones(1,k) for _ in 1:length(X)]
# w[1][1] = 100
# w[1][4] = 100
# for i = 2:length(X)
#     w[i][1] = 0.01
#     w[i][4] = 0.01
# end
# w = s1.(w)
# W = reduce(vcat,w)
# W ./= sum(W,dims=1)
# w2 = [W[i,:]' for i in 1:length(X)]
# S = ISA(X, iters=1000, printerval=10)
# X̂ = [w2[i].*X[i][:,S[i]] for i in eachindex(X)]
# bc = sum(X̂)
# # @test mean(bc) ≈ mean(X[1]) rtol=0.1
# scatter(eachrow(reduce(hcat,X))...)
# scatter!([X[1][1,:]],[X[1][2,:]])
# scatter!(eachrow(bc)..., m=:square, legend=false, alpha=0.4)




## ======================
"""
    alg1(X, Y, â, b; β = 1, printerval = typemax(Int), tol = 1.0e-5, iters = 10000, solver = IPOT)

Algorithm 1 from "Fast Computation of Wasserstein Barycenters" https://arxiv.org/pdf/1310.4375.pdf Notation is the same as in the paper.

#Arguments:
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
function alg1(X,Y,â,b;β=1, printerval=typemax(Int), tol=1e-5, iters=10000, solver=IPOT, weights=nothing)
    N = length(Y)
    â = copy(â)
    a = copy(â)
    # fill!(â, 1/N)
    ã = copy(â)
    t0 = 1
    t = 0
    # weights = nothing
    𝛂 = similar(a, length(a), N)
    Mth = [distmat_euclidean(X,Y[1]) for i in 1:Threads.nthreads()]
    for outer t = 1:iters
        B = (t0+t)/2
        a .= (1-inv(B)).*â .+ inv(B).*ã
        for i in 1:N
            # Threads.@spawn begin
                M = distmat_euclidean!(Mth[Threads.threadid()], X,Y[i])
                ai = solver(M,a,b[i]; iters=50000, β=β, tol=tol)[2]
                if !all(isfinite, a)
                    @warn "Got nan in inner sinkhorn alg 1, increasing precision"
                    ai = solver(M,big.(â),big.(b[i]); iters=50000, β=β, tol=tol)[2]
                    ai = eltype(â).(ai)
                end
                scale!(ai, i, weights)
                𝛂[:,i] .= ai
            # end
        end

        # @show round.(vec(mean(𝛂, dims=2)), sigdigits=3)
        # @show round.(vec(mean(𝛂, dims=1)), sigdigits=3)
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
        # â ./= sum(â)
    end
    t > printerval && @info "Sinkhorn alg1 maximum number of iterations reached: $iters"
    â
end



"""
    alg2(X, Y, a, b;
            β = 1/10,
            θ = 0.5,
            printerval = typemax(Int),
            tol = 1.0e-6,
            innertol = 1.0e-5,
            iters = 500,
            inneriters = 1000,
            atol = 1.0e-32,
            solver = IPOT,
            γ = 1,
        )

Algorithm 2 from "Fast Computation of Wasserstein Barycenters" https://arxiv.org/pdf/1310.4375.pdf Notation is the same as in the paper.

#Arguments:
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
function alg2(X,Y,a,b; β = 1/10, θ = 0.5, printerval=typemax(Int), tol=1e-6, innertol=1e-4, iters=500, inneriters=1000, atol=1e-32, solver=IPOT, γ=0.0, weights=nothing, uniform=false)
    uniform || @warn("This function is known to be buggy when not enforcing uniform weights")
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
