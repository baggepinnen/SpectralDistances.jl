function barycenter_matrices(d, models, normalize=true; allow_shortcut=true)
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    realpoles = any(any(iszero ∘ imag, r) for r in r)

    realpoles = realpoles || !allow_shortcut # if we don't allow the shortcut it's the  same as if there are real poles.

    if !realpoles
        r = [r[1:end÷2] for r in r]
        w = [2w[1:end÷2] for w in w]
    end
    X = [[real(r)'; imag(r)'] for r in r]
    if !all(sum.(w) .≈ 1)
        if normalize
            w = s1.(w)
        else
            @warn "sum.(w) ≠ 1" sum.(w)
        end
    end

    w = transpose.(s1.(w))
    W = reduce(vcat,w)
    W ./= sum(W,dims=1)
    w2 = [W[i,:]' for i in 1:length(X)]

    X, w2, realpoles
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
function barycenter(d::SinkhornRootDistance,models; normalize=true, kwargs...)
    X, w, realpoles = barycenter_matrices(d, models, normalize)
    S = ISA(X; kwargs...)
    bc = barycentric_weighting(X,w,S)

    r1 = complex.(bc[1,:],bc[2,:])
    if realpoles
        bcr = ContinuousRoots(r1)
    else
        @assert !any(iszero ∘ imag, r1) "A real root was found in barycenter even though inputs had no real roots"
        bcr = ContinuousRoots([r1; conj.(r1)])
    end
    AR(bcr)
end

function barycenter(d::SinkhornRootDistance,models,w; kwargs...)
    r = roots.(SpectralDistances.Continuous(), models)
    realpoles = any(any(iszero ∘ imag, r) for r in r)

    if !realpoles
        r = [r[1:end÷2] for r in r]
    end
    X = [[real(r)'; imag(r)'] for r in r]

    S = ISA(X, w; kwargs...)
    X̂ = [X[i][:,S[i]] for i in eachindex(X)]
    bc = sum(w.*X̂)
    r1 = complex.(bc[1,:],bc[2,:])
    if realpoles
        bcr = ContinuousRoots(r1)
    else
        @assert !any(iszero ∘ imag, r1) "A real root was found in barycenter even though inputs had no real roots"
        bcr = ContinuousRoots([r1; conj.(r1)])
    end

    AR(bcr)
end


function barycenter(d::EuclideanRootDistance,models::AbstractVector)
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    bc = map(1:length(r[1])) do pi
        sum(w[pi]*r[pi] for (w,r) in zip(w,r))/sum(w[pi] for w in w)
    end
    AR(ContinuousRoots(bc))
end


function distmat_euclidean(X,Y)
    C = [mean(abs2, c1-c2) for c1 in eachcol(X), c2 in eachcol(Y)]
    # C ./ median(C)
end


"""
    barycenter(X::Vector{<:AbstractArray}, λ)

Calculate the weighted barycenter for point clouds in `X`.
Each `X[i]` has the shame `n_dims × n_atoms`
`λ` is the weight vector that should sum to 1.
"""
function barycenter(X::Vector{<:AbstractArray}, λ; iters=100, kwargs...)
    sw = ISA(X, λ; iters=iters, kwargs...)
    barycentric_weighting(X,λ,sw)
end

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
    proj,λ = barycentric_coordinates(pl, ql, p, q; options, kwargs...)

Compute the barycentric coordinates `λ` such that
sum(λᵢ W(pᵢ,q) for i in eachindex(p)) is minimized.

`proj` is the resulting projection of `(ql,q)` onto the space of atoms in `(pl,p)`, with coordinates `λ`

#Arguments:
- `pl`: Atoms in measures `p`, vector, length `n_measures`, of matrices of size `n_dims × n_atoms`
- `ql`: Atoms in measure `q`
- `p`: Measures `p`, a matrix of weight vectors, size `n_atoms × n_measures` that sums to 1
- `q`: the veight vector for measure `q`, length is `n_atoms`
- `options`: For the Optim solver. Defaults are `options = Optim.Options(store_trace=false, show_trace=false, show_every=0, iterations=20, allow_f_increases=true, time_limit=100, x_tol=0, f_tol=0, g_tol=1e-8, f_calls_limit=0, g_calls_limit=0)`
- `kwargs`: these are sent to the [`sinkhorn`](@ref) algorithm.
"""
function barycentric_coordinates(pl, ql, p, q, method=BFGS();
    options = Optim.Options(store_trace       = false,
                            show_trace        = false,
                            show_every        = 1,
                            iterations        = 20,
                            allow_f_increases = true,
                            time_limit        = 100,
                            x_tol             = 0,
                            f_tol             = 1e-6,
                            g_tol             = 1e-8,
                            f_calls_limit     = 0,
                            g_calls_limit     = 0),
    robust=true,
    kwargs...)

    C = [[mean(abs2, x1-x2) for x1 in eachcol(Xi), x2 in eachcol(ql)] for Xi in pl]
    # local P

    function fg!(F,G,λl)
        λ = softmax(λl) # optimization done in log domain
        cost,P,∇ℰ = SpectralDistances.sinkhorn_diff(pl,ql,p,q,C,λ; kwargs...)
        if G !== nothing
            G .= ∇ℰ
        end
        if F !== nothing
            return cost
        end
    end

    S = length(pl)
    λl = zeros(S)
    # λl = [0.0, 10]#zeros(S)
    # @warn "above for debugging purposes"
    if robust
        res = Optim.optimize(Optim.only_fg!(fg!), λl, NelderMead(), Optim.Options(iterations=10, store_trace=false))
        λl = res.minimizer
    end
    res = Optim.optimize(Optim.only_fg!(fg!), λl, method, options)
    λh = softmax(res.minimizer)

    cost,P,∇ℰ = SpectralDistances.sinkhorn_diff(pl,ql,p,q,C,λh; kwargs...)
    # iperm = sortperm(perm)
    P,λh
end

function barycentric_coordinates(d::SinkhornRootDistance,models, qmodel, method=BFGS(); kwargs...)
    pl, p, realpolesp = barycenter_matrices(d, models)
    ql, q, realpolesq = barycenter_matrices(d, [qmodel])
    if realpolesp != realpolesq
        pl, p, realpolesp = barycenter_matrices(d, models, allow_shortcut=false)
        ql, q, realpolesq = barycenter_matrices(d, [qmodel], allow_shortcut=false)
    end

    for p in p
        p ./= sum(p)
    end
    q = q[1] ./= sum(q[1])

    @assert sum(q) ≈ 1
    @assert all(sum(p) ≈ 1 for p in p)

    q_proj, λ = barycentric_coordinates(pl, ql[1], reduce(vcat,p)', vec(q), method; kwargs...)


    r1 = complex.(q_proj[1,:],q_proj[2,:])
    if realpolesp
        bcr = ContinuousRoots(r1)
    else
        @assert !any(iszero ∘ imag, r1) "A real root was found in barycenter even though inputs had no real roots"
        bcr = ContinuousRoots([r1; conj.(r1)])
    end
    AR(bcr), λ

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
function alg1(X,Y,â,b;λ=100, printerval=typemax(Int), tol=1e-5, iters=10000)
    N = length(Y)
    â = copy(â)
    # fill!(â, 1/N)
    ã = copy(â)
    t0 = 1
    t = 0
    for outer t = 1:iters
        β = (t0+t)/2
        â .= (1-inv(β)).*â .+ inv(β).*ã
        𝛂 = mean(1:N) do i
            M = distmat_euclidean(X,Y[i])
            a = sinkhorn_log(M,â,b[i]; iters=50, β=1/λ, tol=1e-3)[2]
            if !all(isfinite, a)
                @warn "Got nan in inner sinkhorn alg 1, increasing precision"
                a = sinkhorn_log(M,big.(â),big.(b[i]); iters=50, β=1/λ, tol=1e-3)[2]
                a = eltype(â).(a)
            end
            a
        end

        ã .= ã .* exp.(-β.*𝛂 .* t0)
        ã ./= sum(ã)
        aerr = sum(abs2,â-ã)
        t % printerval == 0 && @info "Sinkhorn alg1:  iter: $t, aerr: $aerr"
        if aerr < tol
            t > printerval && @info "Sinkhorn alg1 done at iter $t"
            return â .= (1-inv(β)).*â .+ inv(β).*ã
        end
        â .= (1-inv(β)).*â .+ inv(β).*ã
        # â ./= sum(â)
    end
    t > printerval && @info "Sinkhorn alg1 maximum number of iterations reached: $iters"
    â
end



function alg2(X,Y,a,b;λ = 2,θ = 0.5, printerval=typemax(Int), tol=1e-4, innertol=1e-3, iters=500, inneriters=50, atol=1e-16)
    N = length(Y)
    a = copy(a)
    ao = copy(a)
    X = copy(X)
    Xo = copy(X)
    fill!(ao, 1/length(ao))
    for iter = 1:iters
        a = alg1(X,Y,ao,b,λ=λ, printerval=printerval, tol=innertol, iters=inneriters)
        YT = mean(1:N) do i
            M = distmat_euclidean(X,Y[i])
            T,_ = SpectralDistances.sinkhorn_log(M,a,b[i]; iters=500, β=1/λ)
            @assert !any(isnan, T) "Got nan in sinkhorn alg 2"
            Y[i]*T'
        end
        error("consider a line search here")
        X .= (1-θ).*X .+ θ.*(YT / Diagonal(a .+ eps()))
        # @show mean(abs2, a-ao), mean(abs2, X-Xo)
        aerr = mean(abs2, a-ao)
        xerr = mean(abs2, X-Xo)
        iter % printerval == 0 && @info "Sinkhorn alg2:  iter: $iter, aerr: $aerr, xerr: $xerr"
        if aerr < atol || xerr < tol
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
#
# function sinkhorn2(C, a, b; λ, iters=1000)
#     K = exp.(.-C .* λ)
#     K̃ = Diagonal(a) \ K
#     u = one.(b)./length(b)
#     uo = copy(u)
#     for iter = 1:iters
#         u .= 1 ./(K̃*(b./(K'uo)))
#         # @show sum(abs2, u-uo)
#         if sum(abs2, u-uo) < 1e-10
#             # @info "Done at iteration $iter"
#             break
#         end
#         copyto!(uo,u)
#     end
#     @assert all(!isnan, u) "Got nan entries in u"
#     u .= max.(u, 1e-20)
#     @assert all(>(0), u) "Got non-positive entries in u"
#     v = b ./ ((K' * u) .+ 1e-20)
#     if any(isnan, v)
#         @show (K' * u)
#         error("Got nan entries in v")
#     end
#     lu = log.(u)# .+ 1e-100)
#     α = -lu./λ .+ sum(lu)/(λ*length(u))
#     α .-= sum(α) # Normalize dual optimum to sum to zero
#     Diagonal(u) * K * Diagonal(v), α
# end
