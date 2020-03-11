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

function barycenter_matrices(d, models, normalize=true)
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    realpoles = any(any(iszero ∘ imag, r) for r in r)

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


function barycentric_coordinates(d::SinkhornRootDistance,models, q; kwargs...)
    pl, p, realpolesp = barycenter_matrices(d, models)
    ql, q, realpolesq = barycenter_matrices(d, [q])
    barycentric_coordinates(pl, ql[1], p, q[1]; kwargs...)
end





# function barycenter(d::SinkhornRootDistance,models,w; kwargs...)
#     # bc = barycenter(EuclideanRootDistance(domain=domain(d), p=d.p, weight=residueweight),models).pc
#     # X0 = [real(bc)'; imag(bc)']
#
#     # TODO: would be faster to only run on half of the poles and then duplicate them in the end. Would also enforce conjugacy. Special fix needed for systems with real poles.
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
#     bc = sum(X̂)
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


function distmat_euclidean(X,Y)
    [mean(abs2, c1-c2) for c1 in eachcol(X), c2 in eachcol(Y)]
end


"""
    barycenter(X::Vector{<:AbstractArray}, λ)

Calculate the weighted barycenter for point clouds in `X`.
Each `X[i]` has the shame `n_dims × n_atoms`
`λ` is the weight vector that should sum to 1.
"""
function barycenter(X::Vector{<:AbstractArray}, λ)
    sw = ISA(X, λ, iters=100, printerval=10)
    barycentric_weighting(X,λ,sw)
end

barycentric_weighting(X,λ,sw) = sum(λ[i].*X[i][:,sw[i]] for i in eachindex(sw))

function softmax(x)
    e = exp.(x)
    return e./sum(e)
end


"""
    bc,λ = barycentric_coordinates(pl, ql, p, q; options, kwargs...)

Compute the barycentric coordinates `λ` such that
sum(λᵢ W(pᵢ,q) for i in eachindex(p)) is minimized.

#Arguments:
- `pl`: Atoms in measures `p`, vector, length `n_measures`, of matrices of size `n_dims × n_atoms`
- `ql`: Atoms in measure `q`
- `p`: Measures `p`, a matrix of weight vectors, size `n_atoms × n_measures` that sums to 1
- `q`: the veight vector for measure `q`, length is `n_atoms`
- `options`: For the Optim solver. Defaults are `options = Optim.Options(store_trace=false, show_trace=false, show_every=0, iterations=20, allow_f_increases=true, time_limit=100, x_tol=0, f_tol=0, g_tol=1e-8, f_calls_limit=0, g_calls_limit=0)`
- `kwargs`: these are sent to the [`sinkhorn`](@ref) algorithm.
"""
function barycentric_coordinates(pl, ql, p, q;
    options = Optim.Options(store_trace=false, show_trace=false, show_every=0, iterations=20, allow_f_increases=true, time_limit=100, x_tol=0, f_tol=0, g_tol=1e-8, f_calls_limit=0, g_calls_limit=0),
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
    res = Optim.optimize(Optim.only_fg!(fg!), λl, BFGS(), options)
    λh = softmax(res.minimizer)

    cost,P,∇ℰ = SpectralDistances.sinkhorn_diff(pl,ql,p,q,C,λh; kwargs...)
    P,λh
end



# function alg1(X,Y,â,b;λ=100)
#     N = length(Y)
#     â = copy(â)
#     # fill!(â, 1/N)
#     ã = copy(â)
#     t = 0
#     for outer t = 1:10000
#         β = t/2
#         â .= (1-inv(β)).*â .+ inv(β).*ã
#         𝛂 = mean(1:N) do i
#             M = distmat_euclidean(X,Y[i])
#             a = SpectralDistances.sinkhorn2(M,â,b[i]; iters=500, λ=λ)[2]
#             @assert all(!isnan, a) "Got nan in inner sinkhorn alg 1"
#             a
#         end
#
#         ã .= â .* exp.(-β.*𝛂 .* 0.001)
#         ã ./= sum(ã)
#         sum(abs2,â-ã)
#         if sum(abs2,â-ã) < 1e-16
#             @info "Done at iter $t"
#             return â .= (1-inv(β)).*â .+ inv(β).*ã
#         end
#         â .= (1-inv(β)).*â .+ inv(β).*ã
#         # â ./= sum(â)
#     end
#     @show t
#     â
# end
#
#
#
# function alg2(X,Y,a,b;λ = 100,θ = 0.5)
#     N = length(Y)
#     a = copy(a)
#     ao = copy(a)
#     X = copy(X)
#     Xo = copy(X)
#     fill!(ao, 1/length(ao))
#     i = 0
#     for outer i = 1:500
#         a = alg1(X,Y,ao,b,λ=λ)
#         YT = mean(1:N) do i
#             M = distmat_euclidean(X,Y[i])
#             T,_ = SpectralDistances.sinkhorn2(M,a,b[i]; iters=500, λ=λ)
#             @assert all(!isnan, T) "Got nan in sinkhorn alg 2"
#             Y[i]*T'
#         end
#         X .= (1-θ).*X .+ θ.*(YT / Diagonal(a))
#         # @show mean(abs2, a-ao), mean(abs2, X-Xo)
#         mean(abs2, a-ao) < 1e-8 && mean(abs2, X-Xo) < 1e-8 && break
#         copyto!(ao,a)
#         copyto!(Xo,X)
#         ao ./= sum(ao)
#         θ *= 0.99
#     end
#     @show i
#     X,a
# end



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

Iterative swapping algorithm from "On the Computation of Wasserstein barycenters", Giovanni Puccetti1 et al.

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
                if dot(Xik1, ∑jni(X,i,σ,k₁)) + dot(Xik2, ∑jni(X,i,σ,k₂)) < dot(Xik2, ∑jni(X,i,σ,k₁)) + dot(Xik1, ∑jni(X,i,σ,k₂))
                    σᵢ′[k₁],σᵢ′[k₂] = σᵢ[k₂],σᵢ[k₁] # This line can cause σᵢ′ to not contain all indices 1:k
                    swaps += 1
                end
            end
        end
        σ = deepcopy(σ′) # Update assignment
        iter % printerval == 0 && @show iter, swaps
        swaps == 0 && (return σ)
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
