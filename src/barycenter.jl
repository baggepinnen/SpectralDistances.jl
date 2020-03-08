using SpectralDistances

"""
    barycenter(d::SinkhornRootDistance, models; normalize = true, kwargs...)

**Approximately** calculate the barycenter supported on the same number of atoms as the number of poles in the models.

#Arguments:
- `models`: vector of AR models
- `normalize`: make sure weights sum to 1
- `kwargs`: are sent to [`ISA`](@ref)
"""
function barycenter(d::SinkhornRootDistance,models; normalize=true, kwargs...)
    # bc = barycenter(EuclideanRootDistance(domain=domain(d), p=d.p, weight=residueweight),models).pc
    # X0 = [real(bc)'; imag(bc)']

    # TODO: would be faster to only run on half of the poles and then duplicate them in the end. Would also enforce conjugacy. Special fix needed for systems with real poles.
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    realpoles = any(iszero ∘ real, r)

    if !realpoles
        r = [r[1:end÷2] for r in r]
        w = [2w[1:end÷2] for w in w]
        X = [[real(r)'; imag(r)'] for r in r]
    end
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

    S = ISA(X; kwargs...)
    X̂ = [w2[i].*X[i][:,S[i]] for i in eachindex(X)]
    bc = sum(X̂)
    r1 = complex.(bc[1,:],bc[2,:])
    if realpoles
        bcr = ContinuousRoots(r1)
    else
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
    [mean(abs2, c1-c2) for c1 in eachcol(X), c2 in eachcol(Y)]
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
    @inbounds for j in eachindex(X)
        j == i && continue
        s .+= @views X[j][:,S[j][k]]
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
        X = deepcopy(X)
        for i in eachindex(X)
            X[i] .*= w[i] # This should work for both w[i] scalar and vector
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
                if dot(X[i][:,σᵢ[k₁]], ∑jni(X,i,σ,k₁)) + dot(X[i][:,σᵢ[k₂]], ∑jni(X,i,σ,k₂)) < dot(X[i][:,σᵢ[k₂]], ∑jni(X,i,σ,k₁)) + dot(X[i][:,σᵢ[k₁]], ∑jni(X,i,σ,k₂))
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
