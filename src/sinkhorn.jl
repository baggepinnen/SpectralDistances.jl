"""
    γ, u, v = sinkhorn(C, a, b; β=1e-1, iters=1000)

The Sinkhorn algorithm. `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. See also [`IPOT`](@ref).
"""
function sinkhorn(C, a, b; β=1e-1, iters=1000)
    K = exp.(.-C ./ β)
    v = one.(b)
    u = a ./ (K * v)
    v = b ./ (K' * u)

    for iter = 1:iters
        u = a ./ (K * v)
        v = b ./ (K' * u)
    end
    u .* K .* v', u, v
end

"""
    γ, u, v = IPOT(C, a, b; β=1, iters=1000)

The Inexact Proximal point method for exact Optimal Transport problem (IPOT) (Sinkhorn-like) algorithm. `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. See also [`sinkhorn`](@ref). `β` does not have to go to 0 for this alg to return the optimal distance.

A Fast Proximal Point Method for Computing Exact Wasserstein Distance
Yujia Xie, Xiangfeng Wang, Ruijia Wang, Hongyuan Zha
https://arxiv.org/abs/1802.04307
"""
function IPOT(C, μ, ν; β=1, iters=1000)
    G = exp.(.- C ./ β)
    a = similar(μ)
    b = fill(eltype(ν)(1/length(ν)), length(ν))
    Γ = ones(eltype(ν), size(G)...)
    Q = similar(G)
    local a
    for iter = 1:iters
        Q .= G .* Γ
        mul!(a, Q, b)
        a .= μ ./ a
        mul!(b, Q', a)
        b .= ν ./ b
        Γ .= a .* Q .* b'
    end
    Γ, a, b
end


# function IPOT(C, μ, ν; β=1, iters=2)
#     G = exp.(.- C ./ β)
#     b = fill(1/length(ν), length(ν))
#     Γ = ones(size(G)...)
#     local a
#     for iter = 1:iters
#         Q = G .* Γ
#         a = μ ./ (Q * b)
#         b = ν ./ (Q' * a)
#         Γ = a .* Q .* b'
#     end
#     Γ, a, b
# end


function sinkhorn2(C, a, b; λ, iters=1000)
    K = exp.(.-C .* λ)
    K̃ = Diagonal(a) \ K
    u = one.(b)./length(b)
    uo = copy(u)
    for iter = 1:iters
        u .= 1 ./(K̃*(b./(K'uo)))
        # @show sum(abs2, u-uo)
        if sum(abs2, u-uo) < 1e-10
            # @info "Done at iteration $iter"
            break
        end
        copyto!(uo,u)
    end
    @assert all(!isnan, u) "Got nan entries in u"
    u .= max.(u, 1e-20)
    @assert all(>(0), u) "Got non-positive entries in u"
    v = b ./ ((K' * u) .+ 1e-20)
    if any(isnan, v)
        @show (K' * u)
        error("Got nan entries in v")
    end
    lu = log.(u)# .+ 1e-100)
    α = -lu./λ .+ sum(lu)/(λ*length(u))
    α .-= sum(α) # Normalize dual optimum to sum to zero
    Diagonal(u) * K * Diagonal(v), α
end


"""
    cost, barycenter, gradient = sinkhorn_diff(pl,ql, p, q::AbstractVector{T}, C, λ::AbstractVector; γ = 0.1, L = 32) where T


Returns the sinkhonr cost, the estimated barycenter and the gradient w.r.t. λ

This function is called from within [`barycentric_coordinates`](@ref). See help for this function regarding the other parameters.

Ref https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf

The difference in this algorithm compared to the paper is that they operate on histograms where the cost matric `C` is the same for all pairs of pᵢ and q. Here, there is a unique `C` for each pair.

#Arguments:
- `C`: Vector of cost matrices
- `λ`: barycentric coordinates
- `γ`: sinkhorn regularization parameter
- `L`: number of sinkhorn iterations
"""
function sinkhorn_diff(pl,ql, p, q::AbstractVector{T}, C, λ::AbstractVector; γ=1e-1, L=32) where T
    N,S = size(p)
    @assert length(λ) == S "Length of barycentric coordinates bust be same as number of anchors"
    @assert length(q) == N "Dimension of input measure must be same as dimension of anchor measures"
    @assert length(C) == S
    K = [exp.(.-C ./ γ) for C in C]
    b = [fill(1/N, N, S) for _ in 0:L+1]
    w = zeros(T,S)
    r = zeros(T,N,S)
    φ = [Matrix{T}(undef,N,S) for _ in 1:L]
    local P
    for l = 1:L
        for s in 1:S
            φ[l][:,s] = K[s]'* (p[:,s] ./ (K[s] * b[l][:,s]))
        end
        P = prod(φ[l].^λ', dims=2) |> vec
        b[l+1] .= P ./ φ[l]
    end

    # Since we are not working with histograms where the support of all bins remain the same, we must calculate the location of the barycenter and not only the weights. The locatoin is then used to get a new cost matrix between q's location and the projected barycenter location.
    Pl = barycenter(pl, λ)
    Cbc = distmat_euclidean(Pl, ql)

    Γ, a = sinkhorn(Cbc, P, q, β=γ, iters=2L) # We run a bit longer here to get a good value
    a = s1(a)
    ∇W = γ.*log.(a)
    g = ∇W .* P

    for l = L:-1:1
        w .+= log.(φ[l])'g
        for s in 1:S
            r[:,s] .= (-K[s]'*(K[s]*((λ[s].* g .- r[:,s]) ./ φ[l][:,s]) .* p[:,s] ./ (K[s]*b[l][:,s]).^2)) .* b[l][:,s]
        end
        g = sum(r, dims=2) |> vec

    end
    sum(Cbc .* Γ), Pl,w
end
