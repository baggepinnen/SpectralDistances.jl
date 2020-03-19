"""
    γ, u, v = sinkhorn(C, a, b; β=1e-1, iters=1000)

The Sinkhorn algorithm. `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. See also [`IPOT`](@ref).
"""
function sinkhorn(C, a, b; β=1e-1, iters=1000)
    ϵ = eps()
    K = exp.(.-C ./ β)
    v = one.(b)
    local u
    for iter = 1:iters
        u = a ./ (K * v .+ ϵ)
        v = b ./ (K' * u .+ ϵ)
    end
    u .* K .* v', u, v
end

"""
    γ, u, v = sinkhorn_log(C, a, b; β=1e-1, iters=1000, tol=1e-8)

The Sinkhorn algorithm (log-stabilized). `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. See also [`IPOT`](@ref).

https://arxiv.org/pdf/1610.06519.pdf
"""
function sinkhorn_log(C, a, b; β=1e-1, τ=1e3, iters=1000, tol=1e-8, printerval = typemax(Int))

    alpha,beta = (similar(a) .= 0), (similar(b) .= 0)
    ϵ = eps()
    K = @. exp(-C / β)
    Γ = similar(K)
    u = similar(a) .= 1
    local v, iter
    iter = 0
    for iter = 1:iters
        v = b ./ (K' * u .+ ϵ)
        u = a ./ (K * v .+ ϵ)
        if maximum(abs, u) > τ || maximum(abs, v) > τ
            @. alpha += β * log(u)
            @. beta  += β * log(v)
            u .= one.(u)
            v .= one.(v)
            @. K = exp(-(C-alpha-beta') / β)
        end
        if any(!isfinite, u) || any(!isfinite, u)
            error("Got NaN in sinkhorn_log")
        end
        if iter % 10 == 0 || iter % printerval == 0
            @. Γ = exp(-(C-alpha-beta') / β + log(u) + log(v'))
            err = norm(vec(sum(Γ, dims=1)) - b)
            iter % printerval == 0 && @info "Iter: $iter, err: $err"
            if err < tol
               break
            end
        end

    end
    @. Γ = exp(-(C-alpha-beta') / β + log(u) + log(v'))

    @. u = -β*log(u) - alpha
    u .-= mean(u)
    @. v = -β*log(v) - beta
    v .-= mean(v)

    @assert isapprox(sum(u), 0, atol=1e-15) "sum(α) should be 0 but was = $(sum(u))" # Normalize dual optimum to sum to zero
    iter == iters && iters > printerval && @info "Maximum number of iterations reached. Final error: $(norm(vec(sum(Γ, dims=1)) - b))"

    Γ, u, v
end

"""
    γ, u, v = IPOT(C, a, b; β=1, iters=1000)

The Inexact Proximal point method for exact Optimal Transport problem (IPOT) (Sinkhorn-like) algorithm. `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. See also [`sinkhorn`](@ref). `β` does not have to go to 0 for this alg to return the optimal distance.

A Fast Proximal Point Method for Computing Exact Wasserstein Distance
Yujia Xie, Xiangfeng Wang, Ruijia Wang, Hongyuan Zha
https://arxiv.org/abs/1802.04307
"""
function IPOT(C, μ, ν; β=1, iters=1000, tol=1e-8, printerval = typemax(Int))
    ϵ = eps()
    G = exp.(.- C ./ β)
    a = similar(μ)
    b = fill(eltype(ν)(1/length(ν)), length(ν))
    Γ = ones(eltype(ν), size(G)...)
    Q = similar(G)
    local a
    for iter = 1:iters
        Q .= G .* Γ
        mul!(a, Q, b)
        a .= μ ./ (a .+ ϵ)
        mul!(b, Q', a)
        b .= ν ./ (b .+ ϵ)
        Γ .= a .* Q .* b'

        if iter % 10 == 0 || iter % printerval == 0
            err = norm(vec(sum(Γ, dims=2)) - μ)
            iter % printerval == 0 && @info "Iter: $iter, err: $err"
            if err < tol && iter > 3
               break
            end
        end

    end
    @. a = -β*log(a)
    a .-= mean(a)
    @. b = -β*log(b)
    b .-= mean(b)
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
function sinkhorn_diff(pl,ql, p, q::AbstractVector{T}, C, λ::AbstractVector; γ=0.2, L=32) where T
    N,S = size(p)
    w = zeros(T,S)
    any(isnan, λ) && return NaN,ql,w
    @assert length(λ) == S "Length of barycentric coordinates bust be same as number of anchors"
    @assert length(q) == N "Dimension of input measure must be same as dimension of anchor measures"
    @assert length(C) == S
    K = [exp.(.-C ./ γ) for C in C]
    b = [fill(1/N, N, S) for _ in 0:L+1]
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
    # @show norm(Pl-ql), sum(isnan,Pl), sum(isnan, λ)
    Cbc = distmat_euclidean(Pl, ql)

    Γ, a = sinkhorn(Cbc, P, q, β=γ, iters=2L) # We run a bit longer here to get a good value
    cost = sum(Cbc .* Γ)

    la = log.(a) # a should have geometric mean 1
    la .= la .- mean(la)
    ∇W = γ.*la
    g = ∇W .* P

    for l = L:-1:1
        w .+= log.(φ[l])'g
        for s in 1:S
            r[:,s] .= (-K[s]'*(K[s]*((λ[s].* g .- r[:,s]) ./ φ[l][:,s]) .* p[:,s] ./ (K[s]*b[l][:,s]).^2)) .* b[l][:,s]
        end
        g = sum(r, dims=2) |> vec
    end
    cost, Pl,w
end


# geomean(x) = prod(x)^(1/length(x))
# a = rand(3)
# la = log.(a)
# lag = la .- mean(la)
# geomean(exp.(lag))
