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
