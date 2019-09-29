function sinkhorn(C, a, b; β=1e-1, iters=2)
    K = exp.(-C / β)
    v = one.(b)
    u = a ./ (K * v)
    v = b ./ (K' * u)

    for iter = 1:iters
        u = a ./ (K * v)
        v = b ./ (K' * u)
    end
    u .* K .* v', u, v
end

function IPOT(C, μ, ν; β=1, iters=1000)
    G = exp.(.- C ./ β)
    a = similar(μ)
    b = fill(1/length(ν), length(ν))
    Γ = ones(size(G)...)
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
