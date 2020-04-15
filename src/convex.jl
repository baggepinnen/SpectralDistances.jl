import .Convex
using .GLPK

"""
    ot_convex(D, P1, P2)

Solve the optimal transport problem using Convex.jl. This function is only available if `using Convex.jl, GLPK`.

# Arguments:
- `D`: Distance matrix
- `P1`: Weight vector 1
- `P2`: Weight vector 2
"""
function ot_convex(D::AbstractMatrix{T}, P1, P2; kwargs...) where T
    n = length(P1)
    @assert all(isfinite, P1) "Got nonfinite P1"
    @assert all(isfinite, P2) "Got nonfinite P2"
    @assert sum(P1) ≈ 1 "Sum(P1) ≠ 1: $(sum(P1))"
    @assert sum(P2) ≈ 1 "Sum(P2) ≠ 1: $(sum(P2))"
    γ = Convex.Variable(n^2)
    Γ = reshape(γ,n,n)
    con1 = sum(Γ, dims=1) == P2'
    con2 = sum(Γ, dims=2) == P1
    problem = Convex.minimize(γ'vec(D), γ >= 0, con1, con2)
    Convex.solve!(problem, GLPK.Optimizer)
    if Int(problem.status) != 1
        @error problem.status
    end
    α, β = -con2.dual[:], -con1.dual[:]
    α .-= mean(α)
    β .-= mean(β)
    reshape(γ.value,n,n), α, β
end
