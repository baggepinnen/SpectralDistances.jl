import JuMP: Model, @objective, @variable, @constraint, termination_status, dual, value

"""
    ot_jump(D, P1, P2)

Solve the optimal transport problem using JuMP. This function is only available if `using JuMP, GLPK`.

#Arguments:
- `D`: Distance matrix
- `P1`: Weight vector 1
- `P2`: Weight vector 2
"""
function ot_jump(D, P1, P2; kwargs...)
    n = length(P1)
    @assert all(isfinite, P1) "Got nonfinite P1"
    @assert all(isfinite, P2) "Got nonfinite P2"
    @assert sum(P1) ≈ 1 "Sum(P1) ≠ 1: $(sum(P1))"
    @assert sum(P2) ≈ 1 "Sum(P2) ≠ 1: $(sum(P2))"

    model = Model(GLPK.Optimizer)
    @variable(model, γ[1:n^2])
    @objective(model, Min, γ'vec(D))
    Γ = reshape(γ,n,n)
    con1 = JuMP.@constraint(model, con1,  sum(Γ, dims=1)[:] .== P2)
    con2 = @constraint(model, con2,  sum(Γ, dims=2)[:] .== P1)
    con3 = @constraint(model, con3,  γ .>= 0)
    JuMP.optimize!(model)
    if Int(termination_status(model)) != 1
        @error Int(termination_status(model))
    end
    α, β = -dual.(con2)[:], -dual.(con1)[:]
    α .-= mean(α)
    β .-= mean(β)
    reshape(value.(γ),n,n), α, β #, -dual.(con3)[:]
end
