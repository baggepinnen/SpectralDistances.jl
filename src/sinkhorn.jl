abstract type SolverWorkspace end
struct SinkhornLogWorkspace{T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}} <: SolverWorkspace
    K::MT
    Γ::MT
    u::VT
    v::VT
    alpha::VT
    beta::VT
end

function SinkhornLogWorkspace(T,n,m)
    alpha,beta = zeros(T, n), zeros(T, m)
    K = Matrix{T}(undef, n, m)
    Γ = similar(K)
    u = ones(T, n)
    v = ones(T, m)
    SinkhornLogWorkspace(K,Γ,u,v,alpha,beta)
end

"""
    Γ, u, v = sinkhorn(C, a, b; β=1e-1, iters=1000)

The Sinkhorn algorithm. `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. This function is relatively slow, see also [`sinkhorn_log!`](@ref) [`IPOT`](@ref) and [`sinkhorn_log`](@ref) for faster algorithms.
"""
function sinkhorn(C, a, b; β=1e-1, iters=1000, kwargs...)
    ϵ = eps()
    K = exp.(.-C ./ β)
    v = one.(b)
    local u
    for iter = 1:iters
        u = a ./ (K * v .+ ϵ)
        v = b ./ (K' * u .+ ϵ)
    end
    Γ =  u .* K .* v'
    @. u = -β*log(u + ϵ)
    u .-= mean(u)
    @. v = -β*log(v + ϵ)
    v .-= mean(v)

    Γ, u, v
end

"""
    Γ, u, v = sinkhorn_log(C, a, b; β=1e-1, iters=1000, tol=1e-8)

The Sinkhorn algorithm (log-stabilized). `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. See also [`sinkhorn_log!`](@ref) for a faster implementation operating in-place, and [`IPOT`](@ref) for a potentially more exact solution.

When this function is being differentiated, warnings about inaccurate solutions are turned off. You may choose to manually asses the error in the constrains by `ea, eb = SpectralDistances.ot_error(Γ, a, b)`.

The IPOT algorithm: https://arxiv.org/pdf/1610.06519.pdf
"""
function sinkhorn_log(C, a, b; β=1e-1, τ=1e3, iters=1000, tol=1e-8, printerval = typemax(Int), kwargs...)

    @assert sum(a) ≈ 1 "Input measure not normalized, expected sum(a) ≈ 1, but got $(sum(a))"
    @assert sum(b) ≈ 1 "Input measure not normalized, expected sum(b) ≈ 1, but got $(sum(b))"
    T = real(promote_type(eltype(a), eltype(b), eltype(C)))
    alpha,beta = zeros(T, size(a)), zeros(T, size(b))
    ϵ = eps(T)
    K = @. exp(-C / β)
    Γ = zeros(T, size(K))
    u = ones(T, size(a))
    local v, iter
    iter = 0
    for outer iter = 1:iters
        v = b ./ (K' * u .+ ϵ)
        u = a ./ (K * v .+ ϵ)
        if maximum(abs, u) > τ || maximum(abs, v) > τ
            alpha += @. β * log(u)
            beta  += @. β * log(v)
            u = one.(u)
            v = one.(v)
            K = @. exp(-(C-alpha-beta') / β)
        end
        if any(!isfinite, u) || any(!isfinite, u)
            error("Got NaN in sinkhorn_log")
        end
        if iter % 10 == 0 || iter % printerval == 0
            Γ = @. exp(-(C-alpha-beta') / β + log(u) + log(v'))
            err = +(ot_error(Γ, a, b)...)
            iter % printerval == 0 && println("Iter: $iter, err: $err")
            if real(err) < tol
               break
            end
        end

    end
    Γ = @. exp(-(C-alpha-beta') / β + log(u) + log(v'))

    u = @. -β*log(u + ϵ) - alpha
    u = u .- mean(u)
    v = @. -β*log(v + ϵ) - beta
    v = v .- mean(v)

    @assert isapprox(sum(u), 0, atol=1e-10) "sum(α) should be 0 but was = $(sum(u))" # Normalize dual optimum to sum to zero
    iter == iters && iters > printerval && println("Maximum number of iterations reached. Final error: $(norm(vec(sum(Γ, dims=1)) - b))")

    ea, eb = ot_error(Γ, a, b)
    if (real(ea) > tol || real(eb) > tol) && !isderiving()
        println("sinkhorn_log: iter: $iter Inaccurate solution - ea: $ea, eb: $eb, tol: $tol")
    end

    Γ, u, v
end


"""
Same as [`sinkhorn_log`](@ref) but operates in-place to save memory allocations. This function has higher performance than `sinkhorn_log`, but might not work as well with AD libraries.

This function can be made completely allocation free with the interface
    sinkhorn_log(w::SinkhornLogWorkspace{T}, C, a, b; kwargs...)

The `sinkhorn_log!` solver also accepts a keyword argument `check_interval = 20` that determines how often the convergence criteria is checked. If `β` is large, the algorithm might converge very fast and you can save some iterations by reducing the check interval. If `β` is small and the algorithm requires many iterations, a larger number saves you from computing the check too often.

The workspace `w` is created linke this: `w = SinkhornLogWorkspace(FloatType, length(a), length(b))`
"""
function sinkhorn_log!(C, a, b; kwargs...)
    T = promote_type(eltype(a), eltype(b), eltype(C))
    # if T <: Double64
    #     T = Float64
    #     C,a,b = Float64.(C), Float64.(a), Float64.(b)
    # end
    w = SinkhornLogWorkspace(T, length(a), length(b))
    sinkhorn_log!(w, C, a, b; kwargs...)
end


# This is just the same as the one above, but with @avx so it only supports simple types
function sinkhorn_log!(w::SinkhornLogWorkspace{T}, C, a, b; β=1e-1, τ=1e3, iters=1000, tol=1e-8, printerval = typemax(Int),
    check_interval = 20, kwargs...) where T
    @assert sum(a) ≈ 1.0 "Input measure not normalized, expected sum(a) ≈ 1, but got $(sum(a))"
    @assert sum(b) ≈ 1.0 "Input measure not normalized, expected sum(b) ≈ 1, but got $(sum(b))"
    ϵ = eps()
    K, Γ, u, v, alpha, beta = w.K, w.Γ, w.u, w.v, w.alpha, w.beta
    @avx @. K = exp(-C / β)
    u .= 1
    v .= 1
    alpha .= 0
    beta  .= 0
    local v, iter
    iter = 0
    for outer iter = 1:iters
        mul!(v,K',u)
        @avx v .= b ./ (v .+ ϵ) # Some tests fail due to https://github.com/chriselrod/LoopVectorization.jl/issues/103
        mul!(u,K,v)
        @avx u .= a ./ (u .+ ϵ)

        if maximum(abs, u) > τ || maximum(abs, v) > τ
            @avx @. alpha += β * log(u)
            @avx @. beta  += β * log(v)
            u .= 1
            v .= 1
            @avx @. K = exp(-(C-alpha-beta') / β)
        end
        if any(!isfinite, u) || any(!isfinite, u)
            error("Got NaN in sinkhorn_log")
        end
        # @show lowerbound(a,b,u,v,alpha,beta,β)

        if iter % check_interval == 0 || iter % printerval == 0
            @avx @. Γ = exp(-(C-alpha-beta') / β + log(u) + log(v'))
            err = +(ot_error(Γ, a, b)...)
            iter % printerval == 0 && @info "Iter: $iter, err: $err"
            if err < tol
               break
            end
        end

    end
    @avx @. Γ = exp(-(C-alpha-beta') / β + log(u + ϵ) + log(v' + ϵ))
    @avx @. u = -β*log(u + ϵ) - alpha
    @avx u .-= mean(u)
    @avx @. v = -β*log(v + ϵ) - beta
    @avx v .-= mean(v)

    @assert isapprox(sum(u), 0, atol=sqrt(eps(T))*length(u)) "sum(α) should be 0 but was = $(sum(u))" # Normalize dual optimum to sum to zero
    iter == iters && iters > printerval && @info "Maximum number of iterations reached. Final error: $(norm(vec(sum(Γ, dims=1)) - b))"

    ea, eb = ot_error(Γ, a, b)
    if ea > tol || eb > tol
        @error "sinkhorn_log: iter: $iter Inaccurate solution - ea: $ea, eb: $eb, tol: $tol"
    end

    Γ, u, v
end

# function lowerbound(a,b,u,v,alpha,beta,β)
#     ϵ = 1e-16
#     u = @.  -β*log(u + ϵ) - alpha
#     # u .-= mean(u)
#     v = @.  -β*log(v + ϵ) - beta
#     # v .-= mean(v)
#     u'a + v'b
# end

"""
    Γ, u, v = IPOT(C, a, b; β=1, iters=1000)

The Inexact Proximal point method for exact Optimal Transport problem (IPOT) (Sinkhorn-like) algorithm. `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. See also [`sinkhorn`](@ref). `β` does not have to go to 0 for this alg to return the optimal distance, in fact, if β is set too low, this alg will encounter numerical problems.

A Fast Proximal Point Method for Computing Exact Wasserstein Distance
Yujia Xie, Xiangfeng Wang, Ruijia Wang, Hongyuan Zha
https://arxiv.org/abs/1802.04307
"""
IPOT
function IPOT(C, μ, ν; β=1, iters=10000, tol=1e-8, printerval = typemax(Int), kwargs...)
    @assert sum(μ) ≈ 1 "Input measure not normalized - sum(μ) = $(sum(μ))"
    @assert sum(ν) ≈ 1 "Input measure not normalized - sum(ν) = $(sum(ν))"
    T = promote_type(eltype(μ), eltype(ν), eltype(C))
    ϵ = eps(T)
    G = exp.(.- C ./ β)
    a = zeros(T, size(μ))
    b = fill(T(1/length(ν)), length(ν))
    Γ = ones(T, size(G)...)
    Q = zeros(T, size(G))
    local a
    iter = 0
    for outer iter = 1:iters
        @avx Q .= G .* Γ
        mul!(a, Q, b)
        @avx a .= μ ./ (a .+ ϵ)
        mul!(b, Q', a)
        @avx b .= ν ./ (b .+ ϵ)
        @avx Γ .= a .* Q .* b'

        if iter % 20 == 0 || iter % printerval == 0
            err = +(ot_error(Γ, μ, ν)...)
            iter % printerval == 0 && @info "Iter: $iter, err: $err"
            if err < tol && iter > 3
                printerval < iters && @info "IPOT: iter: $iter success - $err < $tol"
               break
            end
        end

    end
    printerval < iters && iter == iters && @info "IPOT: maximum number of iterations reached: $iters"

    @avx @. a = -β*log(a + ϵ)
    @avx a .-= mean(a)
    @avx @. b = -β*log(b + ϵ)
    @avx b .-= mean(b)
    eμ,eν = ot_error(Γ, μ, ν)
    if eμ > tol || eν > tol
        @error "IPOT: iter: $iter Inaccurate solution - eμ: $eμ, eν: $eν, tol: $tol"
    end
    Γ, a, b
end

@fastmath @inbounds @inline function ot_error(Γ, μ, ν)
    T = eltype(Γ)
    s1 = n1 = s2 = n2 = zero(T)
    for i = 1:length(μ)
        sg = zero(T)
        for j = 1:length(ν)
            sg += Γ[i,j]
        end
        s1 += abs2(μ[i] - sg)
        n1 += μ[i]
    end
    for i = 1:length(ν)
        sg = zero(T)
        for j = 1:length(μ)
            sg += Γ[j,i]
        end
        s2 += abs2(ν[i] - sg)
        n2 += ν[i]
    end

    s1 / sqrt(n1), s2 / sqrt(n2)
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


# function sinkhorn_cost(pl,ql, p, q::AbstractVector, C, λ::AbstractVector{T}; β=1, L=32, solver=IPOT, kwargs...) where T
#     N = length(p[1])
#     S = length(p)
#     any(isnan, λ) && return NaN
#     @assert length(λ) == S "Length of barycentric coordinates bust be same as number of anchors"
#     @assert length(q) == N "Dimension of input measure must be same as dimension of anchor measures"
#     @assert length(C) == S
#     K = [exp.(.-C ./ β) for C in C]
#     b = fill(T(1/N), N, S)
#     r = zeros(T,N,S)
#     φ = Matrix{T}(undef,N,S)
#     local P
#     for l = 1:L
#         for s in 1:S
#             φ[:,s] .= K[s]'* (p[s] ./ (K[s] * b[:,s]))
#         end
#         P = prod(φ.^λ', dims=2) |> vec
#         b .= P ./ φ
#     end
#
#     # Since we are not working with histograms where the support of all bins remain the same, we must calculate the location of the barycenter and not only the weights. The location is then used to get a new cost matrix between q's location and the projected barycenter location.
#     Pl = barycenter(pl, λ; kwargs...)
#     # @show norm(Pl-ql), sum(isnan,Pl), sum(isnan, λ)
#     Cbc = distmat_euclidean(Pl, ql)
#
#     Γ, a = solver(Cbc, P, q; β=β, iters=4L) # We run a bit longer here to get a good value
#     cost = sum(Cbc .* Γ)
# end


# function sinkhorn_cost(C, p, q::AbstractVector, λ::AbstractVector{T}; β=1, solver=IPOT, kwargs...) where T
#    c = sum(eachindex(λ)) do i
#        Ci = C[i]
#        λ[i] * sqrt(sum(solver(Ci, p[i], q; β=β, kwargs...)[1] .* Ci))
#    end
#    c
# end


function sinkhorn_cost(pl,ql, p, q::AbstractVector, λ::AbstractVector{T}; β=0.01, solver=sinkhorn_log!, kwargs...) where T

    bc,bcp = barycenter(pl,p,λ; kwargs...)
    # @show typeof(bc), typeof(ql)
    M = distmat_euclidean(bc,ql)
    Γ = solver(M,bcp,q; β=β, kwargs...)[1]
    sqrt(sum(M.*Γ))
end


"""
    Γ, u, v = sinkhorn_unbalanced(C, a, b, divergence; β=1e-1, iters=1000, tol=1e-8)

The Unbalanced Sinkhorn algorithm (log-stabilized). `C` is the cost matrix and `a,b` are vectors that *are not required to sum to one*.

Ref: "Sinkhorn Divergences for Unbalanced Optimal Transport" https://arxiv.org/abs/1910.12958
Makes use of [UnbalancedOptimalTransport.jl](https://github.com/ericphanson/UnbalancedOptimalTransport.jl)
"""
function sinkhorn_unbalanced(
    C,
    a,
    b,
    divergence;
    β = 1e-1,
    iters = 1000,
    tol = 1e-8,
    printerval = typemax(Int),
    kwargs...,
)
    a = UnbalancedOptimalTransport.DiscreteMeasure(a)
    b = UnbalancedOptimalTransport.DiscreteMeasure(b)
    Γ = optimal_coupling!(divergence, C, a, b, β; max_iters = iters, tol = tol, warn = true)
    return Γ, a.dual_potential, b.dual_potential
end

unbalanced_solver_closure(divergence::UnbalancedOptimalTransport.AbstractDivergence) =
    (C,a,b; kwargs...)->sinkhorn_unbalanced(C,a,b,divergence;kwargs...)

function unbalanced_solver_closure(distance::AbstractDistance, solver, kwargs...)
    (C,a,b; kwargs...)->sinkhorn_unbalanced(C,a,b,distance.divergence;kwargs...), kwargs
end

function unbalanced_solver_closure(distance::AbstractDistance)
    (C,a,b; kwargs...)->sinkhorn_unbalanced(C,a,b,distance.divergence;kwargs...)
end


function transport_plan(dist,m1::AbstractModel,m2::AbstractModel; kwargs...)
    rf(a) = roots(domain(dist), a)
    e1 = rf(m1)
    e2 = rf(m2)
    transport_plan(dist,e1,e2; kwargs...)
end

function transport_plan(dist,e1,e2; kwargs...)
    D = distmat_euclidean(e1, e2, dist.p)
    w1 = dist.weight(e1)
    w2 = dist.weight(e2)
    solver = dist.divergence === nothing ? sinkhorn_log! : unbalanced_solver_closure(dist)
    Γ = solver(D, w1, w2; β = 0.01, kwargs...)[1]
end







struct SCWorkspace{T}
    #K::TK
    V::Array{T,2}
    U::Array{T,2}
    S::Array{T,2}
    S2::Array{T,2}
    alpha::Array{T,2}
    beta::Array{T,2}
    xi1::Array{T,2}
    xi2::Array{T,2}
end

"""
    SCWorkspace(A, B, β)

Workspace object for [`sinkhorn_convolutional`](@ref). Manually construct this in order to save allocations between consequtive calls to the solver.

# Arguments:
- `A`: The first matrix
- `B`: The second matrix
- `β`: the regularization parameter
"""
function SCWorkspace(A, B, β)
    m, n = size(A)
    T    = eltype(eltype(A))
    β    = T(β)

    U   = ones(T, size(B))
    V   = ones(T, size(A))
    S   = Matrix{T}(undef, m, n)
    S2  = Matrix{T}(undef, size(B))

    xi1 = Matrix{T}(undef, m, m)
    xi2 = Matrix{T}(undef, n, n)

    alpha,beta = zeros(T, size(A)), zeros(T, size(B))

    SCWorkspace(V, U, S, S2, alpha, beta, xi1, xi2)
end

"""
    sinkhorn_convolutional(w::SCWorkspace{T}, A::AbstractMatrix, B::AbstractMatrix; β = 0.01, τ = 1 / eps(T), iters = 1000, tol = 1.0e-6, ϵ = eps(T) ^ 2, verbose = false, initUV = true) where T

Calculate the entropically regularizaed Sinkhorn distance between two matrices where the ground cost is squared euclidean. This function uses an efficient convolutional algorithm and is much more efficient than the corresponding [`sinkhorn_log!`](@ref) in this special case.

This function spends most of its time doing matrix multiplications. You may want to tune `BLAS.set_num_threads` in order to maximize performance. 1-2 threads is often best, especially if you thread outside of calls to this function.

# Arguments:
- `w`: workspace object
- `A`: The first matrix
- `B`: The second matrix
- `β`: regularization parameter. To get smooth distance profiles ([`distance_profile`](@ref)), a slightly higher β than for barycenters is recommended.  β around 0.01 should do fine.
- `τ`: stabilization parameter
- `iters`: maximum number of iterations
- `tol`: tolerance (change in oen of the dual variables)
- `ϵ`: other stabilization parameter
- `verbose`: print stuff?
- `initUV`: if true, initializes dual variables in `w` to one. It can be useful to set this to false if you have a good initial guess (warm start). Set the corresponding `w.U, w.V` to provide the initial guess.
"""
function sinkhorn_convolutional(
    w::SCWorkspace{T},
    A::AbstractMatrix,
    B::AbstractMatrix;
    β = 0.01,
    τ = 1/eps(T),
    iters = 1000,
    tol = 1e-6,
    ϵ = eps(T)^2,
    verbose = false,
    initUV = true,
) where {T}

    # @fastmath sum(A) ≈ sum(B) || @warn "Input matrices do not appear to have the same mass (sum)"
    # V, U, S, S2, xi1, xi2, alpha, beta = w.V, w.U, w.S, w.S2, w.xi1, w.xi2, w.alpha, w.beta
    V, U, S, S2, xi1, xi2 = w.V, w.U, w.S, w.S2, w.xi1, w.xi2
    size(A) == size(V) && size(U) == size(B) || throw(ArgumentError("Sizes of input matrices do not match sizes of cache in the workspace. This can happen if a distance object was previously used on inputs with other dimensions."))
    if initUV
        U .= 1
        V .= 1
    end
    alpha = zero(T)
    beta  = zero(T)
    iter = 0
    err = one(T)
    _initialize_conv_op!(xi1, xi2, β)


    while err > tol && iter < iters
        iter = iter + 1
        copyto!(S2, U)
        # K(V, U)
        mul!(S,U,xi2)
        mul!(V,xi1,S) # TODO: this is by far the most expensive operation
        @avx @. V = A / max(V, ϵ)
        # K(U, V)
        mul!(S,V,xi2)
        mul!(U,xi1,S)
        @avx @. U = B / max(U, ϵ)
        mU, mV = maximum(abs, U), maximum(abs, V)
        if mU > τ || mV > τ
            alpha += log(mU)
            beta  += log(mV)
            U ./= mU
            V ./= mV
            _initialize_conv_op!(xi1, xi2, alpha, beta, β)
        end

        if iter % 10 == 1
            @fastmath err = sum(abs(S2 - U) for (S2, U) in zip(S2, U))
            # verbose && @info "Sinkhorn conv: iter = $iter, error = $err"
        end
    end
    @avx @. V = log(V + ϵ) + alpha
    @avx @. U = log(U + ϵ) + beta
    c = β*(dot(A, V) + dot(B, U))
    V .-= mean(V)
    U .-= mean(U)
    c, V, U

end

#
# function sinkhorn_convolutional_diff(
#     A::AbstractMatrix,
#     B::AbstractMatrix;
#     β = 0.01,
#     T = promote_type(eltype(A),eltype(B)),
#     τ = 1/eps(T),
#     iters = 10,
#     tol = 1e-3,
#     ϵ = eps(T)^2,
#     verbose = false,
#     initUV = true,
# )
#
#     A = A ./ sum(A)
#     B = B ./ sum(B)
#
#     U = ones(size(B))
#     V = ones(size(A))
#
#     alpha = zero(T)
#     beta  = zero(T)
#     iter = 0
#     err = one(T)
#     m,n = size(A)
#     xi1 = Matrix{T}(undef, m, m)
#     xi2 = Matrix{T}(undef, n, n)
#     _initialize_conv_op!(xi1, xi2, β)
#
#
#     while err > tol && iter < iters
#         iter = iter + 1
#         S2 = U
#         # K(V, U)
#         V = xi1*U*xi2
#         @. V = A / max(V, ϵ)
#         # K(U, V)
#         U = xi1*V*xi2
#         @. U = B / max(U, ϵ)
#         mU, mV = maximum(abs, U), maximum(abs, V)
#         if mU > τ || mV > τ
#             alpha += log(mU)
#             beta  += log(mV)
#             U ./= mU
#             V ./= mV
#             _initialize_conv_op!(xi1, xi2, alpha, beta, β)
#         end
#
#         if iter % 10 == 1
#             @fastmath err = sum(abs(S2 - U) for (S2, U) in zip(S2, U))
#             verbose && @info "Sinkhorn conv: iter = $iter, error = $err"
#         end
#     end
#     @. V = log(V + ϵ) + alpha
#     @. U = log(U + ϵ) + beta
#     β*(dot(A, V) + dot(B, U))
#
# end

"""
    sinkhorn_convolutional(A::AbstractMatrix, B::AbstractMatrix; β = 0.001, kwargs...)
"""
function sinkhorn_convolutional(
    A::AbstractMatrix,
    B::AbstractMatrix;
    β = 0.001,
    kwargs...,
)
    w = SCWorkspace(A, B, β)
    sinkhorn_convolutional(w, A, B; β=β, kwargs...)
end


function _initialize_conv_op!(xi1::AbstractMatrix{T}, xi2::AbstractMatrix{T}, β::Real) where T
    m,n = size(xi1,2), size(xi2, 1)
    t = LinRange(zero(T), one(T), m)
    for i = 1:m, j = 1:m
        xi1[i,j] = exp(-(t[i] - t[j])^2 / β)
    end
    t = LinRange(zero(T), one(T), n)
     for i = 1:n, j = 1:n
        xi2[i,j] = exp(-(t[i] - t[j])^2 / β)
    end
end

function _initialize_conv_op!(xi1::AbstractMatrix{T}, xi2::AbstractMatrix{T}, alpha, beta, β::Real) where T
    m,n = size(xi1,2), size(xi2, 1)
    t    = LinRange(zero(T), one(T), m)
    alpha, beta = alpha, beta
    @avx for i = 1:m, j = 1:m
        xi1[i,j]  = exp(-(t[i] - t[j])^2 / β + alpha + beta)
    end
    # t    = LinRange(zero(T), one(T), n)
    # @avx for i = 1:n, j = 1:n
    #     xi2[i,j]  = exp(-((t[i] - t[j])^2) / β)
    # end
end

"""
    ConvOptimalTransportDistance <: AbstractDistance

Distance between matrices caluclated using [`sinkhorn_convolutional`](@ref). This type automatically creates a workspace object that is reused between invocations. Functions that internally performs threaded computations copy this object to each thread, but if manual threading is performed, this must be handled manually, e.g.:
```julia
using ThreadTools
dists = [deepcopy(dist) for _ in 1:Threads.nthreads()]
D = tmap(eachindex(X)) do i
    evaluate(dists[Threads.threadid()], Q, X[i]; kwargs...)
end
```

It's important to tune the two parameters below, see the docstring for [`sinkhorn_convolutional`](@ref) for more help.

- To get sharp barycenters, a smaller β around 0.001 is recommended.
- To get smooth distance profiles ([`distance_profile`](@ref)), a slightly higher β than for barycenters is recommended.  β around 0.01-0.05 should do fine.

- `β = 0.001`
- `dynamic_floor = -10.0`
- `invariant_axis::Int = 0` If this is set to 1 or 2, the distance will be approximately invariant to translations along the invariant axis. As an example, to be invariant to a spectrogram being shifted slightly in time, set `invariant_axis = 2`.
"""
ConvOptimalTransportDistance
Base.@kwdef mutable struct ConvOptimalTransportDistance{T} <: AbstractDistance
    β::T = 0.001
    dynamic_floor::T = NaN
    invariant_axis::Int = 0
    workspace::Union{Nothing, SCWorkspace{T}} = nothing
end

function evaluate(d::ConvOptimalTransportDistance, w1::DSP.Periodograms.TFR, w2::DSP.Periodograms.TFR; kwargs...)
    A  = s1(normalize_spectrogram(w1, d.dynamic_floor))
    B  = s1(normalize_spectrogram(w2, d.dynamic_floor))
    sinkhorn_convolutional( A, B; β = d.β, kwargs...)[1]
end

function evaluate(d::ConvOptimalTransportDistance, A::AbstractMatrix, B::AbstractMatrix{T}; kwargs...) where T
    if d.workspace === nothing
        d.workspace = SCWorkspace(A,B,d.β)
    end
    c,V,U = sinkhorn_convolutional(d.workspace::SCWorkspace{T}, A, B; β = d.β, kwargs...)
    if d.invariant_axis != 0
        isderiving() && error("Can not differentiate a distance with an invariant axis (I haven't bothered figure out the adjoint).")
        ia = d.invariant_axis
        sum_axis = ia == 1 ? 2 : 1
        v,u = vec(sum(A, dims=sum_axis)), vec(sum(B, dims=sum_axis))
        # v ./= sum(v); u ./= sum(u) # this should already hold
        # Γ = discrete_grid_transportplan(v, u; inplace=false)
        # invariant_cost = sum(Γ[i,j]*abs2((i-j)/2) for i = 1:size(Γ,1), j = 1:size(Γ,2))
        invariant_cost = discrete_grid_transportcost(v, u; inplace=true)
        return c - invariant_cost / size(A,ia)
    end
    return c
end

"""
    normalize_spectrogram(S, dynamic_floor = default_dynamic_floor(S))
"""
function normalize_spectrogram(x::AbstractArray{T}, dynamic_floor = default_dynamic_floor(x)) where T
    isnan(dynamic_floor) && (dynamic_floor = default_dynamic_floor(x))
    @avx max.(log.(x .+ eps(T)), dynamic_floor) .- dynamic_floor
end
function normalize_spectrogram!(o, x::AbstractArray{T}, dynamic_floor = default_dynamic_floor(x)) where T
    isnan(dynamic_floor) && (dynamic_floor = default_dynamic_floor(x))
    @avx o .= max.(log.(x .+ eps(T)), dynamic_floor) .- dynamic_floor
end

normalize_spectrogram(x::Periodograms.TFR, args...) = normalize_spectrogram(power(x), args...)
normalize_spectrogram!(o, x::Periodograms.TFR, args...) = normalize_spectrogram!(o, power(x), args...)

default_dynamic_floor(x::AbstractArray{T}) where T = max(log(quantile(vec(x), 0.5)), T(-100))
default_dynamic_floor(x::Periodograms.TFR) = default_dynamic_floor(power(x))
default_dynamic_floor(models::Vector{<:Periodograms.TFR}) = mean(default_dynamic_floor(m) for m in models)
