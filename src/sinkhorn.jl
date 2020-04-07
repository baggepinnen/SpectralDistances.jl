"""
    γ, u, v = sinkhorn(C, a, b; β=1e-1, iters=1000)

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
    γ, u, v = sinkhorn_log(C, a, b; β=1e-1, iters=1000, tol=1e-8)

The Sinkhorn algorithm (log-stabilized). `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. See also [`sinkhorn_log!`](@ref) for a faster implementation operating in-place, and [`IPOT`](@ref) for a potentially more exact solution.

https://arxiv.org/pdf/1610.06519.pdf
"""
function sinkhorn_log(C, a, b; β=1e-1, τ=1e3, iters=1000, tol=1e-3, printerval = typemax(Int), kwargs...)

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
            if err < tol
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
    if ea > tol || eb > tol
        println("sinkhorn_log: iter: $iter Inaccurate solution - ea: $ea, eb: $eb, tol: $tol")
    end

    Γ, u, v
end


"""
Same as [`sinkhorn_log`](@ref) but operates in-place to save memory allocations. This function has higher performance than `sinkhorn_log`, but might not work as well with AD libraries.
"""
function sinkhorn_log!(C, a, b; β=1e-1, τ=1e3, iters=1000, tol=1e-8, printerval = typemax(Int), kwargs...)
    @assert sum(a) ≈ 1.0 "Input measure not normalized, expected sum(a) ≈ 1, but got $(sum(a))"
    @assert sum(b) ≈ 1.0 "Input measure not normalized, expected sum(b) ≈ 1, but got $(sum(b))"
    T = promote_type(eltype(a), eltype(b), eltype(C))
    alpha,beta = (zeros(T, size(a)) .= 0), (zeros(T, size(b)) .= 0)
    ϵ = eps()
    K = @. exp(-C / β)
    Γ = zeros(T, size(K))
    u = ones(T, size(a))
    v = ones(T, size(b))
    local v, iter
    iter = 0
    for outer iter = 1:iters
        mul!(v,K',u)
        v .= b ./ (v .+ ϵ)
        mul!(u,K,v)
        u .= a ./ (u .+ ϵ)

        if maximum(abs, u) > τ || maximum(abs, v) > τ
            @. alpha += β * log(u)
            @. beta  += β * log(v)
            u .= 1
            v .= 1
            @. K = exp(-(C-alpha-beta') / β)
        end
        if any(!isfinite, u) || any(!isfinite, u)
            error("Got NaN in sinkhorn_log")
        end
        if iter % 20 == 0 || iter % printerval == 0
            @. Γ = exp(-(C-alpha-beta') / β + log(u) + log(v'))
            err = +(ot_error(Γ, a, b)...)
            iter % printerval == 0 && @info "Iter: $iter, err: $err"
            if err < tol
               break
            end
        end

    end
    @. Γ = exp(-(C-alpha-beta') / β + log(u + ϵ) + log(v' + ϵ))
    @. u = -β*log(u + ϵ) - alpha
    u .-= mean(u)
    @. v = -β*log(v + ϵ) - beta
    v .-= mean(v)

    @assert isapprox(sum(u), 0, atol=1e-10length(u)) "sum(α) should be 0 but was = $(sum(u))" # Normalize dual optimum to sum to zero
    iter == iters && iters > printerval && @info "Maximum number of iterations reached. Final error: $(norm(vec(sum(Γ, dims=1)) - b))"

    ea, eb = ot_error(Γ, a, b)
    if ea > tol || eb > tol
        @error "sinkhorn_log: iter: $iter Inaccurate solution - ea: $ea, eb: $eb, tol: $tol"
    end

    Γ, u, v
end

"""
    γ, u, v = IPOT(C, a, b; β=1, iters=1000)

The Inexact Proximal point method for exact Optimal Transport problem (IPOT) (Sinkhorn-like) algorithm. `C` is the cost matrix and `a,b` are vectors that sum to one. Returns the optimal plan and the dual potentials. See also [`sinkhorn`](@ref). `β` does not have to go to 0 for this alg to return the optimal distance, in fact, if β is set too low, this alg will encounter numerical problems.

A Fast Proximal Point Method for Computing Exact Wasserstein Distance
Yujia Xie, Xiangfeng Wang, Ruijia Wang, Hongyuan Zha
https://arxiv.org/abs/1802.04307
"""
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
        Q .= G .* Γ
        mul!(a, Q, b)
        a .= μ ./ (a .+ ϵ)
        mul!(b, Q', a)
        b .= ν ./ (b .+ ϵ)
        Γ .= a .* Q .* b'

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

    @. a = -β*log(a + ϵ)
    a .-= mean(a)
    @. b = -β*log(b + ϵ)
    b .-= mean(b)
    eμ,eν = ot_error(Γ, μ, ν)
    if eμ > tol || eν > tol
        @error "IPOT: iter: $iter Inaccurate solution - eμ: $eμ, eν: $eν, tol: $tol"
    end
    Γ, a, b
end

ot_error(Γ, μ, ν) = norm(vec(sum(Γ, dims=2)) - μ)/norm(μ), norm(vec(sum(Γ, dims=1)) - ν)/norm(ν)


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



# """
#     cost, barycenter, gradient = sinkhorn_diff(pl,ql, p, q::AbstractVector{T}, C, λ::AbstractVector; β = 1, L = 32) where T
#
#
# Returns the sinkhorn cost, the estimated barycenter and the gradient w.r.t. λ
#
# This function is called from within [`barycentric_coordinates`](@ref). See help for this function regarding the other parameters.
#
# Ref https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf
#
# The difference in this algorithm compared to the paper is that they operate on histograms where the cost matric `C` is the same for all pairs of pᵢ and q. Here, there is a unique `C` for each pair.
#
# #Arguments:
# - `C`: Vector of cost matrices
# - `λ`: barycentric coordinates
# - `β`: sinkhorn regularization parameter
# - `L`: number of sinkhorn iterations
# """
# function sinkhorn_diff(pl,ql, p, q::AbstractVector{T}, C, λ::AbstractVector; β=1, L=32, solver=IPOT) where T
#     N = length(p)
#     S = length(p[1])
#     w = zeros(T,S)
#     any(isnan, λ) && return NaN,ql,w
#     @assert length(λ) == S "Length of barycentric coordinates bust be same as number of anchors"
#     @assert length(q) == N "Dimension of input measure must be same as dimension of anchor measures"
#     @assert length(C) == S
#     K = [exp.(.-C ./ β) for C in C]
#     b = [fill(1/N, N, S) for _ in 0:L+1]
#     r = zeros(T,N,S)
#     φ = [Matrix{T}(undef,N,S) for _ in 1:L]
#     local P
#     for l = 1:L
#         for s in 1:S
#             φ[l][:,s] = K[s]'* (p[s] ./ (K[s] * b[l][:,s]))
#         end
#         P = prod(φ[l].^λ', dims=2) |> vec
#         b[l+1] .= P ./ φ[l]
#     end
#
#     # Since we are not working with histograms where the support of all bins remain the same, we must calculate the location of the barycenter and not only the weights. The location is then used to get a new cost matrix between q's location and the projected barycenter location.
#     Pl = barycenter(pl, λ)
#     # @show norm(Pl-ql), sum(isnan,Pl), sum(isnan, λ)
#     Cbc = distmat_euclidean(Pl, ql)
#
#     Γ, a, bb = solver(Cbc, P, q, β=β, iters=4L) # We run a bit longer here to get a good value
#     cost = sum(Cbc .* Γ)
#
#     ∇W = a
#     # ∇W = bb
#     g = ∇W .* P
#
#     for l = L:-1:1
#         w .+= log.(φ[l])'g
#         for s in 1:S
#             r[:,s] .= (-K[s]'*((K[s]*((λ[s].* g .- r[:,s]) ./ φ[l][:,s])) .* p[s] ./ (K[s]*b[l][:,s]).^2)) .* b[l][:,s]
#         end
#         g = sum(r, dims=2) |> vec
#     end
#     cost, Pl,w
# end


# geomean(x) = prod(x)^(1/length(x))
# a = rand(3)
# la = log.(a)
# lag = la .- mean(la)
# geomean(exp.(lag))




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
