function ngradient(f, xs::AbstractArray...; δ = sqrt(eps()))
    grads = zero.(xs)
    xs = copy.(xs)

    for (x, Δ) in zip(xs, grads), i in eachindex(x)
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
    end
    return grads[1]
end

function njacobian(f,x)
    y  = f(x)
    k  = length(y)
    n  = length(x)
    J  = Matrix{eltype(y)}(undef,k,n)
    for i = 1:k
        g = ngradient(x->f(x)[i], x)
        J[i,:] .= g
    end
    J
end

function nhessian(f, a)
    H = njacobian(a->ngradient(f,a), a)
end

curvature(dist, a) = Symmetric(nhessian(b->dist(a,b), a))

@inline isderiving() = false

@adjoint isderiving() = true, _ -> nothing

# ZygoteRules.@adjoint sortperm(args...; kwargs...) = sortperm(args...; kwargs...), x->nothing

ZygoteRules.@adjoint function ContinuousRoots(r)
    p = sortperm(r; by=imageigsortby)
    ContinuousRoots(r), x->(x[invperm(p)],)
end
ZygoteRules.@adjoint function ContinuousRoots(r::DiscreteRoots)
    _,f = Zygote.pullback(r->log.(r), r.r)
    ContinuousRoots(r), f
end
ZygoteRules.@adjoint function DiscreteRoots(r)
    p = sortperm(r; by=angle)
    DiscreteRoots(r), x->(x[invperm(p)],)
end

@adjoint function sort(x::AbstractArray; by=identity) # can be removed when https://github.com/FluxML/Zygote.jl/pull/586 is merged
  p = sortperm(x; by=by)

  return x[p], x̄ -> (x̄[invperm(p)],)
end

# ZygoteRules.@adjoint SArray{T1,T2,T3,T4}(r) where {T1,T2,T3,T4} = SArray{T1,T2,T3,T4}(r), x->(SArray{T1,T2,T3,T4}(x),) # got "internal error" but julia did not terminate
# ZygoteRules.@adjoint SArray{T1,T2,T3,T4}(r) where {T1,T2,T3,T4} = SArray{T1,T2,T3,T4}(r), x->(x,) # got segfault after defining this
# ZygoteRules.@adjoint SArray{T1,T2,T3,T4}(r::Vector) where {T1,T2,T3,T4} = SArray{T1,T2,T3,T4}(r), x->(x,) # does not appear to do anything
# ZygoteRules.@adjoint (T::Type{<:SArray})(x::Number...) = T(x...), y->(nothing, y...)


ZygoteRules.@adjoint function getARregressor(y,na)
    getARregressor((y),na),  function (Δ)
        d = zero(y)
        Δ[1] === nothing || (d[na+1:end] .= Δ[1])

        Δ[2] === nothing || for j in 1:size(Δ[2], 2)
            for i in 1:size(Δ[2], 1)
                di = na+1-j + i-1
                d[di] += Δ[2][i,j]
            end
        end
        (d,nothing)
    end
end

ZygoteRules.@adjoint function getARXregressor(y, u, na::Int, nb)
    @assert nb <= na # This is not a fundamental requirement, but this adjoint does not support it yet.
    getARXregressor((y),(u),na,nb),  function (Δ)
    dy = zero(y)
    du = zero(u)
    Δ[1] === nothing || (dy[na+1:end] .= Δ[1])
    # du[na+1:end] .= Δ[1] #src
    Δ[2] === nothing || for j in 1:size(Δ[2], 2)
        for i in 1:size(Δ[2], 1)
            if j <= na
                dyi = na+1-j + i-1
                dy[dyi] += Δ[2][i,j]
            else
                ju = j -na
                dui = na+1-ju + i-1
                du[dui] += Δ[2][i,j]
            end
        end
    end
    (dy,du,nothing,nothing)
end
end

PolynomialRoots.roots(p) = eigsort(eigvals(companion(p)))

ZygoteRules.@adjoint function polyconv(a,b)
    c = polyconv(a,b)
    function polyconv_back(Δ)
        da = zeros(eltype(c), length(c), length(a))
        for i = 0:length(b)-1
            da[diagind(da,-i)] .= b[i+1]
        end
        d1 = da'Δ

        db = zeros(eltype(c), length(c), length(b))
        for i = 0:length(a)-1
            db[diagind(db,-i)] .= a[i+1]
        end
        d2 = db'Δ

        d1,d2
    end
    c, polyconv_back
end

ZygoteRules.@adjoint function polyvec(w)
    pv = polyvec(w)
    function polyvec_back(Δ)
        (-Δ[2:end],)
    end
    pv, polyvec_back
end

# using ForwardDiff
# PolynomialRoots.roots(p::Vector{<: Complex{<:ForwardDiff.Dual}}) = eigvals(companion(p))



function companion(r)
    A = zeros(eltype(r), length(r)-1, length(r)-1)
    A[diagind(A,-1)] .= 1
    A[:,end] .= .- r[1:end-1]
    A
end

function rootadjoint(eV, perm)
    function rootadjoint_inner(Δ)
        iperm = invperm(perm)
        eltype(Δ) == Nothing && return (nothing,)
        e,V = eV
        # V = V[:,perm]
        # V = transpose(V)
        # V = conj.(V)
        V = V'
        d = [-(inv(V)*Diagonal(Δ[iperm])*V)[:,end]; 0]
        (d, )
    end
end

ZygoteRules.@adjoint function roots(p)
    eV = LinearAlgebra.eigen(companion(Float64.(p)))
    perm = sortperm(eV.values, by=imageigsortby)
    eV.values[perm], rootadjoint(eV, perm)
end

ZygoteRules.@adjoint function hproots(p)
    eV = LinearAlgebra.eigen(companion(Float64.(p)))
    perm = sortperm(eV.values, by=imageigsortby)
    eV.values[perm], rootadjoint(eV, perm)
end

# ZygoteRules.@adjoint function svd(A)
#     s = svd(A)
#     s,  function (Δ)
#         BackwardsLinalg.svd_back(s.U, s.S, s.V, Δ...)
#     end
# end

# This is probably not completely correct, also one has to do many iterations. Doing many iterations can be okay, since then the forward pass is expensive but the backwards pass is free.
# ZygoteRules.@adjoint function sinkhorn(C,a,b; β=0.1, iters=1000)
#     G,f,g = sinkhorn(C,a,b; β=β, iters=iters)
#     (G,f,g), function (Δ)
#         lf = log.(f) .* β
#         lg = log.(g) .* β
#         (G .* β, lf.-sum(lf), lg.-sum(lg))
#     end
# end



# function make_eigen_dual(val::Real, partial)
#     ForwardDiff.Dual{ForwardDiff.tagtype(partial)}(val, partial.partials)
# end
#
# function make_eigen_dual(val::Complex, partial::Complex)
#     Complex(ForwardDiff.Dual{ForwardDiff.tagtype(real(partial))}(real(val), real(partial).partials),
#         ForwardDiff.Dual{ForwardDiff.tagtype(imag(partial))}(imag(val), imag(partial).partials))
# end
#
# using GenericLinearAlgebra
# function LinearAlgebra.eigen(A::StridedMatrix{<:ForwardDiff.Dual})
#     A_values = map(d -> d.value, A)
#     A_values_eig = LinearAlgebra.eigen(A_values)
#     UinvAU = A_values_eig.vectors \ A * A_values_eig.vectors
#     vals_diff = diag(UinvAU)
#     F = similar(A_values, eltype(A_values_eig.values))
#     for i ∈ axes(A_values, 1), j ∈ axes(A_values, 2)
#         if i == j
#             F[i, j] = 0
#         else
#             F[i, j] = inv(A_values_eig.values[j] - A_values_eig.values[i])
#         end
#     end
#     vectors_diff = A_values_eig.vectors * (F .* UinvAU)
#     for i ∈ eachindex(vectors_diff)
#         vectors_diff[i] = make_eigen_dual(A_values_eig.vectors[i], vectors_diff[i])
#     end
#     return Eigen(vals_diff, vectors_diff)
# end

# using ChainRulesCore
# function rrule(::typeof(sinkhorn_convolutional), w, A, B; kwargs...)
#     cost, V, U = sinkhorn_convolutional(w, A,B; kwargs...)
#     function sinkhorn_convolutional_pullback(Δc)
#         return (NO_FIELDS, @thunk(V .= Δc.*V), @thunk(U .= Δc.*U))
#     end
#     return cost, sinkhorn_convolutional_pullback
# end


ZygoteRules.@adjoint function sinkhorn_convolutional(w, A, B; β, kwargs...)
    cost, V, U = sinkhorn_convolutional(w,A,B; β=β, kwargs...)
    V2, U2 = copy(V), copy(U) # This is absolutely required!
    function sinkhorn_convolutional_pullback(Δc)
        Δc *= β
        mV = mean(V2) # If this normalization is not done, one has to normalize the gradients later instead, otherwise most of the gradient will point "into the constraints"
        mU = mean(U2)
        @avx @. V2 = (V2 - mV)*Δc
        @avx @. U2 = (U2 - mU)*Δc

        return (nothing, V2, U2)
    end
    return cost, sinkhorn_convolutional_pullback
end
