const AbstractTuple = Tuple #Union{Tuple, Flux.Tracker.TrackedTuple}

abstract type AbstractModel end
Base.Broadcast.broadcastable(p::AbstractModel) = Ref(p)

"""
    struct AR{T} <: AbstractModel

Represents an all-pole transfer function, i.e., and AR model

# Arguments:
- `a`: denvec
- `ac`: denvec cont. time
- `p`: poles
"""
struct AR{T,Rt <: DiscreteRoots,Ct <: ContinuousRoots} <: AbstractModel
    a::T
    ac::T
    p::Rt
    pc::Ct
    # function AR(xo::AbstractTuple,λ=1e-2)
    #     a = ls(getARregressor(xo[1], xo[2]),λ) |> polyvec
    #     r = DiscreteRoots(hproots(reverse(nograd(a))))
    #     rc = ContinuousRoots(r)
    #     ac = roots2poly(rc)
    #     new{typeof(a), typeof(r), typeof(rc)}(a, ac, r, rc)
    # end
    function AR(a::AbstractVector)
        a = SVector{length(a)}(a)
        r = DiscreteRoots(hproots(reverse(a)))
        rc = ContinuousRoots(r)
        ac = roots2poly(rc)
        new{typeof(a), typeof(r), typeof(rc)}(a, ac, r, rc)
    end
    function AR(rc::ContinuousRoots)
        r = DiscreteRoots(rc)
        a = roots2poly(r)
        ac = roots2poly(rc)
        new{typeof(a), typeof(r), typeof(rc)}(a, ac, r, rc)
    end
    function AR(r::DiscreteRoots)
        rc = ContinuousRoots(r)
        a = roots2poly(r)
        ac = roots2poly(rc)
        new{typeof(a), typeof(r), typeof(rc)}(a, ac, r, rc)
    end
end

"`checkroots(r::DiscreteRoots)` prints a warning if there are roots on the negative real axis."
checkroots(r::DiscreteRoots) = any(imag(r) == 0 && real(r) < 0 for r in r) && @warn "Roots on negative real axis, no corresponding continuous time representation exists."

"""
    AR(X::AbstractArray, order::Int, λ=0.01)

Fit an AR model using [`LS`](@ref) as `fitmethod`

# Arguments:
- `X`: a signal
- `order`: number of roots
- `λ`: reg factor
"""
AR(X::AbstractArray,order::Int,λ=1e-2) = fitmodel(LS(na=order, λ=λ), X)

"""
    struct ARMA{T} <: AbstractModel

Represents an ARMA model, i.e., transfer function

# Arguments:
- `c`: numvec
- `cc`: numvec cont. time
- `a`: denvec
- `ac`: denvec cont. time
- `z`: zeros
- `p`: poles
"""
struct ARMA{T,Rt} <: AbstractModel  where Rt <: DiscreteRoots
    c::T
    cc::T
    a::T
    ac::T
    z::Rt
    p::Rt
end


hproots(a::AbstractVector{T}) where T = roots(Double64.(a))
"""
    ControlSystems.tf(m::AR, ts)

Convert model to a transfer function compatible with ControlSystems.jl
"""
ControlSystems.tf(m::AR, ts) = tf(1, m.a, ts)
"""
    ControlSystems.tf(m::AR)

Convert model to a transfer function compatible with ControlSystems.jl
"""
ControlSystems.tf(m::AR) = tf(1, m.ac)
ControlSystems.tf(m::ARMA, ts) = tf(m.c, m.a, ts)
ControlSystems.tf(m::ARMA) = tf(m.cc, m.ac)
"""
    roots(m::AbstractModel)

Returns the roots of a model
"""
roots

PolynomialRoots.roots(::Discrete, m::AR) = m.p
PolynomialRoots.roots(::Continuous, m::AR) = m.pc
PolynomialRoots.roots(::Discrete, m::ARMA) = m.p
PolynomialRoots.roots(::Continuous, m::ARMA) = m.pc
ControlSystems.pole(::TimeDomain, m::AbstractModel) = roots(d,m)
ControlSystems.tzero(m::ARMA) = m.z
"""
    ControlSystems.denvec(m::AbstractModel) = begin

DOCSTRING
"""
ControlSystems.denvec(::Discrete, m::AbstractModel) = m.a
ControlSystems.numvec(::Discrete, m::ARMA) = m.c
ControlSystems.denvec(::Continuous, m::AbstractModel) = m.ac
ControlSystems.numvec(::Continuous, m::ARMA) = m.cc
"""
    coefficients(::Domain, m::AbstractModel)

Return all fitted coefficients
"""
coefficients(::Discrete, m::AR) = m.a[2:end]
coefficients(::Discrete, m::ARMA) = [m.a[2:end]; m.c]
coefficients(::Continuous, m::AR) = m.ac[2:end]
coefficients(::Continuous, m::ARMA) = [m.ac[2:end]; m.cc]

function domain_transform(d::Continuous, m::AR)
    p = domain_transform(d, roots(m))
    roots2poly(p)
end

"Abstract type that represents a way to fit a model to data"
abstract type FitMethod end
Base.Broadcast.broadcastable(p::FitMethod) = Ref(p)
(fm::FitMethod)(X) = fitmodel(fm, X)

fitmodel(fm,X::AbstractModel) = X

"""
    LS <: FitMethod


# Arguments:
- `na::Int`: number of roots (order of the system)
- `λ::Float64 = 0.01`: reg factor
"""
@kwdef struct LS <: FitMethod
    na::Int
    λ::Float64 = 1e-2
end

"""
    fitmodel(fm::LS, X::AbstractArray)
"""
function fitmodel(fm::LS,X::AbstractArray)
    y,A = getARregressor(X, fm.na)
    a = ls(A, y, fm.λ) |> polyvec
    AR(a)
end


"""
    ls(yA::AbstractTuple, λ=0.01)

Regularized Least-squares
"""
function ls(A, y, λ=1e-2)
    # (A'A + 1e-9I)\(A'y) #src
    A2 = [A; λ*I]
    (A2'A2)\(A'y)
end


"""
    TLS <: FitMethod

Total least squares

# Arguments:
- `na::Int`: number of roots (order of the system)
"""
@kwdef struct TLS <: FitMethod
    na::Int
end

"""
    fitmodel(fm::TLS, X::AbstractArray)
"""
function fitmodel(fm::TLS,X::AbstractArray)
    Ay = getARregressor_(X, fm.na)
    a = tls!(Ay, size(Ay,2)-1) |> vec |> reverse |> polyvec
    AR(a)
end

function fitmodel(fm::TLS,X::AbstractArray,diff::Bool)
    y,A = getARregressor(X, fm.na)
    a = tls(A,y) |> vec |> reverse |> polyvec
    AR(a)
end

"""
    PLR <: FitMethod

DOCSTRING

# Arguments:
- `nc::Int`: order of numerator
- `na::Int`: order of denomenator
- `initial::T = TLS(na=80)`: fitmethod for the initial fit. Can be, e.g., [`LS`](@ref), [`TLS`](@ref) or any function that returns a coefficient vector
- `λ::Float64 = 0.0001`: reg factor
"""
@kwdef struct PLR{T} <: FitMethod
    nc::Int
    na::Int
    initial::T = TLS(na=80)
    λ::Float64 = 1e-4
end
"""
    fitmodel(fm::PLR, X::AbstractArray)
"""
function fitmodel(fm::PLR,X::AbstractArray)
    plr(X,fm.na,fm.nc, fm.initial, λ=fm.λ)
end


"""
    plr(y, na, nc, initial; λ=0.01)

Performs pseudo-linear regression to estimate an ARMA model.

# Arguments:
- `y`: signal
- `na`: denomenator order
- `nc`: numerator order
- `initial`: fitmethod for the initial fit. Can be, e.g., [`LS`](@ref), [`TLS`](@ref) or any function that returns a coefficient vector
- `λ`: reg
"""
function plr(y,na,nc,initial; λ = 1e-2)
    na >= 1 || throw(ArgumentError("na must be positive"))
    w1 = initial(y)
    yhat = A*w1
    ehat = yhat - y_train
    ΔN = length(y)-length(ehat)
    y_train,A = getARXregressor(y[ΔN+1:end-1],ehat[1:end-1],na,nc)
    w = ls(A,y_train,λ)
    a,c = params2poly(w,na,nc)
    rc = DiscreteRoots(hproots(reverse(c)))
    ra = DiscreteRoots(hproots(reverse(a)))
    checkroots(ra)
    ARMA{typeof(c), typeof(rc)}(c,roots2poly(log.(rc)),a,roots2poly(log.(ra)),rc,ra)
end

function params2poly(w,na,nb)
    a = [1; -w[1:na]]
    w = w[na+1:end]
    b = map(nb) do nb
        b = w[1:nb]
        w = w[nb+1:end]
        b
    end
    a,b
end

"""
    toeplitz(c, r)

Returns a toepliz matrix with first column and row specified (c[1] == r[1]).
"""
function toeplitz(c,r)
    @assert c[1] == r[1]
    nc = length(c)
    nr = length(r)
    A  = similar(c, nc, nr)
    A[:,1] = c
    A[1,:] = r
    for i in 2:nr
        A[2:end,i] = A[1:end-1,i-1]
    end
    A
end

function getARregressor(y, na)
    m    = na+1 # Start of yr
    n    = length(y) - m + 1 # Final length of yr
    A    = toeplitz(y[m:m+n-1],y[m:-1:m-na])
    @assert size(A,2) == na+1
    y    = A[:,1] # extract yr
    A    = A[:,2:end]
    return y,A
end

function getARregressor_(y, na)
    y    = reverse(y)
    m    = na+1 # Start of yr
    n    = length(y) - m + 1 # Final length of yr
    Ay   = toeplitz(y[m:m+n-1],y[m:-1:m-na])
    return Ay
end





# getARregressor(a::TrackedArray, b) = Flux.Tracker.track(getARregressor, a, b)




# @grad reshape(xs, dims) = reshape(data(xs), dims), Δ -> (reshape(Δ, size(xs)),nothing)
function getARXregressor(y::AbstractVector,u::AbstractVector, na, nb)
    m    = max(na,nb)+1 # Start of yr
    @assert m >= 1
    n    = length(y) - m + 1 # Final length of yr
    @assert n <= length(y)
    A    = toeplitz(y[m:m+n-1],y[m:-1:m-na])
    @assert size(A,2) == na+1
    y = A[:,1] # extract yr
    A = A[:,2:end]
    s = m-1
    A = [A toeplitz(u[s:s+n-1],u[s:-1:s-nb+1])]
    return y,A
end



poly(w) = [-reverse(w); 1]
polyvec(w) = [1; -w]
polyroots(w) = roots(poly(w))

riroots(p) = (r=roots(p); (real.(r),imag.(r)))
# polyroots(w::TrackedArray) = riroots(poly(w))



function d2c(a,c=1)
    error("This method should go")
    @assert a[1] == 1 "Convert to polynomial first"
    Gd = ss(tf(c,a,1))
    n = Gd.nx
    Md = [Gd.A Gd.B; zeros(1,n) 1]
    Mc = log(Md)
    Ac = Mc[1:n, 1:n]
    e = eigvals(Ac)
    e
end

function polyconv(a,b)
    na,nb = length(a),length(b)
    c = zeros(eltype(a), na+nb-1)
    for i = 1:length(c)
        for j = 1:min(i,na,nb)
            av = j ∈ eachindex(a) ? a[j] : zero(eltype(a))
            bv = i-j+1 ∈ eachindex(b) ? b[i-j+1] : zero(eltype(b))
            c[i] += av*bv
        end
    end
    c
end


"""
    roots2poly(roots)

Accepts a vector of complex roots and returns the polynomial with those roots
"""
function roots2poly(roots)
    p = @MVector [1.]
    for r in 1:length(roots)
        p = _roots2poly_kernel(p, roots[r])
    end
    (real(p))
end

function _roots2poly_kernel(a::StaticVector{N,T},b) where {N,T}
    vT = T <: Real ? Complex{T} : T
    c = MVector{N+1,vT}(ntuple(_->0, N+1))
    c[1] = 1
    n = length(a)
    for i in 2:n
        c[i] = -b*a[i-1] + a[i]
    end
    c[end] = -b*a[end]
    c
end


"""
    residues(a::AbstractVector, r=roots(reverse(a)))

Returns a vector of residues for the system represented by denominator polynomial `a`
"""
function residues(a::AbstractVector, r = roots(reverse(a)))
    a[1] ≈ 1 || @warn "a[1] is $(a[1])"
    n = length(a)-1
    res = map(r) do r
        powers = n:-1:1
        ap     = powers.*a[1:end-1]
        1/sum(j->ap[j]*r^(n-j), 1:n)
    end
end

"""
    residues(r::AbstractRoots)

Returns a vector of residues for the system represented by roots `r`

"""
residues(r::AbstractRoots) = residues(roots2poly(Vector(r)), Vector(r))

# Base.delete_method.(methods(residues))
"""
    poles2model(r::AbstractVector{<:Real}, i::AbstractVector{<:Real})

Returns a transer function with the desired poles. There will be twice as many poles as the length of `i` and `r`.
"""
function poles2model(r::AbstractVector{<:Real}, i::AbstractVector{<:Real})
    roots = [complex.(r, i); complex.(r, -i)]
    AR(roots2poly(roots))
end

function Base.rand(::Type{AR}, dr, di, n=2)
    @assert n % 2 == 0
    n = n ÷ 2
    poles2model(rand(dr,n), rand(di,n))
end


"""
    spectralenergy(G::LTISystem)

Calculates the energy in the spectrum associated with `G`
"""
function spectralenergy(G::LTISystem)
    sys = ss(G)
    A = sys.A
    eltype(A) <: Complex && @warn "This function is known to be incorrect for complex systems"
    e = eigen(A)
    e.values .= reflect(ContinuousRoots(e.values))
    sys.A .= real((e.vectors) * Diagonal(e.values) * inv(e.vectors))
    sys,X = balreal(sys)
    2π*tr(sys.C*X*sys.C')
end

# spectralenergy(a::AbstractArray) = spectralenergy(determine_domain(roots(reverse(a))), a)
spectralenergy(d::TimeDomain, m::AbstractModel) = spectralenergy(d, denvec(d, m))

"""
    spectralenergy(d::TimeDomain, a::AbstractVector)

Calculates the energy in the spectrum associated with denominator polynomial `a`
"""
function spectralenergy(d::TimeDomain, ai::AbstractVector{T})::T where T
    a = Double64.(ai)
    ac = a .* (-1).^(length(a)-1:-1:0)
    a2 = polyconv(ac,a)
    r2 = roots(reverse(a2))
    filterfun = d isa Continuous ? r -> real(r) < 0 : r -> abs(r) < 1
    r2 = filter(filterfun, r2)
    res = residues(a2, r2)
    e = 2π*sum(res)
    ae = real(e)
    if ae < 1e-3  && !(T <: BigFloat) # In this case, accuracy is probably compromised and we should do the calculation with higher precision.
        return spectralenergy(d, big.(a))

    end
    abs(imag(e))/abs(real(e)) > 1e-3 && @warn "Got a large imaginary part in the spectral energy $(abs(imag(e))/abs(real(e)))"
    ae
end

"""
    determine_domain(r)

Tries to automatically figure out which time domain the roots represent. Throws an error if ambiguous.
"""
function determine_domain(r)
    if all(<(0), real.(r)) # Seems like Cont
        if all(<(1), abs.(r)) # Seems like Disc as well
            error("I can't determine if domain is discrete or continuous. Wrap in the correct wrapper (ContinuousRoots / DiscreteRoots)")
        end
        return Continuous()
    end
    all(<(1), abs.(r)) || error("I can't determine if domain is discrete or continuous. Wrap in the correct wrapper (ContinuousRoots / DiscreteRoots)")
    Discrete()
end

function AutoRoots(r)
    determine_domain(r) isa Discrete ? DiscreteRoots(r) : ContinuousRoots(r)
end
normalization_factor(r) = normalization_factor(AutoRoots(r))


function normalization_factor(r::AbstractRoots)
    e = spectralenergy(domain(r), roots2poly(r))
    n = length(r)
    s = e^(1/(2n-1))
end
function normalize_energy(r)
    normalization_factor(r)*r
end



# roots2poly([-1,-2,-3])


# function PolynomialRoots.roots(poly::AbstractVector{<:N}; epsilon::AbstractFloat=NaN,
#     polish::Bool=false) where {N}
#     degree = length(poly) - 1
#     PolynomialRoots.roots!(zeros(Complex{real((N))}, degree), (complex(poly)),
#     epsilon, degree, polish)
# end
#
# function PolynomialRoots.roots!(roots::AbstractVector{<:Complex{T}}, poly::AbstractVector{<:Complex{T}}, epsilon::E,
#     degree::Integer, polish::Bool) where {T<:Union{AbstractFloat, AbstractParticles},E<:Union{AbstractFloat, AbstractParticles}}
#     isnan(epsilon) && (epsilon = eps(T))
#     poly2 = copy(poly)
#     if degree <= 1
#         if degree == 1
#             roots[1] = -poly[1] / poly[2]
#         end
#         return roots
#     end
#     @inbounds for n = degree:-1:3
#         roots[n], iter, success = PolynomialRoots.laguerre2newton(poly2, n, roots[n], 2, epsilon)
#         if ! success
#             roots[n], iter, success = PolynomialRoots.laguerre(poly2, n,
#             zero(Complex{T}), epsilon)
#         end
#         coef = poly2[n+1]
#         @inbounds for i = n:-1:1
#             prev = poly2[i]
#             poly2[i] = coef
#             coef = prev + roots[n]*coef
#         end
#     end
#     roots[1], roots[2] = PolynomialRoots.solve_quadratic_eq(poly2)
#     if polish
#         @inbounds for n = 1:degree # polish roots one-by-one with a full polynomial
#             roots[n], iter, success = PolynomialRoots.laguerre(poly, degree, roots[n], epsilon)
#         end
#     end
#     return roots
# end
