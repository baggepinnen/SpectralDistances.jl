const AbstractTuple = Tuple #Union{Tuple, Flux.Tracker.TrackedTuple}

"abstract type AbstractModel <: ControlSystems.LTISystem end"
abstract type AbstractModel <: ControlSystems.LTISystem end
Base.Broadcast.broadcastable(p::AbstractModel) = Ref(p)

"""
    struct AR <: AbstractModel

Represents an all-pole transfer function, i.e., and AR model

# Arguments:
- `a`: denvec
- `ac`: denvec cont. time
- `p`: discrete time poles
- `pc`: continuous time poles
- `b`: Numerator scalar
"""
struct AR{T,Rt <: DiscreteRoots,Ct <: ContinuousRoots} <: AbstractModel
    a::T
    ac::T
    p::Rt
    pc::Ct
    b::Float64
    # function AR(xo::AbstractTuple,λ=1e-2)
    #     a = ls(getARregressor(xo[1], xo[2]),λ) |> polyvec
    #     r = DiscreteRoots(hproots(reverse(nograd(a))))
    #     rc = ContinuousRoots(r)
    #     ac = roots2poly(rc)
    #     new{typeof(a), typeof(r), typeof(rc)}(a, ac, r, rc)
    # end
end
function AR(a::AbstractVector, σ²=nothing)
    a = isderiving() ? complex.(a) : a # SVector{length(a)}(a)
    r = DiscreteRoots(hproots(rev(a)))
    rc = ContinuousRoots(r)
    ac = roots2poly(rc)
    ac = isderiving() ? complex.(ac) : ac
    b = σ² === nothing ? one(eltype(a)) : scalefactor(Continuous(), ac, σ²)
    AR{typeof(a), typeof(r), typeof(rc)}(a, ac, r, rc, b)
end

function AR(::Continuous, ac::AbstractVector, σ²=nothing)
    ac = isderiving() ? complex.(ac) : ac # SVector{length(ac)}(ac)
    rc = ContinuousRoots(hproots(rev(ac)))
    r = DiscreteRoots(rc)
    a = roots2poly(r)
    a = isderiving() ? complex.(a) : a
    b = σ² === nothing ? one(eltype(a)) : scalefactor(Continuous(), ac, σ²)
    AR{typeof(a), typeof(r), typeof(rc)}(a, ac, r, rc, b)
end
function AR(rc::ContinuousRoots, σ²=nothing)
    r = DiscreteRoots(rc)
    a = roots2poly(r)
    ac = roots2poly(rc)
    b = σ² === nothing ? one(eltype(a)) : scalefactor(Continuous(), ac, σ²)
    AR{typeof(a), typeof(r), typeof(rc)}(a, ac, r, rc, b)
end
function AR(r::DiscreteRoots, σ²=nothing)
    rc = ContinuousRoots(r)
    a = roots2poly(r)
    ac = roots2poly(rc)
    b = σ² === nothing ? one(eltype(a)) : scalefactor(Continuous(), ac, σ²)
    AR{typeof(a), typeof(r), typeof(rc)}(a, ac, r, rc, b)
end

"`checkroots(r::DiscreteRoots)` prints a warning if there are roots on the negative real axis."
checkroots(r::DiscreteRoots) = any(imag(r) == 0 && real(r) < 0 for r in r) && @warn("Roots on negative real axis, no corresponding continuous time representation exists. Consider prefiltering the signal or decreasing the regularization factor.", maxlog=5)

"""
    AR(X::AbstractArray, order::Int)

Fit an AR model using [`TLS`](@ref) as `fitmethod`

# Arguments:
- `X`: a signal
- `order`: number of roots
"""
AR(X::AbstractArray,order::Int) = fitmodel(TLS(na=order), X, var(X))

"""
    struct ARMA{T} <: AbstractModel

Represents an ARMA model, i.e., transfer function

# Arguments:
- `b`: numvec
- `bc`: numvec cont. time
- `a`: denvec
- `ac`: denvec cont. time
- `z`: zeros
- `p`: poles
"""
struct ARMA{T1,T2,T3,T4,Rt <: DiscreteRoots,Crt <: ContinuousRoots} <: AbstractModel
    b::T1
    bc::T2
    a::T3
    ac::T4
    z::Rt
    zc::Crt
    p::Rt
    pc::Crt

    function ARMA(b::AbstractVector,a::AbstractVector)
        r = DiscreteRoots(hproots(rev(a)))
        rc = ContinuousRoots(r)
        ac = roots2poly(rc)
        ac = isderiving() ? complex.(ac) : ac
        z  = DiscreteRoots(hproots(rev(b)))
        zc = ContinuousRoots(z)
        b   = roots2poly(z)
        bc  = roots2poly(zc)
        new{typeof(b), typeof(bc), typeof(a), typeof(ac), typeof(r), typeof(rc)}(b, bc, a, ac, z, zc, r, rc)
    end

    function ARMA(::Continuous, bc, ac::AbstractVector)
        rc = ContinuousRoots(hproots(rev(ac)))
        zc = ContinuousRoots(hproots(rev(bc)))
        r = DiscreteRoots(rc)
        z = DiscreteRoots(zc)
        a = roots2poly(r) #|> Vector
        b = roots2poly(z) #|> Vector
        new{typeof(b), typeof(bc), typeof(a), typeof(ac), typeof(r), typeof(rc)}(b, bc, a, ac, z, zc, r, rc)
    end
    function ARMA(zc::ContinuousRoots, rc::ContinuousRoots)
        r = DiscreteRoots(rc)
        z = DiscreteRoots(zc)
        a = roots2poly(r)
        ac = roots2poly(rc)
        b = roots2poly(z)
        bc = roots2poly(zc)
        new{typeof(b), typeof(bc), typeof(a), typeof(ac), typeof(r), typeof(rc)}(b, bc, a, ac, z, zc, r, rc)
    end
    function ARMA(z::DiscreteRoots, r::DiscreteRoots)
        rc = ContinuousRoots(r)
        zc = ContinuousRoots(z)
        a = roots2poly(r)
        ac = roots2poly(rc)
        b = roots2poly(z)
        bc = roots2poly(zc)
        new{typeof(b), typeof(bc), typeof(a), typeof(ac), typeof(r), typeof(rc)}(b, bc, a, ac, z, zc, r, rc)
    end

end


hproots(a::AbstractVector{T}) where T = eigsort(roots(Double64.(a)))
hproots(a::AbstractVector{<:Double64}) = eigsort(roots(a))
"""
    ControlSystems.tf(m::AR, ts)

Convert model to a transfer function compatible with ControlSystems.jl
"""
ControlSystems.tf(m::AR, ts) = tf(1, Vector(m.a), ts)
"""
    ControlSystems.tf(m::AR)

Convert model to a transfer function compatible with ControlSystems.jl
"""
ControlSystems.tf(m::AR) = tf(1, Vector(m.ac))
ControlSystems.tf(m::ARMA, ts) = tf(Vector(m.c), Vector(m.a), ts)
ControlSystems.tf(m::ARMA) = tf(Vector(m.cc), Vector(m.ac))

Base.convert(::Type{T}, m::AbstractModel) where T <: TransferFunction{<:ControlSystems.SisoRational} = tf(m)

Base.convert(::Type{T}, m::AbstractModel) where T <: TransferFunction{<:ControlSystems.SisoZpk} = zpk(tf(m))

function ARMA(g::ControlSystems.TransferFunction)
    ControlSystems.issiso(g) || error("Can only convert SISO systems to ARMA")
    b,a = numvec(g)[], denvec(g)[]
    ControlSystems.iscontinuous(g) && return ARMA(Continuous(), b, a)
    ARMA(b, a)
end

Base.convert(::Type{ControlSystems.TransferFunction}, m::AbstractModel) = tf(m)
Base.promote_rule(::Type{<:ControlSystems.TransferFunction}, ::Type{<:AbstractModel}) = ControlSystems.TransferFunction

function Base.isapprox(m1::AR, m2::AR, args...; kwargs...)
    all(isapprox(getfield(m1,field), getfield(m2,field), args...; kwargs...) for field in fieldnames(typeof(m1)))
end

function change_precision(F, m::AR)
    CF = Complex{F}
    a  = F.(m.a)
    r  = change_precision(F,m.p)
    rc = change_precision(F,m.pc)
    AR{typeof(a), typeof(r), typeof(rc)}(a, F.(m.ac), r, rc, m.b)
end


"""
    roots(m::AbstractModel)

Returns the roots of a model
"""
roots

PolynomialRoots.roots(::Discrete, m::AR) = m.p
PolynomialRoots.roots(::Continuous, m::AR) = m.pc
PolynomialRoots.roots(::Discrete, m::ARMA) = m.p
PolynomialRoots.roots(::Continuous, m::ARMA) = m.pc
ControlSystems.pole(d::TimeDomain, m::AbstractModel) = roots(d,m)
ControlSystems.tzero(::Discrete, m::ARMA) = m.z
ControlSystems.tzero(::Continuous, m::ARMA) = m.zc
"""
    ControlSystems.denvec(::TimeDomain, m::AbstractModel)

Get the denominator polynomial vector
"""
ControlSystems.denvec(::Discrete, m::AbstractModel) = m.a
ControlSystems.numvec(::Discrete, m::AR) = error("Not yet implemented")#[m.b]
ControlSystems.numvec(::Continuous, m::AR) = [m.b]
ControlSystems.numvec(::Discrete, m::ARMA) = m.b
ControlSystems.denvec(::Continuous, m::AbstractModel) = m.ac
ControlSystems.numvec(::Continuous, m::ARMA) = m.bc


for T in (:AR, :ARMA)
    @eval begin
        ControlSystems.bode(m::$T, w::AbstractVector{<:Real}, args...; kwargs...) = bode(tf(m), w, args...; kwargs...)
        ControlSystems.nyquist(m::$T, w::AbstractVector{<:Real}, args...; kwargs...) = nyquist(tf(m), w, args...; kwargs...)
        ControlSystems.freqresp(m::$T, w::AbstractVector{<:Real}, args...; kwargs...) = freqresp(tf(m), w, args...; kwargs...)
        ControlSystems.step(m::$T, Tf::Real, args...; kwargs...) = step(tf(m), Tf, args...; kwargs...)
    end
end

function Base.getproperty(m::AbstractModel, p::Symbol)
    p === :Ts && return 0.0
    p === :nx && return length(m.ac)
    p === :nu && return 1
    p === :ny && return 1
    getfield(m,p)
end

"""
    coefficients(::Domain, m::AbstractModel)

Return all fitted coefficients
"""
coefficients(::Discrete, m::AR) = m.a[2:end]
coefficients(::Discrete, m::ARMA) = [m.a[2:end]; m.b]
coefficients(::Continuous, m::AR) = m.ac[2:end]
coefficients(::Continuous, m::ARMA) = [m.ac[2:end]; m.bc]

Base.getindex(m::AbstractModel, i) = i < length(m.ac) ? m.ac[1+i] : m.b[i-length(m.ac)+1]
Base.length(m::AbstractModel) = length(m.a)+length(m.b)-1
Base.size(m::AbstractModel) = (1,1)

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

This fitmethod is a good default option.

# Arguments:
- `na::Int`: number of roots (order of the system). The number of peaks in the spectrum will be `na÷2`.
- `λ::Float64 = 0.01`: reg factor
"""
LS
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
    AR(a, var(X))
end


# diffpol(n) = [(-1)^k*binomial(n,k) for k in 1:n]
diffpol(n) = diagm(0=>ones(n), 1=>-ones(n))[:,1:n]

"""
    ls(A, y, λ=0)

Regularized Least-squares
"""
function ls(A, y, λ=0)
    isderiving() && (return _ls(A,y,λ))
    n = size(A,2)
    if λ > 0
        A = [A; sqrt(λ)*I]
        y = [y;zeros(n)]
    end
    # @show size(A2), size(y)
    # svd(A2)\y
    # (A'A - γ*I)\(A'y)
    A\y
end

function _ls(A,y,λ)
    # if λ > 0
        # A2 = [A; sqrt(λ)*I]
    # else
        # A2 = A
    # end
    # @show size(A2), size(y), size(A)
   # (A2'A2)\(A'y)
   (A'A + λ*I)\A'y
end


"""
    TLS <: FitMethod

Total least squares. This fit method is good if the spectrum has sharp peaks, in particular if the number of peaks is known in advance.

# Arguments:
- `na::Int`: number of roots (order of the system). The number of peaks in the spectrum will be `na÷2`.
- `λ::Float64 = 0`: reg factor
"""
TLS
@kwdef struct TLS <: FitMethod
    na::Int
    λ::Float64 = 0
end

"""
    fitmodel(fm::TLS, X::AbstractArray)
"""
function fitmodel(fm::TLS,X::AbstractArray)
    isderiving() && return fitmodel(fm,X,true)
    Ay = getARregressor_(X, fm.na)
    if fm.λ > 0
        Ay = [Ay;sqrt(fm.λ)*I]
        Ay[end,end] = 0
    end
    a = tls!(Ay, size(Ay,2)-1) |> vec |> rev |> polyvec
    AR(a, var(X))
end

function fitmodel(fm::TLS,X::AbstractArray,diff::Bool)
    y,A = getARregressor(X, fm.na)
    if fm.λ > 0
        A = [A;sqrt(fm.λ)*I]
        y = [y;zeros(size(A,2))]
    end
    a = tls(A,y) |> vec |> rev |> polyvec
    AR(a, var(X))
end

"""
    IRLS <: FitMethod

Iteratively reqeighted least squares. This fitmethod is currently not recommended, it does not appear to produce nice spectra. (feel free to try it out though).

# Arguments:
- `na::Int`: number of roots (order of the system)
"""
IRLS
@kwdef struct IRLS <: FitMethod
    na::Int
end

"""
    fitmodel(fm::IRLS, X::AbstractArray)
"""
function fitmodel(fm::IRLS,X::AbstractArray)
    y,A = getARregressor(X, fm.na)
    a = irls(A,y) |> vec |> rev |> polyvec
    AR(a, var(X))
end

"""
    PLR <: FitMethod

Pseudo linear regression. Estimates the noise components by performing an initial fit of higher order. Tihs fitmethod produces an [`ARMA`](@ref) model. Support for ARMA models is not as strong as for [`AR`](@ref) models.

# Arguments:
- `nc::Int`: order of numerator
- `na::Int`: order of denomenator
- `initial::T = TLS(na=80)`: fitmethod for the initial fit. Can be, e.g., [`LS`](@ref), [`TLS`](@ref) or any function that returns a coefficient vector
- `λ::Float64 = 0.0001`: reg factor
"""
PLR
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
    y_train,A = getARregressor(y,initial.na)
    w1 = initial isa TLS ? tls(A,y_train) : ls(A,y_train,initial.λ)
    yhat = A*vec(w1)
    ehat = y_train - yhat
    ΔN = length(y)-length(ehat)
    y_train,A = getARXregressor(y[ΔN:end-1],ehat,na,nc)
    w = tls(A,y_train)
    a,c = params2poly(w,na,nc)
    c[1] = 1
    # TODO: scalefactor for PLR
    ARMA(c,a)
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
    y    = rev(y)
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



poly(w) = [-rev(w); 1]
polyvec(w) = [1; -w]
polyroots(w) = roots(poly(w))


function polyconv(a,b)
    na,nb = length(a),length(b)
    if na > nb
        (a,b) = (b,a)
    end
    c = zeros(promote_type(eltype(a), eltype(b)), na+nb-1)
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
function roots2poly(roots::AbstractVector{CT})::Vector{real(CT)} where CT
    isderiving() && return roots2poly_zygote(roots)
    p = @MVector [1.]
    for r in 1:length(roots)
        p = _roots2poly_kernel(p, roots[r])
    end
    Vector{real(CT)}(real(p))
end

function _roots2poly_kernel(a::StaticVector{N,T},b) where {N,T}
    T2 = promote_type(T,typeof(b))
    vT = T2 <: Real ? Complex{T} : T2
    c = MVector{N+1,vT}(ntuple(_->0, N+1))
    c[1] = 1
    n = length(a)
    for i in 2:n
        c[i] = -b*a[i-1] + a[i]
    end
    c[end] = -b*a[end]
    c
end

function roots2poly_zygote(roots)
    p = [complex(1.)]
    for r in roots
        p = polyconv(p, [1, -r])
    end
    real(p)
end


@inline function polyderivative(a)
    n = length(a)-1
    powers = n:-1:1
    ap     = powers.*a[1:end-1]
end

# @inline function polyval(ap,r)
#     n = length(ap)
#     sum(j->ap[j]*r^(n-j), 1:n)
# end # Below is workaround for Zygote bug

@inline function polyval(ap,r)
    n = length(ap)
    s = zero(promote_type(eltype(ap), typeof(r)))
    @inbounds for j = 1:n
        s += ap[j]*r^(n-j)
    end
    s
end

@inline polyval(b::Number,_) = b

function rev(x)
    if isderiving()
        x[end:-1:1] # Workaround for Zygote not being able to reverse vectors
    else
        reverse(x)
    end
end

"""
    residues(a::AbstractVector, b, r=roots(reverse(a)))

Returns a vector of residues for the system represented by denominator polynomial `a`

Ref: slide 21 https://stanford.edu/~boyd/ee102/rational.pdf
Tihs methid is numerically sensitive. Note that if two poles are on top of each other, the residue is infinite.
"""
function residues(a::AbstractVector, b, r::AbstractVector{CT} = hproots(rev(a)))::Vector{CT} where CT
    # a[1] ≈ 1 || println("Warning: a[1] is ", a[1]) # This must not be @warn because Zygote can't differentiate that
    n = length(a)-1
    ap = polyderivative(a)
    res = map(r) do r
        polyval(b, r)/polyval(ap, r)
    end
end

residues(d::TimeDomain, m::AbstractModel) = residues(denvec(d,m), numvec(d,m), roots(d,m))


"""
    residues(r::AbstractRoots)

Returns a vector of residues for the system represented by roots `r`

"""
function residues(r::AbstractRoots)
    a = isderiving() ? complex.(roots2poly(r)) : roots2poly(r)
    residues(a, 1, r)
end

function residueweight(m::AbstractModel)
    res = residues(Continuous(), m)
    e = roots(Continuous(), m)
    rw = abs.(π*abs2.(res)./ real.(e))
    isderiving() ? complex.(rw) : rw
end

function unitweight(m::AbstractModel)
    r = roots(Continuous(), m)
    RT = float(real(eltype(r)))
    N = length(r)
    fill(RT(1/N), N)
end

# Base.delete_method.(methods(residues))
"""
    poles2model(r::AbstractVector{<:Real}, i::AbstractVector{<:Real})

Returns a transer function with the desired poles. There will be twice as many poles as the length of `i` and `r`.
"""
function poles2model(r::AbstractVector{<:Real}, i::AbstractVector{<:Real})
    roots = [complex.(r, i); complex.(r, -i)]
    AR(Continuous(),roots2poly(roots))
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
function spectralenergy(d::TimeDomain, m::AbstractModel)
    spectralenergy(d, denvec(d, m), numvec(d,m))
end

"""
    spectralenergy(d::TimeDomain, a::AbstractVector, b)

Calculates the energy in the spectrum associated with transfer function `b/a`
"""
function spectralenergy(d::TimeDomain, ai::AbstractVector{T}, b)::T where T
    a = Double64.(ai)
    ac = a .* (-1).^(length(a)-1:-1:0)
    a2 = polyconv(ac,a)
    r2 = roots(rev(a2))
    filterfun = d isa Continuous ? r -> real(r) < 0 : r -> abs(r) < 1
    # r2 = filter(filterfun, r2)
    r2 = r2[filterfun.(r2)] # Zygote can't diff the filter above
    res = residues(a2, b, r2)
    e = 2π*sum(res)
    ae = real(e)
    if ae < 1e-3  && !(T <: BigFloat) && !isderiving() # In this case, accuracy is probably compromised and we should do the calculation with higher precision.
        return spectralenergy(d, big.(a), b)

    end
    abs(imag(e))/abs(real(e)) > 1e-3 && println("Got a large imaginary part in the spectral energy $(abs(imag(e))/abs(real(e)))")
    @assert ae >= 0 "Computed energy was negative: $ae"
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

function AutoRoots(d::TimeDomain, r)
    d isa Discrete ? DiscreteRoots(r) : ContinuousRoots(r)
end
function AutoRoots(r)
    determine_domain(r) isa Discrete ? DiscreteRoots(r) : ContinuousRoots(r)
end
normalization_factor(r) = normalization_factor(AutoRoots(r))


"""
    normalize_energy(r)

Returns the factor that, when used to multiply the poles, results in a system with unit spectral energy.
"""
function normalization_factor(r::AbstractRoots)
    e = spectralenergy(domain(r), roots2poly(r), 1)
    n = length(r)
    s = e^(1/(2n-1))
end

"""
    normalize_energy(r)

Returns poles scaled to achieve unit spectral energy.
"""
function normalize_energy(r)
    normalization_factor(r)*r
end

"""
    scalefactor(::TimeDomain, a, [b,] σ²)

Returns `k` such that the system with numerator `kb` and denomenator `a` has spectral energy `σ²`. `σ²` should typically be the variance of the corresponding time signal.
"""
function scalefactor(d::TimeDomain, a, b, σ²)
    e = spectralenergy(d, a, b)
    (σ²/(e))
end

function scalefactor(d::TimeDomain, a, σ²)
    e = spectralenergy(d, a, 1)
    (σ²/(e))
end


"""
    examplemodels(n = 10)

Return `n` random models with 6 complex poles each.
"""
function examplemodels(n=10)
    ζ = [0.1, 0.3, 0.7]
    models = map(1:n) do _
        pol = [1]
        for i = eachindex(ζ)
            pol = SpectralDistances.polyconv(pol, [1,2ζ[i] + 0.1randn(),1+0.1randn()])
        end
        AR(Continuous(),pol)
    end
end
