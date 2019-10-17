abstract type TimeDomain end
struct Continuous <: TimeDomain end
struct Discrete <: TimeDomain end

abstract type AbstractAssignmentMethod end

struct SortAssignement{F} <: AbstractAssignmentMethod
    by::F
end
(a::SortAssignement)(e1,e2) = sortperm(nograd(e1), by=a.by), sortperm(nograd(e2), by=a.by)

struct HungarianAssignement <: AbstractAssignmentMethod
end

(a::HungarianAssignement)(e1,e2) = 1:length(e1), hungarian(distmat_euclidean(e1,e2))[1]

abstract type AbstractRoots{T} <: AbstractVector{T} end

struct DiscreteRoots{T, V <: AbstractVector{T}} <: AbstractRoots{T}
    r::V
    DiscreteRoots(r) = new{eltype(r), typeof(r)}(reflectd.(anglesort(r)))
end
StaticArrays.similar_type(r::DiscreteRoots{T,V}) where {T,V} = DiscreteRoots

struct ContinuousRoots{T, V <: AbstractVector{T}} <: AbstractRoots{T}
    r::V
    ContinuousRoots(r) = new{eltype(r), typeof(r)}(reflectc.(eigsort(r)))
end
StaticArrays.similar_type(r::ContinuousRoots{T,V}) where {T,V} = ContinuousRoots

"""
    DiscreteRoots(r::ContinuousRoots) = begin

Represents roots of a polynomial in discrete time
"""
DiscreteRoots(r::ContinuousRoots) = domain_transform(Discrete(), r)
DiscreteRoots(r::DiscreteRoots) = r
"""
    ContinuousRoots(r::DiscreteRoots) = begin

Represents roots of a polynomial in continuous time
"""
ContinuousRoots(r::DiscreteRoots) = domain_transform(Continuous(), r)
ContinuousRoots(r::ContinuousRoots) = r

Base.convert(::Type{DiscreteRoots}, r::Vector{<: Complex}) = DiscreteRoots(r)
Base.convert(::Type{ContinuousRoots}, r::Vector{<: Complex}) = ContinuousRoots(r)

Base.:(≈)(r1::T, r2::T) where T <: AbstractRoots = sort(r1, by=LinearAlgebra.eigsortby) ≈ sort(r2, by=LinearAlgebra.eigsortby)

"""
    domain_transform(d::Domain, e::AbstractRoots)

Change the domain of the roots
"""
domain_transform(d::Continuous,e::ContinuousRoots) = e
domain_transform(d::Discrete,e::ContinuousRoots) = exp(e)
domain_transform(d::Continuous,e::DiscreteRoots) = log(e)
domain_transform(d::Discrete,e::DiscreteRoots) = e
domain(::DiscreteRoots) = Discrete()
domain(::ContinuousRoots) = Continuous()


eigsort(e) = sort(e, by=imag)
anglesort(e) = sort(e, by=angle)

Base.Vector(r::AbstractRoots) = r.r
Base.log(r::DiscreteRoots) = ContinuousRoots(log.(r.r))
Base.exp(r::ContinuousRoots) = DiscreteRoots(exp.(r.r))
ControlSystems.c2d(r::DiscreteRoots,h=1) = ContinuousRoots(log.(r.r) ./ h)
d2c(r::ContinuousRoots,h=1) = DiscreteRoots(exp.(h .* r.r))

Lazy.@forward DiscreteRoots.r (Base.length, Base.getindex, Base.setindex!, Base.size, Base.enumerate, Base.abs, Base.abs2, Base.real, Base.imag)
Lazy.@forward ContinuousRoots.r (Base.length, Base.getindex, Base.setindex!, Base.size, Base.enumerate, Base.abs, Base.abs2, Base.real, Base.imag)

sqrtmag(magang::Tuple) = sqrt.(magang[1]), magang[2]
logmag(magang::Tuple) = log.(magang[1]), magang[2]

# FIXME: what does it mean to have negative magnitude??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function logmag(e::Number)
    m,a = polar(e)
    m = log(m)
    polar2complex(m,a)
end

polar2complex(m,a) = m * exp(im*a)

function sqrtreim(e)
    r,i = real(e), imag(e)
    complex(sqrt(abs(r))*sign(r), sqrt(abs(i))*sign(i))
end

function sqrtim(e)
    r,i = real(e), imag(e)
    complex(r, sqrt(abs(i))*sign(i))
end

function sqrtre(e)
    r,i = real(e), imag(e)
    complex(sqrt(abs(r))*sign(r), i)
end

function logreim(e, ϵ=1e-4)
    r,i = real(e), imag(e)
    complex(log(abs(r) + ϵ)*sign(r), log(abs(i) + ϵ)*sign(i))
end

function logim(e, ϵ=1e-4)
    r,i = real(e), imag(e)
    complex(r, log(abs(i) + ϵ)*sign(i))
end

function logre(e, ϵ=1e-4)
    r,i = real(e), imag(e)
    complex(log(abs(r) + ϵ)*sign(r), i)
end

function invre(e)
    r,i = real(e), imag(e)
    complex(1/r,i)
end

function squarere(e)
    r,i = real(e), imag(e)
    complex(r*r,i)
end

function expre(e)
    r,i = real(e), imag(e)
    complex(exp(r),i)
end

function projim(e)
    complex(0,imag(e))
end

function polar(e)
    abs.(e), angle.(e)
end

function residueweight(e)
    res = residues(ContinuousRoots(e))
    s1(abs.(π*abs2.(res)./ real.(e)))
end

"""
    polar(e::Number)

magnitude and angle of a complex number
"""
function polar(e::Number)
    abs(e), angle(e)
end

function polar_ang(e)
    mag, ang = polar(e)
    I   = sortperm(ang)
    mag[I], ang[I], I
end

function polar_mag(e)
    mag, ang = polar(e)
    I   = sortperm(mag)
    mag[I], ang[I], I
end

toreim(x::AbstractVector{<:Complex}) = (real.(x), imag.(x))
toreim(x::Tuple) = x
# toreim(x::Flux.Tracker.TrackedTuple) = x

reflectc(x::Real) = x < 0 ? x : -x
reflectc(x::Complex) = complex(x.re < 0 ? x.re : -x.re,x.im)
function reflectd(x)
    a = abs(x)
    a < 1 && return x
    1/a * exp(im*angle(x))
end
"""
    reflect(r::AbstractRoots)

Reflects unstable roots to a corresponding stable position (in unit circle for disc. in LHP for cont.)
"""
reflect(r::ContinuousRoots) = ContinuousRoots(reflectc.(r.r))
reflect(r::DiscreteRoots) = DiscreteRoots(reflectd.(r.r))
# function remove_negreal(p)
#     pn = filter(x->x.im == 0 && x.re < 0, p)
#     pn = sort(pn, by=real)
#     i = 1
#     if length(pn) % 2 != 0
#         p[findfirst(==(pn[1]),p)] = 0.01
#         i += 1
#     end
#     while i < length(pn)
#         a = conv([1, -pn[i]], [1, -pn[i+1]])
#         w = sqrt(a[3])
#         @show z = a[2]/(2w)
#         @assert z >= 1
#         z = 0.99
#         @show pi = -z*w + sqrt((z*w)^2-4w^2 + 0*im)
#         p[p .== pn[i]] = pi
#         p[p .== pn[i+1]] = conj(pn)
#         i += 2
#     end
#     p
# end

scalereal(x) = complex(1x.re, x.im)

"""
    hungariansort(p1, p2)

takes two vectors of numbers and sorts and returns `p2` such that it is in the order of the best Hungarian assignement between `p1` and `p2`. Uses `abs` for comparisons, works on complex numbers.
"""
function hungariansort(p1,p2)
    length(p1) <= 2 && (return p2)
    distmat = abs.(p1 .- transpose(p2))
    ass,cost = hungarian(distmat)
    p2[ass]
end
