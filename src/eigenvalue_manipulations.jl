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
end

struct ContinuousRoots{T, V <: AbstractVector{T}} <: AbstractRoots{T}
    r::V
end

Base.convert(::Type{DiscreteRoots}, r::Vector{<: Complex}) = DiscreteRoots(r)
Base.convert(::Type{ContinuousRoots}, r::Vector{<: Complex}) = ContinuousRoots(r)

domain_transform(d::Continuous,e::ContinuousRoots) = e
domain_transform(d::Discrete,e::ContinuousRoots) = exp(e)
domain_transform(d::Continuous,e::DiscreteRoots) = log(e)
domain_transform(d::Discrete,e::DiscreteRoots) = e
domain_transform(d, e) = domain_transform(domain(d), e)

Base.Vector(r::AbstractRoots) = r.r
Base.log(r::DiscreteRoots) = ContinuousRoots(log.(r.r))
Base.exp(r::ContinuousRoots) = DiscreteRoots(exp.(r.r))
ControlSystems.c2d(r::DiscreteRoots,h=1) = ContinuousRoots(log.(r.r) ./ h)
d2c(r::ContinuousRoots,h=1) = DiscreteRoots(exp.(h .* r.r))

Lazy.@forward DiscreteRoots.r (Base.length, Base.getindex, Base.setindex!, Base.size, Base.enumerate, Base.abs, Base.abs2)
Lazy.@forward ContinuousRoots.r (Base.length, Base.getindex, Base.setindex!, Base.size, Base.enumerate, Base.abs, Base.abs2)

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

function polar(e)
    abs.(e), angle.(e)
end

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
toreim(x::Flux.Tracker.TrackedTuple) = x


reflectc(x) = complex(x.re < 0 ? x.re : -x.re,x.im)
reflectd(x) = error("Not implemented")
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

function hungariansort(p1,p2)
    length(p1) <= 2 && (return p2)
    distmat = abs.(p1 .- transpose(p2))
    ass,cost = hungarian(distmat)
    p2[ass]
end
