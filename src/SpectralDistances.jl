"""
$(README)

# Exports
$(EXPORTS)
"""
module SpectralDistances

using LinearAlgebra, Statistics
using DSP, Distances, PolynomialRoots, ControlSystems, Hungarian, ZygoteRules, Lazy, RecipesBase, StatsBase, Roots, QuadGK, OrdinaryDiffEq, ThreadTools, DoubleFloats, StaticArrays, TotalLeastSquares, DocStringExtensions, Optim

import Base.@kwdef

export ls, plr, logmag, polar, polar_ang, polar_ang, toreim, reflect, hungariansort, coefficients, batch_loss, s1, v1, n1, roots, pole, twoD, threeD, precompute, assignmentplot, roots2poly, evalfr, Log, Identity, residues, residueweight, unitweight, normalize_energy, spectralenergy, normalization_factor, polyconv, domain, fitmodel, move_real_poles, checkroots

export AbstractDistance,
AbstractRationalDistance,
AbstractSignalDistance,
CoefficientDistance,
ModelDistance,
EuclideanRootDistance,
HungarianRootDistance,
SinkhornRootDistance,
KernelWassersteinRootDistance,
DiscretizedRationalDistance,
WelchOptimalTransportDistance,
RationalOptimalTransportDistance,
OptimalTransportHistogramDistance,
RationalCramerDistance,
BuresDistance,
EnergyDistance,
CompositeDistance,
discrete_grid_transportplan
# closed_form_wass,
# closed_form_wass_noinverse

export sinkhorn,
sinkhorn_log,
IPOT,
sinkhorn_diff

export AbstractModel,
AR,
ARMA,
PLR,
LS,
TLS,
IRLS


export TimeDomain,
Continuous,
Discrete,
SortAssignement,
HungarianAssignement,
DiscreteRoots,
ContinuousRoots,
domain_transform

export interpolator, barycenter, ISA, barycentric_coordinates, barycentric_weighting

include("eigenvalue_manipulations.jl")
include("ltimodels.jl")
include("losses.jl")
include("interpolations.jl")
include("utils.jl")
include("plotting.jl")
include("sinkhorn.jl")
include("barycenter.jl")
include("adjoints.jl")

import Requires
function __init__()
    Requires.@require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
        @eval using BackwardsLinalg
    end
end
end # module
