"""
$(README)

# Exports
$(EXPORTS)
"""
module SpectralDistances

using LinearAlgebra, Statistics, Printf
using DSP,
    ControlSystems,
    Distances,
    DocStringExtensions,
    DoubleFloats,
    FillArrays,
    Hungarian,
    Lazy,
    LoopVectorization,
    Optim,
    OrdinaryDiffEq,
    PolynomialRoots,
    ProgressMeter,
    QuadGK,
    RecipesBase,
    Roots,
    StaticArrays,
    StatsBase,
    ThreadTools,
    TotalLeastSquares,
    ZygoteRules,
    UnbalancedOptimalTransport

using SlidingDistancesBase
import SlidingDistancesBase: floattype, lastlength, distance_profile, distance_profile!

import Base.@kwdef

export ls,
    plr,
    polar,
    toreim,
    reflect,
    hungariansort,
    coefficients,
    s1,
    v1,
    v1!,
    n1,
    roots,
    pole,
    twoD,
    threeD,
    precompute,
    assignmentplot,
    roots2poly,
    evalfr,
    Log,
    Identity,
    residues,
    residueweight,
    simplex_residueweight,
    unitweight,
    normalize_energy,
    spectralenergy,
    normalization_factor,
    polyconv,
    domain,
    fitmodel,
    move_real_poles,
    checkroots,
    change_precision,
    embedding,
    distmat,
    distmat_euclidean,
    distmat_euclidean!

export evaluate,
distance_profile,
AbstractDistance,
AbstractRationalDistance,
AbstractSignalDistance,
SymmetricDistance,
CoefficientDistance,
ModelDistance,
EuclideanRootDistance,
HungarianRootDistance,
OptimalTransportRootDistance,
KernelWassersteinRootDistance,
DiscretizedRationalDistance,
WelchOptimalTransportDistance,
WelchLPDistance,
RationalOptimalTransportDistance,
OptimalTransportHistogramDistance,
DiscreteGridTransportDistance,
RationalCramerDistance,
BuresDistance,
EnergyDistance,
CompositeDistance,
discrete_grid_transportplan
# closed_form_wass,
# closed_form_wass_noinverse

export sinkhorn,
sinkhorn_log,
sinkhorn_log!,
IPOT,
sinkhorn_diff,
sinkhorn_unbalanced,
transport_plan

using UnbalancedOptimalTransport: KL, TV, Balanced, RG
export KL, TV, Balanced, RG

export FitMethod,
AbstractModel,
AR,
ARMA,
PLR,
LS,
TLS,
IRLS,
examplemodels


export TimeDomain,
Continuous,
Discrete,
SortAssignement,
HungarianAssignement,
DiscreteRoots,
ContinuousRoots,
domain_transform

export interpolator, barycenter, ISA, barycentric_coordinates, barycentric_weighting, barycenter_convolutional, BCWorkspace

export TimeDistance, TimeVaryingAR, TimeWindow

include("eigenvalue_manipulations.jl")
include("ltimodels.jl")
include("losses.jl")
include("interpolations.jl")
include("utils.jl")
include("sinkhorn.jl")
include("barycenter.jl")
include("time.jl")
include("plotting.jl")
include("adjoints.jl")

import Requires
function __init__()
    Requires.@require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
        @eval using BackwardsLinalg
        @inline nograd(x) = Zygote.dropgrad(x)
    end

    Requires.@require GLPK = "60bf3e95-4087-53dc-ae20-288a0d20c6a6" begin
        Requires.@require JuMP = "4076af6c-e467-56ae-b986-b466b2749572" begin
            export ot_jump
            include("jump.jl")
        end
        Requires.@require Convex = "f65535da-76fb-5f13-bab9-19810c17039a" begin
            export ot_convex
            include("convex.jl")
        end
    end

    Requires.@require Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5" begin
        export kbarycenters
        include("kbarycenters.jl")
    end

    Requires.@require DynamicAxisWarping = "aaaaaaaa-4a10-5553-b683-e707b00e83ce" begin
        include("dtw.jl")
    end
end
end # module
