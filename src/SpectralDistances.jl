module SpectralDistances

using LinearAlgebra, Statistics
using DSP, Distances, PolynomialRoots, ControlSystems, SinkhornDistance, Hungarian, ZygoteRules, Optim, Clustering, Lazy, RecipesBase, StatsBase, Roots, QuadGK, OrdinaryDiffEq, ThreadTools, DoubleFloats, StaticArrays
import FiniteDifferences
import Base.@kwdef

export ls, plr, logmag, polar, polar_ang, polar_ang, toreim, reflect, hungariansort, coefficients, batch_loss, s1, v1, n1, roots, pole, twoD, threeD, precompute, plot_assignment, roots2poly, evalfr, Log, Identity, residues, residueweight, normalize_energy, normalization_factor, polyconv

export CoefficientDistance,
ModelDistance,
EuclideanRootDistance,
ManhattanRootDistance,
HungarianRootDistance,
SinkhornRootDistance,
KernelWassersteinRootDistance,
OptimalTransportModelDistance,
OptimalTransportSpectralDistance,
ClosedFormSpectralDistance,
CramerSpectralDistance,
BuresDistance,
EnergyDistance,
CompositeDistance,
trivial_transport
# closed_form_wass,
# closed_form_wass_noinverse

export sinkhorn,
IPOT

export AR,
ARMA,
PLR,
LS

export Continuous,
Discrete,
SortAssignement,
HungarianAssignement,
DiscreteRoots,
ContinuousRoots,
domain_transform

include("eigenvalue_manipulations.jl")
include("ltimodels.jl")
include("losses.jl")
include("interpolations.jl")
include("utils.jl")
include("plotting.jl")
include("sinkhorn.jl")

end # module
