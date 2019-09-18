module SpectralDistances

using LinearAlgebra, Statistics
using DSP, Distances, PolynomialRoots, ControlSystems, SinkhornDistance, Hungarian, Flux, Optim, Clustering, Lazy, RecipesBase, StatsBase
import FiniteDifferences
import Base.@kwdef

export ls, plr, logmag, polar, polar_ang, polar_ang, toreim, reflect, hungariansort, coefficients, batch_loss, s1, v1, n1

export EuclideanCoefficientDistance,
InnerProductCoefficientDistance,
ModelDistance,
EuclideanRootDistance,
HungarianRootDistance,
KernelWassersteinRootDistance,
OptimalTransportModelDistance,
OptimalTransportSpectralDistance,
EnergyDistance,
CompositeDistance

export AR,
ARMA,
PLR,
LS

export Continuous,
Discrete,
SortAssignement,
HungarianAssignement,
DiscreteRoots,
ContinuousRoots

include("eigenvalue_manipulations.jl")
include("ltimodels.jl")
include("losses.jl")
include("interpolations.jl")
include("utils.jl")
include("plotting.jl")

end # module
