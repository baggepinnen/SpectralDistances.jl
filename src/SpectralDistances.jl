module SpectralDistances

using DSP, PolynomialRoots, ControlSystems, SinkhornDistance, Hungarian, Flux, Optim, Clustering, Lazy, RecipesBase
import FiniteDifferences

export ls, plr

export loss_spectral_ot, loss_spectral_ot!, ls_loss, ls_loss_angle, ls_loss_eigvals_cont, ls_loss_eigvals_cont_logmag, ls_loss_eigvals_disc

include("ltimodels.jl")
include("losses.jl")
include("eigenvalue_manipulations.jl")
include("interpolations.jl")
include("utils.jl")
include("plotting.jl")

end # module
