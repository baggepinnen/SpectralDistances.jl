using Documenter
using SpectralDistances, ControlSystems, Clustering, JuMP, GLPK
# using DocumenterLaTeX

makedocs(
    sitename = "SpectralDistances",
    # format = LaTeX(),
    format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
    modules = [SpectralDistances],
    pages = ["index.md", "ltimodels.md", "distances.md", "interpolations.md", "plotting.md", "misc.md", "examples.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/baggepinnen/SpectralDistances.jl"
)
