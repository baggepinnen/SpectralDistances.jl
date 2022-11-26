using Documenter
using SpectralDistances, ControlSystemsBase, Clustering, JuMP, Convex, SCS, GLPK, DSP, DynamicAxisWarping, SlidingDistancesBase

using Plots
plotly()


@info "makedocs"
makedocs(
    sitename = "SpectralDistances",
    # format = LaTeX(),
    format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
    modules = [SpectralDistances],
    pages = ["index.md", "ltimodels.md", "distances.md", "time.md", "interpolations.md", "plotting.md", "misc.md", "examples.md", "taskview.md"]
)

# For debugging purposes
# makedocs(
#     sitename = "SpectralDistances",
#     # format = LaTeX(),
#     format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
#     modules = [SpectralDistances],
#     pages = ["time.md"]
# )

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
@info "deploydocs"
deploydocs(
    repo = "github.com/baggepinnen/SpectralDistances.jl"
)
@info "done"
