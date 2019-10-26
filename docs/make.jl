using Documenter
using SpectralDistances
# using DocumenterLaTeX

makedocs(
    sitename = "SpectralDistances",
    # format = LaTeX(),
    format = Documenter.HTML(prettyurls = true),
    modules = [SpectralDistances],
    pages = ["index.md", "ltimodels.md", "distances.md", "interpolations.md", "plotting.md", "misc.md", "examples.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/baggepinnen/SpectralDistances.jl"
)
