using Documenter
using SpectralDistances

makedocs(
    sitename = "SpectralDistances",
    format = Documenter.HTML(prettyurls = false),
    modules = [SpectralDistances]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/baggepinnen/SpectralDistances.jl"
)
