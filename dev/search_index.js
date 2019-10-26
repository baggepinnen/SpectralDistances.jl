var documenterSearchIndex = {"docs":
[{"location":"interpolations/#Interpolations-1","page":"Interpolations","title":"Interpolations","text":"","category":"section"},{"location":"interpolations/#","page":"Interpolations","title":"Interpolations","text":"Whenever you define a distance, that distance implies the existence of a shortest path, a geodesic. An interpolation is essentially a datapoint on that shortest path. We provide some functionality to interpolate between different spectra and models.","category":"page"},{"location":"interpolations/#","page":"Interpolations","title":"Interpolations","text":"Pages = [\"interpolations.md\"]\nOrder   = [:type, :function, :macro, :constant]","category":"page"},{"location":"interpolations/#","page":"Interpolations","title":"Interpolations","text":"Modules = [SpectralDistances]\nPrivate = false\nPages   = [\"interpolations.jl\"]","category":"page"},{"location":"interpolations/#","page":"Interpolations","title":"Interpolations","text":"Below is an example usage of interpolations. We initially create two random systems, we then define the distance under which to interpolate and then calculate the frequency response for some different values of the interpolation parameter t in (01)","category":"page"},{"location":"interpolations/#","page":"Interpolations","title":"Interpolations","text":"using SpectralDistances, ControlSystems, Distances, Plots, Random\nRandom.seed!(0)\n\nn = 4\n\nr1 = complex.(-0.01 .+ 0.001randn(3), 2randn(3))\nr1 = ContinuousRoots([r1; conj.(r1)])\n\nr2 = complex.(-0.01 .+ 0.001randn(3), 2randn(3))\nr2 = ContinuousRoots([r2; conj.(r2)])\n\nr1,r2 = normalize_energy.((r1, r2))\n\nA1 = AR(r1)\nA2 = AR(r2)\n\n\n##\nfig1 = plot()\nt = 0.1\ndist = ClosedFormSpectralDistance(domain=Continuous(), p=2, interval=(0., exp10(1.01)))\ninterp = SpectralDistances.interpolator(dist, A1, A2)\nw = exp10.(LinRange(-1.5, 1, 300))\nfor t = LinRange(0, 1, 7)\n    Φ = clamp.(interp(w,t), 1e-10, 100)\n    plot!(w, sqrt.(Φ), xscale=:log10, yscale=:log10, line_z = t, lab=\"\", xlabel=\"Frequency\", title=\"W_2\", ylims=(1e-3, 1e1), colorbar=false, l=(1,), c=:viridis)\nend\n\nrdist = EuclideanRootDistance(domain=Continuous(), p=2)\ninterp = SpectralDistances.interpolator(rdist, A1, A2, normalize=false)\nfig2 = plot()\nw = exp10.(LinRange(-1.5, 1, 300))\nfor t = LinRange(0, 1, 7)\n    Φ = interp(w,t)\n    plot!(w, sqrt.(Φ), xscale=:log10, yscale=:log10, line_z = t, lab=\"\", xlabel=\"\", title=\"RD\", ylims=(1e-3, 1e1), colorbar=false, l=(1,), c=:viridis)\nend\n\nfig3 = plot()\nw = exp10.(LinRange(-1.5, 1, 300))\nΦ1 = bode(tf(A1), w)[1][:]\nΦ2 = bode(tf(A2), w)[1][:]\nfor t = LinRange(0, 1, 7)\n    plot!(w, (1-t).*Φ1 .+ t.*Φ2, xscale=:log10, yscale=:log10, line_z = t, lab=\"\", xlabel=\"\", title=\"L_2\", ylims=(1e-3, 1e1), colorbar=false, l=(1,), c=:viridis)\nend\n\nfig = plot(fig1, fig2, fig3, layout=(3,1))\nsavefig(\"interpolation.svg\"); nothing # hide","category":"page"},{"location":"interpolations/#","page":"Interpolations","title":"Interpolations","text":"(Image: )","category":"page"},{"location":"misc/#Misc.-1","page":"Misc.","title":"Misc.","text":"","category":"section"},{"location":"misc/#","page":"Misc.","title":"Misc.","text":"This section lists some convenience utilities for normalization and projection of data.","category":"page"},{"location":"misc/#","page":"Misc.","title":"Misc.","text":"Pages = [\"misc.md\"]\nOrder   = [:type, :function, :macro, :constant]","category":"page"},{"location":"misc/#","page":"Misc.","title":"Misc.","text":"Modules = [SpectralDistances]\nPrivate = false\nPages   = [\"utils.jl\"]","category":"page"},{"location":"misc/#SpectralDistances.n1-Tuple{Any}","page":"Misc.","title":"SpectralDistances.n1","text":"n1(x) = begin\n\nnormalize x norm 1\n\n\n\n\n\n","category":"method"},{"location":"misc/#SpectralDistances.s1","page":"Misc.","title":"SpectralDistances.s1","text":"s1(x, dims=:) = begin\n\nnormalize x sums to 1\n\n\n\n\n\n","category":"function"},{"location":"misc/#SpectralDistances.threeD-Tuple{Any}","page":"Misc.","title":"SpectralDistances.threeD","text":"threeD(X)\n\nProject X to three dimensions using PCA\n\n\n\n\n\n","category":"method"},{"location":"misc/#SpectralDistances.twoD-Tuple{Any}","page":"Misc.","title":"SpectralDistances.twoD","text":"twoD(X)\n\nProject X to two dmensions using PCA\n\n\n\n\n\n","category":"method"},{"location":"misc/#SpectralDistances.v1","page":"Misc.","title":"SpectralDistances.v1","text":"v1(x, dims=:)\n\nnormalize x var 1\n\n\n\n\n\n","category":"function"},{"location":"distances/#Distances-1","page":"Distances","title":"Distances","text":"","category":"section"},{"location":"distances/#Overview-1","page":"Distances","title":"Overview","text":"","category":"section"},{"location":"distances/#","page":"Distances","title":"Distances","text":"The following is a reference on all the distances defined in this package. Once a distance is defined, it can be evaluated in one of two ways, defined by the  Distances.jl interface","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"dist = DistanceType(options)\nd = evaluate(d, x1, x2)\nd = dist(x1, x2)","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"Before we proceed, the following distances are available","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"Pages = [\"distances.md\"]\nOrder   = [:type]","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"using SpectralDistances, InteractiveUtils","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"Some of these distances operate directly on signals, these are","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"foreach(println, subtypes(SpectralDistances.AbstractSignalDistance)) # hide","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"Of these, ModelDistance is a bit special, works like this","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"ModelDistance","category":"page"},{"location":"distances/#SpectralDistances.ModelDistance","page":"Distances","title":"SpectralDistances.ModelDistance","text":"ModelDistance{D <: AbstractDistance} <: AbstractSignalDistance\n\nA model distance operates on signals and works by fitting an LTI model to the signals before calculating the distance. The distance between the LTI models is defined by the field distance. This is essentially a wrapper around the inner distance that handles the fitting of a model to the signals. How the model is fit is determined by fitmethod.\n\nArguments:\n\nfitmethod::FitMethod: LS or PLR\ndistance::D: The inner distance between the models\n\n\n\n\n\n","category":"type"},{"location":"distances/#","page":"Distances","title":"Distances","text":"The inner distance in ModelDistance can be any AbstractModelDistance. The options are","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"foreach(println, subtypes(SpectralDistances.AbstractModelDistance)) # hide","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"These distances operate on LTI models. Some operate on the coefficients of the models","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"foreach(println, subtypes(SpectralDistances.AbstractCoefficientDistance)) # hide","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"and some operate on the roots of the models","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"foreach(println, subtypes(SpectralDistances.AbstractRootDistance)) # hide","category":"page"},{"location":"distances/#A-full-example-1","page":"Distances","title":"A full example","text":"","category":"section"},{"location":"distances/#","page":"Distances","title":"Distances","text":"To use the SinkhornRootDistance and let it operate on signals, we may construct our distance object as follows","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"innerdistance = SinkhornRootDistance(domain=Continuous(), β=0.005, p=2)\ndist = ModelDistance(LS(na=30), innerdistance)\nX1, X2 = randn(1000), randn(1000);\ndist(X1,X2)\n\ndist = ModelDistance(LS(na=2), innerdistance);\nt = 0:0.01:10;\nX1, X2 = sin.(2π*1 .*t), sin.(2π*1.1 .*t); # Two signals that are close in frequency\ndist(X1,X2)\nX1, X2 = sin.(2π*1 .*t), sin.(2π*2 .*t);   # Two signals that are further apart in frequency\ndist(X1,X2)","category":"page"},{"location":"distances/#Using-Welch-periodograms-1","page":"Distances","title":"Using Welch periodograms","text":"","category":"section"},{"location":"distances/#","page":"Distances","title":"Distances","text":"We can calculate the Wasserstein distance between spectra estimated using the Welch method like so","category":"page"},{"location":"distances/#","page":"Distances","title":"Distances","text":"dist = WelchOptimalTransportDistance(p=2)\nX1, X2 = randn(1000), randn(1000);\ndist(X1,X2)\nt = 0:0.01:10;\nX1, X2 = sin.(2π*1 .*t), sin.(2π*1.1 .*t); # Two signals that are close in frequency\ndist(X1,X2)\nX1, X2 = sin.(2π*1 .*t), sin.(2π*2 .*t);   # Two signals that are further apart in frequency\ndist(X1,X2)","category":"page"},{"location":"distances/#Function-reference-1","page":"Distances","title":"Function reference","text":"","category":"section"},{"location":"distances/#","page":"Distances","title":"Distances","text":"Pages = [\"distances.md\"]\nOrder   = [:function, :macro, :constant]","category":"page"},{"location":"distances/#Docstrings-1","page":"Distances","title":"Docstrings","text":"","category":"section"},{"location":"distances/#","page":"Distances","title":"Distances","text":"Modules = [SpectralDistances]\nPrivate = false\nPages   = [\"losses.jl\", \"sinkhorn.jl\"]","category":"page"},{"location":"distances/#SpectralDistances.BuresDistance","page":"Distances","title":"SpectralDistances.BuresDistance","text":"BuresDistance <: AbstractDistance\n\nDistance between pos.def. matrices\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.ClosedFormSpectralDistance","page":"Distances","title":"SpectralDistances.ClosedFormSpectralDistance","text":"ClosedFormSpectralDistance{DT, MT} <: AbstractModelDistance\n\ncalculates the Wasserstein distance using the closed-form solution based on integrals and inverse cumulative functions.\n\nArguments:\n\ndomain::DT: Discrete or Continuous\np::Int = 1: order\nmagnitude::MT = Identity():\ninterval = (-(float(π)), float(π)): Integration interval\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.CoefficientDistance","page":"Distances","title":"SpectralDistances.CoefficientDistance","text":"CoefficientDistance{D, ID} <: AbstractCoefficientDistance\n\nDistance metric based on model coefficients\n\nArguments:\n\ndomain::D: Discrete or Continuous\ndistance::ID = SqEuclidean(): Inner distance between coeffs\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.CramerSpectralDistance","page":"Distances","title":"SpectralDistances.CramerSpectralDistance","text":"CramerSpectralDistance{DT} <: AbstractModelDistance\n\nSimilar to ClosedFormSpectralDistance but does not use inverse functions.\n\nArguments:\n\ndomain::DT: Discrete or Continuous\np::Int = 2: order\ninterval = (-(float(π)), float(π)): Integration interval\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.EnergyDistance","page":"Distances","title":"SpectralDistances.EnergyDistance","text":"EnergyDistance <: AbstractSignalDistance\n\nstd(x1) - std(x2)\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.EuclideanRootDistance","page":"Distances","title":"SpectralDistances.EuclideanRootDistance","text":"EuclideanRootDistance{D, A, F1, F2} <: AbstractRootDistance\n\nSimple euclidean distance between roots of transfer functions\n\nArguments:\n\ndomain::D: Discrete or Continuous\nassignment::A = SortAssignement(imag): Determines how roots are assigned. An alternative is HungarianAssignement\ntransform::F1 = identity: DESCRIPTION\nweight : A function used to calculate weights for the induvidual root distances. A good option is residueweight\np::Int = 2 : Order of the distance\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.HungarianRootDistance","page":"Distances","title":"SpectralDistances.HungarianRootDistance","text":"HungarianRootDistance{D, ID <: Distances.PreMetric, F} <: AbstractRootDistance\n\nDOCSTRING\n\nArguments:\n\ndomain::D: Discrete or Continuous\ndistance::ID = SqEuclidean(): Inner distance\ntransform::F = identity: If provided, this Function transforms all roots before the distance is calculated\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.KernelWassersteinRootDistance","page":"Distances","title":"SpectralDistances.KernelWassersteinRootDistance","text":"KernelWassersteinRootDistance{D, F, DI} <: AbstractRootDistance\n\nDOCSTRING\n\nArguments:\n\ndomain::D: Discrete or Continuous\nλ::Float64 = 1.0: reg param\ntransform::F = identity: If provided, this Function transforms all roots before the distance is calculated\ndistance::DI = SqEuclidean(): Inner distance\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.OptimalTransportModelDistance","page":"Distances","title":"SpectralDistances.OptimalTransportModelDistance","text":"OptimalTransportModelDistance{WT, DT} <: AbstractModelDistance\n\nDOCSTRING\n\nArguments:\n\nw::WT = LinRange(0.01, 0.5, 300): Frequency set\ndistmat::DT = distmat_euclidean(w, w): DESCRIPTION\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.SinkhornRootDistance","page":"Distances","title":"SpectralDistances.SinkhornRootDistance","text":"SinkhornRootDistance{D, F1, F2} <: AbstractRootDistance\n\nThe Sinkhorn distance between roots. The weights are provided by weight, which defaults to residueweight.\n\nArguments:\n\ndomain::D: Discrete or Continuous\ntransform::F1 = identity: Probably not needed.\nweight::F2 =s1 ∘ residueweight: A function used to calculate weights for the induvidual root distances.\nβ::Float64 = 0.01: Amount of entropy regularization\niters::Int = 10000: Number of iterations of the Sinkhorn algorithm.\np::Int = 2 : Order of the distance\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.WelchOptimalTransportDistance","page":"Distances","title":"SpectralDistances.WelchOptimalTransportDistance","text":"WelchOptimalTransportDistance{DT, AT <: Tuple, KWT <: NamedTuple} <: AbstractSignalDistance\n\nCalculates the Wasserstein distance between two signals by estimating a Welch periodogram of each.\n\nArguments:\n\ndistmat::DT: you may provide a matrix array for this\nargs::AT = (): Options to the Welch function\nkwargs::KWT = NamedTuple(): Options to the Welch function\np::Int = 2 : Order of the distance\n\n\n\n\n\n","category":"type"},{"location":"distances/#SpectralDistances.domain-Tuple{Any}","page":"Distances","title":"SpectralDistances.domain","text":"domain(d::AbstractDistance)\n\nReturn the domain of the distance\n\n\n\n\n\n","category":"method"},{"location":"distances/#SpectralDistances.domain_transform-Tuple{SpectralDistances.AbstractDistance,Any}","page":"Distances","title":"SpectralDistances.domain_transform","text":"domain_transform(d::AbstractDistance, e)\n\nChange domain of roots\n\n\n\n\n\n","category":"method"},{"location":"distances/#SpectralDistances.precompute","page":"Distances","title":"SpectralDistances.precompute","text":"precompute(d::AbstractDistance, As, threads=true)\n\nPerform computations that only need to be donce once when several pairwise distances are to be computed\n\nArguments:\n\nAs: A vector of models\nthreads: Us multithreading? (true)\n\n\n\n\n\n","category":"function"},{"location":"distances/#SpectralDistances.trivial_transport-Union{Tuple{T}, Tuple{AbstractArray{T,1},AbstractArray{T,1}}, Tuple{AbstractArray{T,1},AbstractArray{T,1},Any}} where T","page":"Distances","title":"SpectralDistances.trivial_transport","text":"trivial_transport(x::AbstractVector{T}, y::AbstractVector{T}, tol=sqrt(eps(T))) where T\n\nCalculate the optimal-transport plan between two vectors that are assumed to have the same support, with sorted support points.\n\n\n\n\n\n","category":"method"},{"location":"examples/#Examples-1","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#The-closed-form-solution-1","page":"Examples","title":"The closed-form solution","text":"","category":"section"},{"location":"examples/#","page":"Examples","title":"Examples","text":"In this example we will simply visalize two spectra, the locations of their poles and the cumulative spectrum functions.","category":"page"},{"location":"examples/#","page":"Examples","title":"Examples","text":"using OrdinaryDiffEq, ControlSystems, SpectralDistances, Plots\ngr(grid=false)\n\nG1   = tf(1,[1,0.12,1])*tf(1,[1,0.1,0.1])\nG2   = tf(1,[1,0.12,2])*tf(1,[1,0.1,0.4])\na1   = denvec(G1)[]\na2   = denvec(G2)[]\nn    = length(a1)\n\nf1c  = w -> abs2(1/sum(j->a1[j]*(im*w)^(n-j), 1:n))\nf2c  = w -> abs2(1/sum(j->a2[j]*(im*w)^(n-j), 1:n))\nsol1 = SpectralDistances.c∫(f1c,0,3π)\nsol2 = SpectralDistances.c∫(f2c,0,3π)\n\nfig1 = plot((sol1.t .+ sol1.t[2]).*2π, sqrt.(sol1 ./ sol1[end]), fillrange=sqrt.(sol2(sol1.t) ./ sol2[end]), fill=(0.6,:purple), l=(2,:blue))\nplot!((sol2.t .+ sol2.t[2]).*2π, sqrt.(sol2(sol2.t) ./ sol2[end]), l=(2,:orange), xscale=:log10, legend=false, grid=false, xlabel=\"Frequency\", xlims=(1e-2,2pi))\n\nfig2 = bodeplot([G1, G2], exp10.(LinRange(-1.5, 1, 200)), legend=false, grid=false, title=\"\", linecolor=[:blue :orange], l=(2,), plotphase=false)\n\nfig3 = pzmap([G1, G2], legend=false, grid=false, title=\"\", markercolor=[:blue :orange], color=[:blue :orange], m=(2,:c), xlims=(-0.5,0.5))\nvline!([0], l=(:black, :dash))\nhline!([0], l=(:black, :dash))\n\nplot(fig1, fig2, fig3, layout=(1,3))\nsavefig(\"cumulative.png\"); nothing # hide","category":"page"},{"location":"examples/#","page":"Examples","title":"Examples","text":"(Image: )","category":"page"},{"location":"ltimodels/#Models-and-root-manipulations-1","page":"Models and root manipulations","title":"Models and root manipulations","text":"","category":"section"},{"location":"ltimodels/#","page":"Models and root manipulations","title":"Models and root manipulations","text":"using SpectralDistances, InteractiveUtils","category":"page"},{"location":"ltimodels/#Overview-1","page":"Models and root manipulations","title":"Overview","text":"","category":"section"},{"location":"ltimodels/#","page":"Models and root manipulations","title":"Models and root manipulations","text":"This package supports two kind of LTI models, AR and ARMA. AR represents a model with only poles whereas ARMA has zeros as well.","category":"page"},{"location":"ltimodels/#","page":"Models and root manipulations","title":"Models and root manipulations","text":"To fit a model to data, one first has to specify a FitMethod, the options are","category":"page"},{"location":"ltimodels/#","page":"Models and root manipulations","title":"Models and root manipulations","text":"foreach(println, subtypes(SpectralDistances.FitMethod)) # hide","category":"page"},{"location":"ltimodels/#","page":"Models and root manipulations","title":"Models and root manipulations","text":"For example, to estimate an AR model of order 30 using least-squares, we can do the following","category":"page"},{"location":"ltimodels/#","page":"Models and root manipulations","title":"Models and root manipulations","text":"data = randn(1000);\nfitmethod = LS(na=30)\nSpectralDistances.fitmodel(fitmethod, data)","category":"page"},{"location":"ltimodels/#Type-reference-1","page":"Models and root manipulations","title":"Type reference","text":"","category":"section"},{"location":"ltimodels/#","page":"Models and root manipulations","title":"Models and root manipulations","text":"Pages = [\"ltimodels.md\"]\nOrder   = [:type]","category":"page"},{"location":"ltimodels/#Function-reference-1","page":"Models and root manipulations","title":"Function reference","text":"","category":"section"},{"location":"ltimodels/#","page":"Models and root manipulations","title":"Models and root manipulations","text":"Pages = [\"ltimodels.md\"]\nOrder   = [:type, :function]","category":"page"},{"location":"ltimodels/#Docstrings-1","page":"Models and root manipulations","title":"Docstrings","text":"","category":"section"},{"location":"ltimodels/#","page":"Models and root manipulations","title":"Models and root manipulations","text":"Modules = [SpectralDistances]\nPrivate = false\nPages   = [\"ltimodels.jl\",\"eigenvalue_manipulations.jl\"]","category":"page"},{"location":"ltimodels/#SpectralDistances.AR","page":"Models and root manipulations","title":"SpectralDistances.AR","text":"struct AR{T} <: AbstractModel\n\nRepresents an all-pole transfer function, i.e., and AR model\n\nArguments:\n\na: denvec\nac: denvec cont. time\np: poles\n\n\n\n\n\n","category":"type"},{"location":"ltimodels/#SpectralDistances.AR","page":"Models and root manipulations","title":"SpectralDistances.AR","text":"AR(X::AbstractArray, order::Int, λ=0.01)\n\nFit an AR model using LS as fitmethod\n\nArguments:\n\nX: a signal\norder: number of roots\nλ: reg factor\n\n\n\n\n\n","category":"type"},{"location":"ltimodels/#SpectralDistances.ARMA","page":"Models and root manipulations","title":"SpectralDistances.ARMA","text":"struct ARMA{T} <: AbstractModel\n\nRepresents an ARMA model, i.e., transfer function\n\nArguments:\n\nc: numvec\ncc: numvec cont. time\na: denvec\nac: denvec cont. time\nz: zeros\np: poles\n\n\n\n\n\n","category":"type"},{"location":"ltimodels/#PolynomialRoots.roots","page":"Models and root manipulations","title":"PolynomialRoots.roots","text":"roots(m::AbstractModel)\n\nReturns the roots of a model\n\n\n\n\n\n","category":"function"},{"location":"ltimodels/#SpectralDistances.checkroots-Tuple{DiscreteRoots}","page":"Models and root manipulations","title":"SpectralDistances.checkroots","text":"checkroots(r::DiscreteRoots) prints a warning if there are roots on the negative real axis.\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.coefficients-Tuple{Discrete,AR}","page":"Models and root manipulations","title":"SpectralDistances.coefficients","text":"coefficients(::Domain, m::AbstractModel)\n\nReturn all fitted coefficients\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.fitmodel-Tuple{LS,AbstractArray}","page":"Models and root manipulations","title":"SpectralDistances.fitmodel","text":"fitmodel(fm::LS, X::AbstractArray)\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.fitmodel-Tuple{PLR,AbstractArray}","page":"Models and root manipulations","title":"SpectralDistances.fitmodel","text":"fitmodel(fm::PLR, X::AbstractArray)\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.fitmodel-Tuple{TLS,AbstractArray}","page":"Models and root manipulations","title":"SpectralDistances.fitmodel","text":"fitmodel(fm::TLS, X::AbstractArray)\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.ls","page":"Models and root manipulations","title":"SpectralDistances.ls","text":"ls(yA::AbstractTuple, λ=0.01)\n\nRegularized Least-squares\n\n\n\n\n\n","category":"function"},{"location":"ltimodels/#SpectralDistances.plr-NTuple{4,Any}","page":"Models and root manipulations","title":"SpectralDistances.plr","text":"plr(y, na, nc, initial; λ=0.01)\n\nPerforms pseudo-linear regression to estimate an ARMA model.\n\nArguments:\n\ny: signal\nna: denomenator order\nnc: numerator order\ninitial: fitmethod for the initial fit. Can be, e.g., LS, TLS or any function that returns a coefficient vector\nλ: reg\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.residues","page":"Models and root manipulations","title":"SpectralDistances.residues","text":"residues(a::AbstractVector, r=roots(reverse(a)))\n\nReturns a vector of residues for the system represented by denominator polynomial a\n\n\n\n\n\n","category":"function"},{"location":"ltimodels/#SpectralDistances.residues-Tuple{SpectralDistances.AbstractRoots}","page":"Models and root manipulations","title":"SpectralDistances.residues","text":"residues(r::AbstractRoots)\n\nReturns a vector of residues for the system represented by roots r\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.roots2poly-Tuple{Any}","page":"Models and root manipulations","title":"SpectralDistances.roots2poly","text":"roots2poly(roots)\n\nAccepts a vector of complex roots and returns the polynomial with those roots\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.Continuous","page":"Models and root manipulations","title":"SpectralDistances.Continuous","text":"Continuous time domain\n\n\n\n\n\n","category":"type"},{"location":"ltimodels/#SpectralDistances.ContinuousRoots","page":"Models and root manipulations","title":"SpectralDistances.ContinuousRoots","text":"ContinuousRoots{T, V <: AbstractVector{T}} <: AbstractRoots{T}\n\nRepresents roots in continuous time\n\n\n\n\n\n","category":"type"},{"location":"ltimodels/#SpectralDistances.ContinuousRoots-Tuple{DiscreteRoots}","page":"Models and root manipulations","title":"SpectralDistances.ContinuousRoots","text":"ContinuousRoots(r::DiscreteRoots) = begin\n\nRepresents roots of a polynomial in continuous time\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.Discrete","page":"Models and root manipulations","title":"SpectralDistances.Discrete","text":"Discrete (sampled) time domain\n\n\n\n\n\n","category":"type"},{"location":"ltimodels/#SpectralDistances.DiscreteRoots","page":"Models and root manipulations","title":"SpectralDistances.DiscreteRoots","text":"DiscreteRoots{T, V <: AbstractVector{T}} <: AbstractRoots{T}\n\nRepresent roots in discrete time\n\n\n\n\n\n","category":"type"},{"location":"ltimodels/#SpectralDistances.DiscreteRoots-Tuple{ContinuousRoots}","page":"Models and root manipulations","title":"SpectralDistances.DiscreteRoots","text":"DiscreteRoots(r::ContinuousRoots) = begin\n\nRepresents roots of a polynomial in discrete time\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.HungarianAssignement","page":"Models and root manipulations","title":"SpectralDistances.HungarianAssignement","text":"HungarianAssignement <: AbstractAssignmentMethod\n\nSort roots using Hungarian method\n\n\n\n\n\n","category":"type"},{"location":"ltimodels/#SpectralDistances.SortAssignement","page":"Models and root manipulations","title":"SpectralDistances.SortAssignement","text":"SortAssignement{F} <: AbstractAssignmentMethod\n\nContains a single Function field that determines what to sort roots by.\n\n\n\n\n\n","category":"type"},{"location":"ltimodels/#SpectralDistances.domain_transform-Tuple{Discrete,ContinuousRoots}","page":"Models and root manipulations","title":"SpectralDistances.domain_transform","text":"domain_transform(d::Domain, e::AbstractRoots)\n\nChange the domain of the roots\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.hungariansort","page":"Models and root manipulations","title":"SpectralDistances.hungariansort","text":"hungariansort(p1, p2)\n\ntakes two vectors of numbers and sorts and returns p2 such that it is in the order of the best Hungarian assignement between p1 and p2. Uses abs for comparisons, works on complex numbers.\n\n\n\n\n\n","category":"function"},{"location":"ltimodels/#SpectralDistances.polar-Tuple{Number}","page":"Models and root manipulations","title":"SpectralDistances.polar","text":"polar(e::Number)\n\nmagnitude and angle of a complex number\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.reflect-Tuple{ContinuousRoots}","page":"Models and root manipulations","title":"SpectralDistances.reflect","text":"reflect(r::AbstractRoots)\n\nReflects unstable roots to a corresponding stable position (in unit circle for disc. in LHP for cont.)\n\n\n\n\n\n","category":"method"},{"location":"ltimodels/#SpectralDistances.residueweight-Tuple{SpectralDistances.AbstractRoots}","page":"Models and root manipulations","title":"SpectralDistances.residueweight","text":"residueweight(e::AbstractRoots)\n\nReturns a vector where each entry is roughly corresponding to the amount of energy contributed to the spectrum be each pole.\n\n\n\n\n\n","category":"method"},{"location":"plotting/#Plotting-1","page":"Plotting","title":"Plotting","text":"","category":"section"},{"location":"plotting/#","page":"Plotting","title":"Plotting","text":"Pages = [\"plotting.md\"]\nOrder   = [:type, :function, :macro, :constant]","category":"page"},{"location":"plotting/#","page":"Plotting","title":"Plotting","text":"Modules = [SpectralDistances]\nPrivate = false\nPages   = [\"plotting.jl\"]","category":"page"},{"location":"plotting/#SpectralDistances.assignmentplot","page":"Plotting","title":"SpectralDistances.assignmentplot","text":"assignmentplot(m1, m2)\n\nPlots the poles of two models and the optimal assignment between them\n\n\n\n\n\n","category":"function"},{"location":"#","page":"SpectralDistances","title":"SpectralDistances","text":"(Image: Build Status) (Image: codecov)","category":"page"},{"location":"#SpectralDistances-1","page":"SpectralDistances","title":"SpectralDistances","text":"","category":"section"},{"location":"#","page":"SpectralDistances","title":"SpectralDistances","text":"This package facilitates the calculation of distances between signals, primarily in the frequency domain. The main functionality revolves around rational spectra, i.e., the spectrum of a rational function, commonly known as a transfer function.","category":"page"},{"location":"#High-level-overview-1","page":"SpectralDistances","title":"High-level overview","text":"","category":"section"},{"location":"#","page":"SpectralDistances","title":"SpectralDistances","text":"The main workflow is as follows","category":"page"},{"location":"#","page":"SpectralDistances","title":"SpectralDistances","text":"Define a distance\nEvaluate the distance between two points (signals, histograms, periodograms, etc.)","category":"page"},{"location":"#","page":"SpectralDistances","title":"SpectralDistances","text":"This package extends Distances.jl and all distance types are subtypes of Distances.PreMetric, even though some technically are true metrics and some are not even pre-metrics.","category":"page"},{"location":"#","page":"SpectralDistances","title":"SpectralDistances","text":"Many distances are differentiable and can thus be used for gradient-based learning. The rest of this manual is divided into the following sections","category":"page"},{"location":"#Contents-1","page":"SpectralDistances","title":"Contents","text":"","category":"section"},{"location":"#","page":"SpectralDistances","title":"SpectralDistances","text":"Pages = [\"distances.md\", \"ltimodels.md\", \"interpolations.md\", \"plotting.md\", \"misc.md\", \"examples.md\"]","category":"page"},{"location":"#All-Exported-functions-and-types-1","page":"SpectralDistances","title":"All Exported functions and types","text":"","category":"section"},{"location":"#","page":"SpectralDistances","title":"SpectralDistances","text":"Pages = [\"distances.md\", \"ltimodels.md\"]","category":"page"}]
}
