
[![Build Status](https://travis-ci.org/baggepinnen/SpectralDistances.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/SpectralDistances.jl)
[![codecov](https://codecov.io/gh/baggepinnen/SpectralDistances.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/SpectralDistances.jl)



# SpectralDistances

This package facilitates the calculation of distances between signals, primarily in the frequency domain. The main functionality revolves around *rational spectra*, i.e., the spectrum of a rational function, commonly known as a transfer function.

This repository implements all optimal-transport based distances between spectra detailed in the following pre-pre print

[New Metrics Between Rational Spectra and their Connection to Optimal Transport, Bagge Carlson 2019](https://drive.google.com/file/d/1EPS_pyC_opKMLlnk02kIfHbpawWFl4W-/view?usp=sharing)

## High-level overview
The main workflow is as follows
1. Define a distance
2. Evaluate the distance between two points (signals, histograms, periodograms, etc.)

You can also calculate barycenters, interpolations, and barycentric coordinates of signals under the chosen distance.

This package extends [Distances.jl](https://github.com/JuliaStats/Distances.jl) and all distance types are subtypes of `Distances.PreMetric`, even though some technically are true metrics and some are not even pre-metrics.

Many distances are differentiable and can thus be used for gradient-based learning. The rest of this manual is divided into the following sections

## Contents
```@contents
Pages = ["distances.md", "ltimodels.md", "interpolations.md", "plotting.md", "misc.md", "examples.md"]
```


## All Exported functions and types
```@index
Pages = ["distances.md", "ltimodels.md"]
```
