using Plots, SpectralDistances, DSP
m = examplemodels(2)

plot(m[1])
assignmentplot(m...)

y = randn(10000)
plot(spectrogram(y))
plot(periodogram(y))
plot(randn(ComplexF64, 10), Continuous())
plot(ContinuousRoots(randn(ComplexF64, 3)))
