# Examples



## The closed-form solution
In this example we will simply visalize two spectra, the locations of their poles and the cumulative spectrum functions.
```@example
using OrdinaryDiffEq, ControlSystems, SpectralDistances, Plots
gr(grid=false)

G1   = tf(1,[1,0.12,1])*tf(1,[1,0.1,0.1])
G2   = tf(1,[1,0.12,2])*tf(1,[1,0.1,0.4])
a1   = denvec(G1)[]
a2   = denvec(G2)[]
n    = length(a1)

f1c  = w -> abs2(1/sum(j->a1[j]*(im*w)^(n-j), 1:n))
f2c  = w -> abs2(1/sum(j->a2[j]*(im*w)^(n-j), 1:n))
sol1 = SpectralDistances.c∫(f1c,0,3π)
sol2 = SpectralDistances.c∫(f2c,0,3π)

fig1 = plot((sol1.t .+ sol1.t[2]).*2π, sqrt.(sol1 ./ sol1[end]), fillrange=sqrt.(sol2(sol1.t) ./ sol2[end]), fill=(0.6,:purple), l=(2,:blue))
plot!((sol2.t .+ sol2.t[2]).*2π, sqrt.(sol2(sol2.t) ./ sol2[end]), l=(2,:orange), xscale=:log10, legend=false, grid=false, xlabel="Frequency", xlims=(1e-2,2pi))

fig2 = bodeplot([G1, G2], exp10.(LinRange(-1.5, 1, 200)), legend=false, grid=false, title="", linecolor=[:blue :orange], l=(2,), plotphase=false)

fig3 = pzmap([G1, G2], legend=false, grid=false, title="", markercolor=[:blue :orange], color=[:blue :orange], m=(2,:c), xlims=(-0.5,0.5))
vline!([0], l=(:black, :dash))
hline!([0], l=(:black, :dash))

plot(fig1, fig2, fig3, layout=(1,3))
savefig("cumulative.png"); nothing # hide
```

![](cumulative.png)
