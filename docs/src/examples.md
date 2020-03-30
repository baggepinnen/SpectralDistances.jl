# Examples


## Typical workflows
In this section, we'll demonstrate some common ways of interacting with the package

To quickly try things out, you can generate some example models of signals using [`examplemodels`](@ref) like this `models = examplemodels(10)`.

### Calculate root embeddings from sound files
In this example, we'll read a bunch of sound files and calculate embedding vectors containing information about the poles of estimated rational spectra. These embeddings are useful for classification etc. See the paper for further explanation.

This example makes use of a few other packages, notably [AudioClustering.jl](https://github.com/baggepinnen/AudioClustering.jl) for some convenience functions.
```julia
using Glob, WAV, SpectralDistances
const fs = 44100
using Grep
##
using DSP, LPVSpectral
using AudioClustering

path = "path/to/folder/with/wav-files"

cd(path)
files = glob("*.wav")
labels0 = match.(r"[a-z_]+", files)..:match .|> String # This regex assumes that the files are named in a certain way, you may adopt as needed, or load the labels separately.
ulabels = unique(labels0)
labels = sum((labels0 .== reshape(ulabels,1,:)) .* (1:30)', dims=2)[:]
na = 18 # Order of the models
fitmethod = LS(na=na, λ=1e-5)

models = mapsoundfiles(files, fs) do sound # mapsoundfiles is defined in AudioClustering
    sound = SpectralDistances.bp_filter(sound, (50/fs, 18000/fs)) # prefiltering is a good idea
    SpectralDistances.fitmodel(fitmethod, sound)
end

X = embeddings(models)
```
We now have a matrix `X` with features, we can run clustering on it like this:
```julia
using Clustering
labels,models,X,Z = get_features(trainpath)
cr = kmeans(v1(X,2), 30) # v1 normalizes mean and variance

Plots.plot(
    scatter(threeD(X'), marker_z=labels, m=(2,0.5), markerstrokealpha=0, colorbar=false, title="Correct assignment"),
    scatter(threeD(X'), marker_z=cr.assignments, m=(2,0.5), markerstrokealpha=0, colorbar=false, title="K-means on w assignment"),
    legend=false
)
```
Another clustering approach is to use [`kbarycenters`](@ref), see example in the docstring.


## Nearest Neighbor classification
Here, we will classify a signal based on it's nearest neighbor in a training dataset. The example assumes that the matrix `X` from the previous example is available, and that there is a similar matrix `Xt` created from a test dataset. We will classify the entries in the test set using the entries in the training set. The example also assumes that there are two vectors `labels::Vector{Int}` and `labelst::Vector{Int}` that contain the class labels.
```julia
using AMD # For permutation of the confusion matrix to more easily identity similar classes.
function knn_classify(labels, X, Xt, k)
    N = size(Xt,2)
    y = zeros(Int, N)
    W = fit(Whitening, X)
    X = MultivariateStats.transform(W,X)
    Xt = MultivariateStats.transform(W,Xt)
    tree = NearestNeighbors.KDTree(X)
    for i in 1:N
        inds, dists = knn(tree, Xt[:,i], k)
        mode(labels[inds])
        y[i] = mode(labels[inds])
    end
    y
end
##
yht = knn_classify(labels,X,Xt,1)
@show mean(labelst .== yht) # This is the accuracy
cm = confusmat(30,labelst,yht)
perm = amd(sparse(cm))
cm = cm[perm,perm]
heatmap(cm./sum(cm,dims=2), xlabel="Predicted class",ylabel="True class", title="Confusion Matrix for Test Data")
anns = [(reverse(ci.I)..., text(val,8)) for (ci,val) in zip(CartesianIndices(cm)[:], vec(cm))]
annotate!(anns)
```


## Pairwise distance matrix
Many algorithms make use of a matrix containing all pairwise distances between points. Given a set of models, we can easily obtain such a matrix:
```julia
distance = OptimalTransportRootDistance(domain=Continuous())
D = SpectralDistances.distmat(distance, models)
```
with this matrix, we can, for instance, run clustering:
```julia
using Clustering
cr = hclust(Symmetric(sqrt.(D)))
assignments = cutree(cr,k=30) # k is desired number of clusters
```
Another clustering approach is to use [`kbarycenters`](@ref), see example in the docstring.


## Detection using examples
A measure of distance can be used for detection, by selecting a few positive examples and calculating the distance to the nearest neighbor within these examples from a new query point, a simple example:

```julia
function scorefunction(query_model)
    distance = OptimalTransportRootDistance(domain=Continuous())
    distance_vector = distance.(Ref(query_model),positive_example_models)
    score = minimum(distance_vector)
end
```
This can be made significantly more effective (but less accurate) using the `knn` approach from the [example above](https://baggepinnen.github.io/SpectralDistances.jl/latest/examples/#Nearest-Neighbor-classification-1).


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
savefig("cumulative.svg"); nothing # hide
```

![](cumulative.svg)
