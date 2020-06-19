# Task View

This page breaks functionality up according to tasks in order to make it easier to find relevant functions

## Classification
There are two main ways to perform classification using the functionality in this package
- Nearest-neighbor based
- Feature based

### Nearest-neighbor based classification
This is very simple, select a distance, and calculate the nearest neighbor from a dataset to your query.

First, we demonstrate how one can do "leave-one-out" prediction within a labeled dataset, i.e., for each example, classify it using all the others.

Any distance can be used to calculate a distance matrix [`distmat`](@ref). Given a distance matrix `D`, you can predict the nearest neighbor class with the following function
```julia
function predict_nn(labels, D)
    D = copy(D)
    D[diagind(D)] .= Inf # The diagonals contain trivial matches
    dists, inds = findmin(D, dims=2)
    inds = vec(getindex.(inds, 2))
    @show size(inds), size(labels)
    yh = map(i->labels[i], inds)
end

predicted_classes = predict_nn(labels, D)
```

When we want to classify a new sample `q`, we can simply broadcast a distance `d` between `q` and all labeled samples in the training set
```julia
dists = d.(models_train, q)
predicted_class = labels[argmin(dists)]
```

By far the fastest neighbor querys can be made by extracting embeddings from estimated models and using a KD-tree to accelerate neigbor searches. Below, we'll go into detail on how to do this. This corresponds to using the [`EuclideanRootDistance`](@ref) without weighting.

The following function finds you the `k` most likely classes corresponding to query embedding `q` from within `Xtrain`. `Xtrain` and `q` are expected to be embeddings formed by the function [`embeddings`](https://github.com/baggepinnen/AudioClustering.jl#estimating-linear-models) from AudioClustering.jl.
```julia
using MultivariateStats, NearestNeighbors, AudioClustering

Xtrain = embeddings(models_train)

function knn_classify(labels, Xtrain, q, k)
    N = size(Xtrain,2)
    W = fit(Whitening, Xtrain)
    X = MultivariateStats.transform(W,Xtrain)
    q = MultivariateStats.transform(W,q)
    tree = NearestNeighbors.KDTree(Xtrain)
    inds, dists = knn(tree, q, min(5k+1, N-1), true)
    ul = unique(labels[inds[2:end]])
    ul[1:min(k, length(ul))]
end
```

Increased performance is often obtained by estimating models with a few different specifications and fitting methods and use them all to form predictions. The following code fits models with different fit methods and of different orders
```julia
using ThreadTools, AudioClustering
modelspecs = collect(Iterators.product(10:2:14, (TLS,LS))) # Model order × fitmethod

manymodels = @showprogress "Estimating models" map(modelspecs) do (na, fm)
    fitmethod = fm(na=na, λ=1e-5)
    tmap(sounds) do sound
        sound = @view(sound[findfirst(!iszero, sound):findlast(!iszero, sound)])
        sound = Float32.(SpectralDistances.bp_filter(sound, (50 / fs, 0.49)))
        fitmethod(sound)
    end
end

manyX = embeddings.(manymodels)
```

To predict a single class, let many classifiers vote for the best class
```julia
using MLBase # For mode

function vote(preds)
    map(1:length(preds[1])) do i
        mode(getindex.(preds, i))
    end
end

votes = [classpred1, classpred2, classpred3, ...]
majority_vote = vote(votes)
@show mean(labels .== majority_vote) # Accuracy
```

To predict "up to k classes", try he following

```julia
using StatsBase # for countmap
function predict_k(labels, preds, k)
    map(eachindex(labels)) do i
        cm = countmap(getindex.(preds, i)) |> collect |> x->sort(x, by=last, rev=true)
        first.(cm[1:min(k,length(cm))])
    end
end

votes = [classpred1, classpred2, classpred3, ...]
k_votes = predict_k(labels, votes, k)
@show mean(labels .∈ k_votes) # Accuracy
```

To figure out which classifier is best, rank them like so

```julia
function ranking(labels, preds)
    scores = [mean(labels .== yh) for yh in preds]
    sortperm(scores, rev=true)
end

votes = [classpred1, classpred2, classpred3, ...]
r = ranking(labels, votes)
```

### Feature-based classification


## Unsupervised learning

## Dimensionality reduction

## Dataset augmentation
