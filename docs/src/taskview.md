# Task View

This page breaks functionality up according to tasks in order to make it easier to find relevant functions. Throughout this page, we assume that you are familiar with how to estimate models ([Models and root manipulations](@ref)) and specify distances ([Distances](@ref)).

## Classification
There are two main ways to perform classification using the functionality in this package
- Nearest-neighbor based
- Feature based

### Nearest-neighbor based classification
This is very simple, select a distance, and calculate the nearest neighbor from a dataset to your query. The dataset can be either a vector of models estimated from signals, a vector of spectrograms, or a matrix of embedding vectors derived from models.

First, we demonstrate how one can perform "leave-one-out" corss validation within a labeled dataset, i.e., for each example, classify it using all the others. Since a distance based classifier does not have an explicit "training phase", this sort of cross-validation is comparatively cheap to perform.

Any distance can be used to calculate a distance matrix using the function [`distmat`](@ref). Given a distance matrix `D`, you can predict the nearest-neighbor class with the following function
```julia
function predict_nn(labels::Vector{Int}, D)
    dists, inds = findmin(D + Inf*I, dims=2) # The diagonal contains trivial matches, hence add infinite Identity
    inds = vec(getindex.(inds, 2))
    map(i->labels[i], inds)
end

predicted_classes = predict_nn(labels, D)
```

When we want to *classify a new sample* `q`, we can simply broadcast[^tmap] a distance `d` between `q` and all labeled samples in the training set
```julia
dists = d.(models_train, q)
predicted_class = labels[argmin(dists)]
```
note that if you're doing *detection*, i.e., looking for a short `q` in a much longer time series, see [Detection](@ref) below, and the function [`distance_profile`](@ref).


[^tmap]: If `d` is an expensive distance to compute, you may want to consider using `tmap` from [ThreadTools.jl](https://github.com/baggepinnen/ThreadTools.jl instead. If you do, make sure you copy the distance object to each thread, in case it contains an internal cache.)


#### Nearest neighbor using embeddings
By far the fastest neighbor querys can be made by extracting embeddings from estimated models and using a KD-tree to accelerate neigbor searches. Below, we'll go into detail on how to do this. This corresponds to using the [`EuclideanRootDistance`](@ref) with uniform weighting on the poles.

The following function finds you the ``k`` most likely classes corresponding to query embedding `q` from within `Xtrain`. `Xtrain` and `q` are expected to be embeddings formed by the function [`embeddings`](https://github.com/baggepinnen/AudioClustering.jl#estimating-linear-models) from [AudioClustering.jl](https://github.com/baggepinnen/AudioClustering.jl). (See [Calculate root embeddings from sound files](@ref) for an intro.)
```julia
using MultivariateStats, NearestNeighbors, AudioClustering

Xtrain = embeddings(models_train) # Assumes that you have already estimated models

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

Increased accuracy is often obtained by estimating models with a few different specifications and fitting methods and use them all to form predictions (this will form an ensemble). The following code fits models with different fit methods and of different orders
```julia
using ThreadTools, AudioClustering, ProgressMeter
modelspecs = collect(Iterators.product(10:2:14, (TLS,LS))) # Model order × fitmethod

manymodels = @showprogress "Estimating models" map(modelspecs) do (na, fm)
    fitmethod = fm(na=na, λ=1e-5)
    tmap(sounds) do sound
        sound = @view(sound[findfirst(!iszero, sound):findlast(!iszero, sound)])
        sound = Float32.(SpectralDistances.bp_filter(sound, (50 / fs, 0.49))) # Apply some bandpass filtering
        fitmethod(sound)
    end
end

manyX = embeddings.(manymodels) # This is not a matrix of matrices
```

To predict a single class, let many classifiers vote for the best class
```julia
using MLBase # For mode

function vote(preds)
    map(1:length(preds[1])) do i
        mode(getindex.(preds, i))
    end
end

votes = [classpred1, classpred2, classpred3, ...] # Each classpred can be obtained by, e.g., knn_classify above.
majority_vote = vote(votes)
@show mean(labels .== majority_vote) # Accuracy
```

To predict "up to ``k`` classes", try the following

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

The embeddings extracted above can be used as features for a standard classifier. Below we show an example using a random forest

```julia
using DecisionTree, MultivariateStats, Random, AudioClustering
N       = length(labels)
X       = embeddings(models)' |> copy # DecisionTree expects features along columns
perm    = randperm(N)
Nt      = N ÷ 2 # Use half dataset for training
train_x = X[perm[1:Nt], :]
train_y = labels[perm[1:Nt]]
test_x  = X[perm[Nt+1:end], :]
test_y  = labels[perm[Nt+1:end]]

model = RandomForestClassifier(n_trees=400, max_depth=15)
DecisionTree.fit!(model, train_x, train_y)

predictions = DecisionTree.predict(model, test_x)
k_predictions =
getindex.(sortperm.(eachrow(predict_proba(model, test_x)), rev = true), Ref(1:3)) # Predict top 3
@show accuracy = mean(predictions .== test_y)    # Top class prediction accuracy
@show accuracy = mean(test_y .∈ k_predictions)  # Top 3 classes predictions accuracy
```
The features derived here can of course be combined with any number of other features, such as from [AcousticFeatures.jl](https://github.com/ymtoo/AcousticFeatures.jl/).



## Detection
Detection refers to finding a short query pattern `q` in a long recording `y`. This task can often be performance optimized for expensive-to-compute distances.

In its most basic form, a dection score can be calculated by simply broadcasting a distance over `y`, see [Detection using examples](@ref).

For spectrogram distances, we have optimized methods for calculating distance profiles, see
[Computing a spectrogram distance profile](@ref). Also [`TimeDistance`](@ref) has an optimized method for [`distance_profile`](@ref).

Detection can also be done using Dynamic Time Warping combined with optimal transport, see [Dynamic Time Warping](@ref). For examples of the combination of DTW and OT, see the following notebooks
- [DTW-OT: Introduction](https://nbviewer.jupyter.org/github/baggepinnen/julia_examples/blob/master/frequency_warping.ipynb)
- [DTW-OT: Detection](https://nbviewer.jupyter.org/github/baggepinnen/julia_examples/blob/master/frequency_warping2.ipynb)


## Unsupervised learning
For clustering applications, there are a number of approaches
- Distance matrix
- Feature-based
- K-barycenters

### Clustering using a distance matrix
Using [`distmat`](@ref) with keyword arg `normalize=true`, you can obtain a distance matrix that can be used with a large number of clustering algorithms from [Clustering.jl](https://juliastats.org/Clustering.jl/stable/index.html) or [HDBSCAN.jl](https://github.com/baggepinnen/HDBSCAN.jl).

### Clustering using features
Using [`embeddings`](https://github.com/baggepinnen/AudioClustering.jl#estimating-linear-models) from AudioClustering.jl, you can run regular K-means which is blazingly fast, but often produces worse clusterings than more sophisticated methods.

### Clustering using K-barycenters
This approach is similar to K-means, but uses a transport-based method to calculate distances and form averages rather than the Euclidean distance. See the example [K-Barycenters](@ref).

### Finding motifs or outliers
To find motifs (recurring patterns) or outliers (discords), see [MatrixProfile.jl](https://github.com/baggepinnen/MatrixProfile.jl) which interacts well with SpectralDistances.jl.

## Dimensionality reduction
Several sounds from the same class can be reduced to a smaller number of sounds by forming a barycenter. See examples [Barycenters](@ref), [Barycenters between spectrograms](@ref) and the figure in the [readme](https://github.com/baggepinnen/SpectralDistances.jl) (reproduced below) which shows how four spectrograms can be used to calculate a "center spectrogram".

![window](https://github.com/baggepinnen/SpectralDistances.jl/blob/master/examples/barycenters.png?raw=true)



## Dataset augmentation
Barycenters can also be used also to augment datasets with points "in-between" other points. The same figure in the [readme](https://github.com/baggepinnen/SpectralDistances.jl) (reproduced above) illustrates how four spectrogams are extended into 25 spectrograms.


## Interpolation between spectra
An interpolation between spectra is obtained by calculating a [`barycenter`](@ref) using varying barycentric coordinates. See [Interpolations and Barycenters](@ref).
