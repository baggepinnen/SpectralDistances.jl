using Test, SpectralDistances, Distributions
const Continuous = SpectralDistances.Continuous

d = EuclideanRootDistance(domain=Continuous(), weight=unitweight)
models = [AR(ContinuousRoots([-1])), AR(ContinuousRoots([-3]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc)[1] ≈ -2

models = [AR(ContinuousRoots([-im,+im])), AR(ContinuousRoots([-3im,+3im]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc) ≈ [-2im,+2im]


d = EuclideanRootDistance(domain=Continuous())
models = [AR(ContinuousRoots([-1])), AR(ContinuousRoots([-3]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc)[1] ≈ -2

models = [AR(ContinuousRoots([-im,+im])), AR(ContinuousRoots([-3im,+3im]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc) ≈ [-2im,+2im]

models = [AR(ContinuousRoots([-im,+im])), AR(ContinuousRoots([-3im,+2im]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc) ≈ [-2im,+1.5im]


d = EuclideanRootDistance(domain=Continuous(), weight=residueweight)
models = [AR(ContinuousRoots([-5-im,-1+im])), AR(ContinuousRoots([-1-im,-1+im]))]
bc = barycenter(d,models)
@test bc isa AR
@test roots.(Continuous(), bc) ≈ [-1.1538-im,-1+im] rtol=0.01



models = [rand(AR, Uniform(-0.11,-0.1), Uniform(-5,5), 4) for _ in 1:2]

##
Xe = barycenter(EuclideanRootDistance(domain=SpectralDistances.Continuous(),p=2), models)





d = SinkhornRootDistance(domain=SpectralDistances.Continuous(),p=2)
X,a = barycenter(d, models, 10)

scatter(eachrow(X)..., color=:blue)
plot!.(roots.(SpectralDistances.Continuous(),models))#, color=:red)
plot!(Xe, color=:green, m=:cross)



Y = [[1. 1], [2. 2], [3. 3]]
X = [1.1 1.1]
a = ones(2) ./ 2
b = [ones(2) ./2 for _ in eachindex(Y)]
@test alg1(X,Y,a,b;λ=0.1) ≈ a
@show Xo,ao = alg2(X,Y,a,b;λ=0.1, θ=0.5)
@test Xo ≈ [2 2] rtol=1e-2

X = [0. 0.]
@show Xo,ao = alg2(X,Y,a,b;λ=0.1,θ=0.5)
@test Xo ≈ [2 2] rtol=1e-2

X = [1. 3.]
@show Xo,ao = alg2(X,Y,a,b;λ=10, θ=0.5)
@test Xo ≈ [2 2] rtol=1e-2


X = [0. 4.]
@show Xo,ao = alg2(X,Y,a,b;λ=2, θ=0.5)
@test Xo ≈ [2 2] rtol=1e-2



Y = [[1. 1], [2. 2], [3. 3]]
X = [1.1 1.1]
a = ones(2) ./ 2
b = [[0.2, 0.8] for _ in eachindex(Y)]
@show alg1(X,Y,a,b;λ=10)
@show Xo,ao = alg2(X,Y,a,b;λ=1, θ=0.5)
@test Xo ≈ [2 2] rtol=1e-2
@test a ≈ b[1] rtol=1e-2
