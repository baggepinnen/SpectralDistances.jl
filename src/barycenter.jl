function barycenter(d::SinkhornRootDistance,models)
    bc = barycenter(EuclideanRootDistance(domain=domain(d), p=d.p),models)
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    X = [real(bc)'; imag(bc)']
    Y = [[real(r)'; imag(r)'] for r in r]
    a = d.weight(bc)
    b = w
    alg2(X,Y,a,b)
end


function barycenter(d::EuclideanRootDistance,models)
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    bc = map(1:length(r[1])) do pi
        sum(w[pi]*r[pi] for (w,r) in zip(w,r))/sum(w[pi] for w in w)
    end
    ContinuousRoots(bc)
end

function distmat_euclidean(X,Y)
    [sum(abs2, c1-c2) for c1 in eachcol(X), c2 in eachcol(Y)]
end

function alg1(X,Y,â,b,λ=100)
    N = length(Y)
    ã = copy(â)
    for t = 1:1
        β = (t+1)/2
        â = (1-inv(β))*â + inv(β)*ã
        𝛂 = mean(1:N) do i
            M = distmat_euclidean(X,Y[i])
            _,u,v = sinkhorn(M,â,b[i]; iters=100, β=1/λ)
            lu = log.(u .+ 1e-100)
            α = -lu./λ .+ sum(lu)/(λ*length(u))
            α .-= sum(α) # Normalize dual optimum to sum to zero
        end
        ã = 𝛂 # Some prox function. Replace with vanilla GD for now?
        â = (1-inv(β))*â + inv(β)*ã
        â ./= sum(â)
    end
    â
end

# TODO: the problem is that some entries in a becomes negative. Maybe the prox is important. The paper gives a simple prox version

function alg2(X,Y,a,b)
    N = length(Y)
    θ = 0.1
    for i = 1:1
        a = alg1(X,Y,a,b)
        YT = mean(1:N) do i
            M = distmat_euclidean(X,Y[i])
            T,_,_ = sinkhorn(M,a,b[i]; iters=100)
            Y[i]*T'
        end
        X .= (1-θ).*X .+ θ.*YT ./ a'
    end
    X,a
end

using SpectralDistances, Distributions
models = [rand(AR, Uniform(-3,-0.1), Uniform(-5,5), 6) for _ in 1:10]

barycenter(EuclideanRootDistance(domain=SpectralDistances.Continuous(),p=2), models)

barycenter(SinkhornRootDistance(domain=SpectralDistances.Continuous(),p=2), models)
