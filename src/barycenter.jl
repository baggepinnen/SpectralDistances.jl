function barycenter(d::SinkhornRootDistance,models)
    bc = barycenter(EuclideanRootDistance(domain=domain(d), p=d.p),models)
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
end


function barycenter(d::EuclideanRootDistance,models)
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    bc = map(1:length(r[1])) do pi
        sum(w[pi]*r[pi] for (w,r) in zip(w,r))/sum(w[pi] for w in w)
    end
    ContinuousRoots(bc)
end

using SpectralDistances, Distributions
models = [rand(AR, Uniform(-3,-0.1), Uniform(-5,5), 6) for _ in 1:10]

barycenter(EuclideanRootDistance(domain=SpectralDistances.Continuous(),p=2), models)

barycenter(SinkhornRootDistance(domain=SpectralDistances.Continuous(),p=2), models)



function alg1(X,Y,â,b)
    ã = copy(â)
    for t = 1:10
        β = (t+1)/2
        a = (1-inv(β))*â + inv(β)*ã
        𝛂 = mean(1:N) do i
            M = SpectralDistances.distmat_euclidean(X,Y[i])
            _,u,v = sinkhorn(M,a,b[i]; iters=100)
            lu = log.(u)
            α = -lu./λ .+ sum(lu)/(λ*length(u))
        end
        ã = Pα # Some prox function. Replace with vanilla GD for now?
        â = (1-inv(β))*â + inv(β)*ã

    end
    â
end

function alg2(X,Y,a,b)
    θ = 0.1
    for i = 1:10
        a = alg1(X,Y,a,b)
        YT = mean(1:N) do i
            M = SpectralDistances.distmat_euclidean(X,Y[i])
            T,_,_ = sinkhorn(M,a,b[i]; iters=100)
            Y[i]*T'
        end
        X .= (1-θ).*X .+ θ.*YT ./ a'
    end
    X,a
end
