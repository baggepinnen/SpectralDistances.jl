using SpectralDistances
function barycenter(d::SinkhornRootDistance,models,λ=100)
    bc = barycenter(EuclideanRootDistance(domain=domain(d), p=d.p),models)
    # bc = roots(SpectralDistances.Continuous(), models[rand(1:length(models))])
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    X = [real(bc)'; imag(bc)']
    Y = [[real(r)'; imag(r)'] for r in r]
    a = d.weight(bc)
    @assert sum(a) ≈ 1
    @assert all(sum.(w) .≈ 1)
    b = w
    alg2(X,Y,a,b,λ=λ)
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
    [mean(abs2, c1-c2) for c1 in eachcol(X), c2 in eachcol(Y)]
end

function alg1(X,Y,â,b;λ=100)
    N = length(Y)
    â = copy(â)
    # fill!(â, 1/N)
    ã = copy(â)
    for t = 1:100
        β = t/2
        â .= (1-inv(β)).*â .+ inv(β).*ã
        𝛂 = mean(1:N) do i
            M = distmat_euclidean(X,Y[i])
            a = SpectralDistances.sinkhorn2(M,â,b[i]; iters=500, λ=λ)[2]
            @assert all(!isnan, a) "Got nan in inner sinkhorn alg 1"
            a
        end

        ã .= ã .* exp.(-β.*𝛂)
        ã ./= sum(ã)
        if sum(abs2,â-ã) < 1e-8
            @info "Done at iter $t"
            return â .= (1-inv(β)).*â .+ inv(β).*ã
        end
        â .= (1-inv(β)).*â .+ inv(β).*ã
        # â ./= sum(â)
    end
    â
end



function alg2(X,Y,a,b;λ=100)
    N = length(Y)
    θ = 0.01
    ao = copy(a)
    Xo = copy(X)
    fill!(ao, 1/length(ao))
    for i = 1:50
        a = alg1(X,Y,ao,b,λ=λ)
        YT = mean(1:N) do i
            M = distmat_euclidean(X,Y[i])
            T,_ = SpectralDistances.sinkhorn2(M,a,b[i]; iters=500, λ=λ)
            @assert all(!isnan, T) "Got nan in sinkhorn alg 2"
            Y[i]*T'
        end
        X .= (1-θ).*X .+ θ.*YT / Diagonal(a)
        mean(abs2, a-ao) < 1e-10 && break
        mean(abs2, X-Xo) < 1e-8 && break
        copyto!(ao,a)
        copyto!(Xo,X)
        ao ./= sum(ao)
    end
    X,a
end

using SpectralDistances, Distributions
models = [rand(AR, Uniform(-1,-0.1), Uniform(-5,5), 4) for _ in 1:2]

##
Xe = barycenter(EuclideanRootDistance(domain=SpectralDistances.Continuous(),p=2), models)

X,a = barycenter(SinkhornRootDistance(domain=SpectralDistances.Continuous(),p=2), models, 0.1)

scatter(eachrow(X)..., color=:blue)
plot!.(roots.(SpectralDistances.Continuous(),models))#, color=:red)
plot!(Xe, color=:green, m=:cross)
