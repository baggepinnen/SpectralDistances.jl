using SpectralDistances
function barycenter(d::SinkhornRootDistance,models,λ=100)
    bc = barycenter(EuclideanRootDistance(domain=domain(d), p=d.p, weight=residueweight),models).pc

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


function barycenter(d::EuclideanRootDistance,models::AbstractVector)
    r = roots.(SpectralDistances.Continuous(), models)
    w = d.weight.(r)
    bc = map(1:length(r[1])) do pi
        sum(w[pi]*r[pi] for (w,r) in zip(w,r))/sum(w[pi] for w in w)
    end
    AR(ContinuousRoots(bc))
end

function distmat_euclidean(X,Y)
    [mean(abs2, c1-c2) for c1 in eachcol(X), c2 in eachcol(Y)]
end

function alg1(X,Y,â,b;λ=100)
    N = length(Y)
    â = copy(â)
    # fill!(â, 1/N)
    ã = copy(â)
    t = 0
    for outer t = 1:10000
        β = t/2
        â .= (1-inv(β)).*â .+ inv(β).*ã
        𝛂 = mean(1:N) do i
            M = distmat_euclidean(X,Y[i])
            a = SpectralDistances.sinkhorn2(M,â,b[i]; iters=500, λ=λ)[2]
            @assert all(!isnan, a) "Got nan in inner sinkhorn alg 1"
            a
        end

        ã .= â .* exp.(-β.*𝛂 .* 0.001)
        ã ./= sum(ã)
        sum(abs2,â-ã)
        if sum(abs2,â-ã) < 1e-16
            @info "Done at iter $t"
            return â .= (1-inv(β)).*â .+ inv(β).*ã
        end
        â .= (1-inv(β)).*â .+ inv(β).*ã
        # â ./= sum(â)
    end
    @show t
    â
end



function alg2(X,Y,a,b;λ = 100,θ = 0.5)
    N = length(Y)
    a = copy(a)
    ao = copy(a)
    X = copy(X)
    Xo = copy(X)
    fill!(ao, 1/length(ao))
    i = 0
    for outer i = 1:500
        a = alg1(X,Y,ao,b,λ=λ)
        YT = mean(1:N) do i
            M = distmat_euclidean(X,Y[i])
            T,_ = SpectralDistances.sinkhorn2(M,a,b[i]; iters=500, λ=λ)
            @assert all(!isnan, T) "Got nan in sinkhorn alg 2"
            Y[i]*T'
        end
        X .= (1-θ).*X .+ θ.*(YT / Diagonal(a))
        # @show mean(abs2, a-ao), mean(abs2, X-Xo)
        mean(abs2, a-ao) < 1e-8 && mean(abs2, X-Xo) < 1e-8 && break
        copyto!(ao,a)
        copyto!(Xo,X)
        ao ./= sum(ao)
        θ *= 0.99
    end
    @show i
    X,a
end



##
"Sum over j≠i"
function ∑jni(X,i,S,k)
    s = zero(X[1][:,1])
    @inbounds for j in eachindex(X)
        j == i && continue
        s .+= @views X[j][:,S[j][k]]
    end
    s
end

function ISA(X; iters=100, printerval = 10)
    N = length(X)
    d,k = size(X[1])
    σ = [collect(1:k) for _ in 1:N] # let σᵢ = Id, 1 ≤ i ≤ n.
    σ′ = deepcopy(σ)
    for iter = 1:iters
        swaps = 0
        for i = 1:N
            σᵢ = σ[i]
            σᵢ′ = σ′[i]
            for k₁ = 1:k-1, k₂ = k₁+1:k
                if dot(X[i][σᵢ[k₁]], ∑jni(X,i,σ,k₁)) + dot(X[i][σᵢ[k₂]], ∑jni(X,i,σ,k₂)) < dot(X[i][σᵢ[k₂]], ∑jni(X,i,σ,k₁)) + dot(X[i][σᵢ[k₁]], ∑jni(X,i,σ,k₂))
                    σᵢ′[k₁],σᵢ′[k₂] = σᵢ[k₂],σᵢ[k₁] # This line can cause σᵢ′ to not contain all indices 1:k
                    swaps += 1
                end
            end
        end
        σ = deepcopy(σ′) # Update assignment
        iter % printerval == 0 && @show iter, swaps
        swaps == 0 && (return σ)
    end
    σ
end

s1(x) = x ./ sum(x) # Normalize a vector to sum to 1
k = 3
X = s1.([rand(1,k) for i = 1:5])
S = ISA(X, iters=1000, printerval=1)

@test all(all(1:k .∈ Ref(S[i])) for i in eachindex(S)) # test that each assignment vector contains all indices 1:k

# X = s1.([[2. 1], [1. 2], [3. 3]])
# Y = repeat(X,9)
@btime ISA($Y)
