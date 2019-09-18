

distmat_euclidean(e1::AbstractArray,e2::AbstractArray) = abs2.(e1 .- transpose(e2))
distmat_euclidean(e1,e2) = (n = length(e1[1]); [(e1[1][i]-e2[1][j])^2 + (e1[2][i]-e2[2][j])^2 for i = 1:n, j=1:n])

distmat_logmag(e1::AbstractArray,e2::AbstractArray) = distmat_euclidean(logmag.(e1), logmag.(e2))

function eigval_dist_hungarian(e1::AbstractVector{<:Complex},e2::AbstractVector{<:Complex})
    dm    = distmat_euclidean(e1,e2)
    hungarian(dm)[2]
end


function eigval_dist_hungarian(e1,e2)
    e1,e2 = toreim(e1), toreim(e2)
    n = length(e1[1])
    dm = [(e1[1][i]-e2[1][j])^2 + (e1[2][i]-e2[2][j])^2 for i = 1:n, j=1:n]
    c = hungarian(dm)[2]
    # P,c = hungarian(Flux.data.(dm))
    # mean([(e1[1][i]-e2[1][j])^2 + (e1[2][i]-e2[2][j])^2 for i = 1:n, j=P])
end

function kernelsum(e1,e2,λ)
    s = 0.
    λ = -λ
    @inbounds for i in eachindex(e1)
        s += exp(λ*abs(e1[i]-e2[i])^2)
        for j in i+1:length(e2)
            s += 2exp(λ*abs(e1[i]-e2[j])^2)
        end
    end
    s / length(e1)^2
end

function logkernelsum(e1,e2,λ)
    s = 0.
    λ = -λ
    le2 = logreim.(e2)
    @inbounds for i in eachindex(e1)
        le1 = logreim(e1[i])
        s += exp(λ*abs2(le1-le2[i]))
        for j in i+1:length(e2)
            s += 2exp(λ*abs2(le1-le2[j]))
        end
    end
    s / length(e1)^2
end

function eigval_dist_euclidean(e1,e2)
    e1 = sort(e1, by=imag)
    e2 = sort(e2, by=imag)
    sum(abs, e1-e2)
end

function eigval_dist_wass_logmag_defective(e1,e2,λ=1)
    error("this does not yield symmetric results")
    # e1 = [complex((logmag(magangle(e1)))...) for e1 in e1]
    # e2 = [complex((logmag(magangle(e2)))...) for e2 in e2]
    # e1    = logreim.(e1)
    # e2    = logreim.(e2)
    # e1    = sqrtreim.(e1)
    # e2    = sqrtreim.(e2)
    dm1   = logkernelsum(e1,e1,λ)
    dm2   = logkernelsum(e2,e2,λ)
    dm12  = logkernelsum(e1,e2,λ)
    dm1 - 2dm12 + dm2
end
function eigval_dist_wass_logmag(e1,e2,λ=1)
    e1    = logreim.(e1)
    e2    = logreim.(e2)
    dm1   = exp.(.- λ .* distmat_euclidean(e1,e1))
    dm2   = exp.(.- λ .* distmat_euclidean(e2,e2))
    dm12  = exp.(.- λ .* distmat_euclidean(e1,e2))
    mean(dm1) - 2mean(dm12) + mean(dm2)
end
function eigval_dist_wass(e1,e2,λ=1)
    e1,e2 = toreim(e1), toreim(e2)
    dm1   = exp.(.- λ .* distmat_euclidean(e1,e1))
    dm2   = exp.(.- λ .* distmat_euclidean(e2,e2))
    dm12  = exp.(.- λ .* distmat_euclidean(e1,e2))
    mean(dm1) - 2mean(dm12) + mean(dm2)
end


# NOTE λ for wass dist can be used as annealing parameter
# TODO: figure out weighted wass dist. Maybe change the kernel function to accomodate for this

function ls_loss_eigvals_disc(X,Xh,order,λ)
    r = ar(X, order) |> polyroots
    rh = ar(Xh, order) |> polyroots
    eigval_dist_wass(r,rh,λ)
end


function ls_loss_eigvals_cont(X,Xh,order,λ)
    r = ar(X, order) |> polyroots .|> log
    rh = ar(Xh, order) |> polyroots .|> log
    # weighted_eigval_dist_hungarian(r,rh)
    eigval_dist_wass(r,rh,λ)
end

function ls_loss_eigvals_cont_logmag(X,Xh,order,λ)
    r = ar(X, order) |> polyroots .|> log
    rh = ar(Xh, order) |> polyroots .|> log
    # weighted_eigval_dist_hungarian(r,rh)
    eigval_dist_wass_logmag(r,rh,λ)
end

function ls_loss(X,Xh,order)
    w = ar(X, order)
    wh = ar(Xh, order)
    sqrt(mean(abs2,w-wh))
end

function plr_loss(X,Xh,na, nc, initial_order=100)
    a,c = plr(X, na, nc, initial_order=initial_order)
    ah,ch = plr(Xh, na, nc, initial_order=initial_order)
    sqrt(mean(abs2,a-ah)) + sqrt(mean(abs2,c-ch))
end

function ls_loss_angle(X,Xh,order)
    w = ar(X, order)
    wh = ar(Xh, order)
    (1-w'wh/norm(w)/norm(wh))
end

function batch_loss(bs::Int, loss, X, Xh)
    l = zero(eltype(Xh))
    lx = length(X)
    n_batches = length(X)÷bs
    inds = 1:bs
    # TODO: introduce overlap for smoother transitions  #src
    for i = 1:n_batches
        l += loss(X[inds],Xh[inds])
        inds = inds .+ bs
    end
    l *= bs
    residual_inds = inds[1]:lx
    lr = length(residual_inds)
    lr > 0 && (l += loss(X[residual_inds],Xh[residual_inds])*lr)
    l /= length(X)
    l / n_batches
end

energyloss(X,Xh) = (std(X)-std(Xh))^2 + (mean(X)-mean(Xh))^2



function loss_spectral_ot(a1,c1,a2,c2,w=exp10.(LinRange(1, log10(fs::Int/2), 300)))
    n = length(w)
    distmat = zeros(n,n)
    for i=1:n
        for j=1:n
            distmat[i, j] = abs((i/n)-(j/n))
        end
    end
    loss_spectral_ot!(distmat,a1,c1,a2,c2,w), distmat
end
function loss_spectral_ot!(distmat,a1,c1,a2,c2,w=exp10.(LinRange(1, log10(fs::Int/2), 300)))
    n = length(w)
    noise_model = tf(c1,a1,1/fs)
    b1,_,_ = bode(noise_model, w.*2pi) .|> vec
    b1 .= log.(b1)
    b1 .-= minimum(b1)
    b1 ./= sum(b1)
    noise_model = tf(c2,a2,1/fs)
    b2,_,_ = bode(noise_model, w.*2pi) .|> vec
    b2 .= log.(b2)
    b2 .-= minimum(b2)
    b2 ./= sum(b2)
    plan = sinkhorn_plan(distmat, b1, b2; ϵ=1e-2, rounds=300)
    # plan = sinkhorn_plan_log(distmat, b1, b2; ϵ=1e-2, rounds=300)
    cost = sum(plan .* distmat)
end
