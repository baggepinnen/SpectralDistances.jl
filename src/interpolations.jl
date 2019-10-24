
function feedback_interpolate(a1,c1,a2,c2,α)
    sys1 = ss(tf(c1,a1,1))
    sys2 = ss(tf(c2,a2,1))
    K = place(sys1, pole(sys2))
    sysi = ss(sys1.A - α*sys1.B*K, sys1.B, sys1.C, sys1.D, 1) |> tf
    sysi = minreal(sysi)
    ar,cr = denvec(sysi)[], numvec(sysi)[]
    ar,cr
end

function spectral_interpolate(args...)
    error("Not implemented")
end


"""
    slerp(p1, p2, t)

The slerp interpolation at `t` between two vectors.
"""
function slerp(p1,p2,t)
    0 <= t <= 1 || throw(DomainError("Interpolation parameter should be between 0 and 1"))
    m = (1-t)*norm(p1) + t*norm(p2)
    p1 = normalize(p1)
    p2 = normalize(p2)
    d = p1'p2
    if d > 0.99
        return m*((1-t)*p1 + t*p2)
    end
    Ω  = acos(d)
    m*((sin((1-t)*Ω)/sin(Ω))*p1 + (sin(t*Ω)/sin(Ω))*p2)
end

function sinkhorn_interpolate(a1,c1,a2,c2,ai,ci,α)
    error("Update to new interface")
    α == 0 && return a1,c1,0
    α == 1 && return a2,c2,0
    # ai = (1-α)*a1 + α*a2
    # ci  = (1-α)*c1 + α*c2
    distmat = loss_spectral_ot(a1,c1,a2,c2)[2]
    function loss(r)
        ar,cr = r[1:length(ai)], r[length(ai)+1:end]
        ar[1] = 1
        (1-α)*loss_spectral_ot!(distmat,a1,c1,ar,cr) +  α*loss_spectral_ot!(distmat,a2,c2,ar,cr)
    end
    res = Optim.optimize(loss, [ai;ci], BFGS(), Optim.Options(store_trace=false, show_trace=true, iterations=500, allow_f_increases=false, time_limit=20, x_tol=1e-6, f_tol=1e-6), inplace=false, autodiff=:forward)

    ai,ci = res.minimizer[1:length(ai)], res.minimizer[length(ai)+1:end]
    ai[1] = 1
    ai, ci, res
end


function centraldiff(v::AbstractVector)
    dv = diff(v)./2
    a1 = [dv[1];dv]
    a2 = [dv;dv[end]]
    a = a1+a2
end

"""
    interpolator(d::ClosedFormSpectralDistance, A1, A2)

Perform displacement interpolation between two models.
"""
function interpolator(d::ClosedFormSpectralDistance,A1,A2)
    @assert d.p == 2 "Interpolation only supported for p=2, you have p=$(d.p)"
    interval   = (0., d.interval[2])
    @assert (interval[1] == 0 || interval[1] == -interval[2])
    e1   = sqrt(2)/sqrt(spectralenergy(domain(d), A1))
    e2   = sqrt(2)/sqrt(spectralenergy(domain(d), A2))
    f1    = w -> evalfr(domain(d), magnitude(d), w, A1, e1)
    f2    = w -> evalfr(domain(d), magnitude(d), w, A2, e2)
    sol1  = c∫(f1,interval...)
    sol2  = c∫(f2,interval...)
    σ1    = sol1(interval[2]) # The total energy in the spectrum
    σ2    = sol2(interval[2]) # The total energy in the spectrum
    F1(w) = sol1(w)/σ1
    F2(w) = sol2(w)/σ2
    iF1   = inv(F1, interval)
    iF2   = inv(F2, interval)
    function (w,t)
        t == 0 && return map(f1,w)
        t == 1 && return map(f2,w)
        r = tmap(w) do w
            finv(e->(1-t)*iF1(e) + t*iF2(e), w, (0,max(σ1, σ2)))
        end
        centraldiff(r) ./ centraldiff(w)
    end
end

function interpolator(d::EuclideanRootDistance,A1,A2; normalize=false)
    p = d.p
    @assert p == 2 "Interpolation only supported for p=2, you have p=$p"
    RT = domain(d) isa Continuous ? ContinuousRoots : DiscreteRoots
    e1,e2 = roots(domain(d), A1), roots(domain(d), A2)
    I1,I2 = d.assignment(e1, e2)
    e1,e2 = RT(e1[I1]),RT(e2[I2])
    w1,w2 = d.weight(e1), d.weight(e2)
    function (w,t)
        A = AR(RT(((1-t)*w1.*e1 + t*w2.*e2)./((1-t).*w1.+t.*w2)))
        e = normalize ? 1/sqrt(spectralenergy(domain(d), A)) : 1
        tmap1(6,w) do w
            evalfr(domain(d), magnitude(d), w, A, e)
        end
    end
end
