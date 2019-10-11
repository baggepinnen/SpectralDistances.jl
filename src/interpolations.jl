

function pz_interpolate(a1,c1,a2,c2,α, ::typeof(hungarian))
    k1 = sum(c1)/sum(a1)
    k2 = sum(c2)/sum(a2)
    ai = (1-α)*a1 + α*a2
    ci  = (1-α)*c1 + α*c2
    ki  = (1-α)*k1 + α*k2
    p1  = a1 |> reverse |> roots .|> log .|> reflectd |> x->sort(x,by=imag)
    p2  = a2 |> reverse |> roots .|> log .|> reflectd |> x->sort(x,by=imag)
    p2 = hungariansort(p1,p2)
    z1 = c1 |> reverse |> roots .|> log .|> reflectd |> x->sort(x,by=imag)
    z2 = c2 |> reverse |> roots .|> log .|> reflectd |> x->sort(x,by=imag)
    z2 = hungariansort(z1,z2)
    pi = (1-α)*p1 + α*p2 .|> exp
    zi  = (1-α)*z1 + α*z2 .|> exp

    ar,cr = roots2poly(pi), roots2poly(zi)
    cr .*= ki*sum(ar)/sum(cr)
    ar,cr
end

function pz_interpolate(a1,c1,a2,c2,α, by=imag)
    k1 = sum(c1)/sum(a1)
    k2 = sum(c2)/sum(a2)
    ai = (1-α)*a1 + α*a2
    ci  = (1-α)*c1 + α*c2
    ki  = (1-α)*k1 + α*k2
    p1  = a1 |> reverse |> roots .|> log .|> reflectd |> x->sort(x,by=by) #.|> exp
    p2  = a2 |> reverse |> roots .|> log .|> reflectd |> x->sort(x,by=by) #.|> exp
    z1 = c1 |> reverse |> roots .|> log .|> reflectd |> x->sort(x,by=by) #.|> exp
    z2 = c2 |> reverse |> roots .|> log .|> reflectd |> x->sort(x,by=by) #.|> exp
    pi = (1-α)*p1 + α*p2 .|> exp
    zi  = (1-α)*z1 + α*z2 .|> exp

    ar,cr = roots2poly(pi), roots2poly(zi)
    cr .*= ki*sum(ar)/sum(cr)
    ar,cr
end

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

function wasserstein_interpolate(a1,c1,a2,c2,α,aw)
    error("Update to new interface")
    # ai = (1-α)*a1 + α*a2
    # ci  = (1-α)*c1 + α*c2
    r1  = a1 |> reverse |> roots .|> log |> x->sort(x,by=imag)
    r2  = a2 |> reverse |> roots .|> log |> x->sort(x,by=imag)
    ri  = ai |> reverse |> roots .|> log |> x->sort(x,by=imag)
    rc1 = c1 |> reverse |> roots .|> log |> x->sort(x,by=imag)
    rc2 = c2 |> reverse |> roots .|> log |> x->sort(x,by=imag)
    rci = ci |> reverse |> roots .|> log |> x->sort(x,by=imag)
    r1  = [r1; rc1]
    r2  = [r2; rc2]
    ri  = [ri; rci]

    function loss(r)
        r = complex.(r[1:end÷2], r[end÷2+1:end])
        # (1-α)*eigval_dist_wass_logmag(r1,r,10) +  α*eigval_dist_wass_logmag(r2,r,10) + 0.01norm(aw-ai)^2 #+ 0.1norm(cw-ci)^2
        (1-α)*eigval_dist_hungarian(r1,r) +  α*eigval_dist_hungarian(r2,r) + 0.01sum(abs, r-ri)^2 #+ 0.1norm(cw-ci)^2
    end
    res = Optim.optimize(loss, [real(ri);imag(ri)], Newton(), Optim.Options(store_trace=false, show_trace=false, iterations=150, allow_f_increases=false, g_tol=1e-10), inplace=false)
    w = complex.(res.minimizer[1:end÷2], res.minimizer[end÷2+1:end])
    ra,rc = w[1:length(a1)-1],w[length(a1):end]
    ar,cr = roots2poly(ra), roots2poly(rc)
    cr .*= sum(ci)/sum(ai)*sum(ar)/sum(cr)
    # ar[1] = 1
    ar, cr, res
end

function interpolator(d::ClosedFormSpectralDistance,A1,A2)
    @assert d.p == 2 "Interpolation only supported for p=2, you have p=$(d.p)"
    interval = d.interval
    f1    = w -> abs2(evalfr(domain(d), w, A1))
    f2    = w -> abs2(evalfr(domain(d), w, A2))
    sol1  = c∫(f1,interval...)
    sol2  = c∫(f2,interval...)
    σ1    = sol1(interval[2]) # The total energy in the spectrum
    σ2    = sol2(interval[2]) # The total energy in the spectrum
    F1(w) = sol1(w)/σ1
    F2(w) = sol2(w)/σ2
    iF1   = inv(F1, interval)
    iF2   = inv(F2, interval)
    (w,t) -> tmap1(6,w) do w
        finv(e->(1-t)*iF1(e) + t*iF2(e), w, (0,1))
    end
end
