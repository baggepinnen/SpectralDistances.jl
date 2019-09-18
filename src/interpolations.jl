
function pz_interpolate(a1,c1,a2,c2,α, ::typeof(hungarian))
    k1 = sum(c1)/sum(a1)
    k2 = sum(c2)/sum(a2)
    ai = (1-α)*a1 + α*a2
    ci  = (1-α)*c1 + α*c2
    ki  = (1-α)*k1 + α*k2
    p1  = a1 |> reverse |> roots .|> log .|> reflect |> x->sort(x,by=imag)
    p2  = a2 |> reverse |> roots .|> log .|> reflect |> x->sort(x,by=imag)
    p2 = hungariansort(p1,p2)
    z1 = c1 |> reverse |> roots .|> log .|> reflect |> x->sort(x,by=imag)
    z2 = c2 |> reverse |> roots .|> log .|> reflect |> x->sort(x,by=imag)
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
    p1  = a1 |> reverse |> roots .|> log .|> reflect |> x->sort(x,by=by) #.|> exp
    p2  = a2 |> reverse |> roots .|> log .|> reflect |> x->sort(x,by=by) #.|> exp
    z1 = c1 |> reverse |> roots .|> log .|> reflect |> x->sort(x,by=by) #.|> exp
    z2 = c2 |> reverse |> roots .|> log .|> reflect |> x->sort(x,by=by) #.|> exp
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
