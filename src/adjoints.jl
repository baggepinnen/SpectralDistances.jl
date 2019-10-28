# myfun!(y, x) = ...
#
# function myfun(x)
#     y = make_buffer(myfun, x)
#     myfun!(y, x)
#     return release_buffer(myfun, y, x)
# end
#
isderiving() = false

@adjoint isderiving() = true, _ -> nothing


ZygoteRules.@adjoint function getARregressor(y,na)
    getARregressor((y),na),  function (Δ)
        d = zero(y)
        Δ[1] === nothing || (d[na+1:end] .= Δ[1])

        Δ[2] === nothing || for j in 1:size(Δ[2], 2)
            for i in 1:size(Δ[2], 1)
                di = na+1-j + i-1
                d[di] += Δ[2][i,j]
            end
        end
        (d,nothing)
    end
end

ZygoteRules.@adjoint function getARXregressor(y, u, na::Int, nb)
    @assert nb <= na # This is not a fundamental requirement, but this adjoint does not support it yet.
    getARXregressor((y),(u),na,nb),  function (Δ)
    dy = zero(y)
    du = zero(u)
    Δ[1] === nothing || (dy[na+1:end] .= Δ[1])
    # du[na+1:end] .= Δ[1] #src
    Δ[2] === nothing || for j in 1:size(Δ[2], 2)
        for i in 1:size(Δ[2], 1)
            if j <= na
                dyi = na+1-j + i-1
                dy[dyi] += Δ[2][i,j]
            else
                ju = j -na
                dui = na+1-ju + i-1
                du[dui] += Δ[2][i,j]
            end
        end
    end
    (dy,du,nothing,nothing)
end
end

# ZygoteRules.@adjoint function riroots(p)
#     dp = (p)
#     r = riroots(dp)
#     r, function (Δ)
#         fd = FiniteDifferences.central_fdm(3,1)
#         d = FiniteDifferences.j′vp(fd, riroots, Δ, dp)
#         (d,)
#     end
# end

PolynomialRoots.roots(p) = eigvals(companion(p))


function companion(r)
    A = zeros(length(r)-1, length(r)-1)
    A[diagind(A,-1)] .= 1
    A[:,end] .= .- r[1:end-1]
    A
end

ZygoteRules.@adjoint function roots(p)
    eV = eigen(companion(p))
    eV.values, function (Δ)
        e,V = eV
        d = [-(inv(V)'*Diagonal(Δ)*V')[:,end]; 0]
        (d, )
    end
end

# ZygoteRules.@adjoint function svd(A)
#     s = svd(A)
#     s,  function (Δ)
#         BackwardsLinalg.svd_back(s.U, s.S, s.V, Δ...)
#     end
# end
