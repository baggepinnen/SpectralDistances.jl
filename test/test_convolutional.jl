using SpectralDistances, Distances, DSP


@testset "Sinkhorn convolutional" begin
    @info "Testing Sinkhorn convolutional"
    m,n,T = 8, 10, Float64
    for m = 8:2:12, n = 8:2:12, T = (Float64, )
        # @show m,n,T

        a = zeros(T, m, n)
        a[2,2] = 1
        b = zeros(T, m, n)
        b[3,3] = 1
        d1 = sinkhorn_convolutional(a,b,τ=1e2)[1]
        b = zeros(T, m, n)
        b[4,4] = 1
        d2 = sinkhorn_convolutional(a,b,τ=1e2)[1]
        @test 2sqrt(d1) ≈ sqrt(d2) rtol = 0.1
        b = zeros(T, m, n)
        b[6,6] = 1
        d3 = sinkhorn_convolutional(a,b,τ=1e2)[1]
        @test 2sqrt(d2) ≈ sqrt(d3) rtol = 0.1

        # Do the same as above but with different log stabilization
        a = zeros(T, m, n)
        a[2,2] = 1
        b = zeros(T, m, n)
        b[3,3] = 1
        d4 = sinkhorn_convolutional(a,b,τ=1e5)[1]
        b = zeros(T, m, n)
        b[4,4] = 1
        d5 = sinkhorn_convolutional(a,b,τ=1e5)[1]
        @test 2sqrt(d4) ≈ sqrt(d5) rtol = 0.1
        b = zeros(T, m, n)
        b[6,6] = 1
        d6 = sinkhorn_convolutional(a,b,τ=1e5)[1]
        @test 2sqrt(d5) ≈ sqrt(d6) rtol = 0.1

        @test d1 ≈ d4 # Test that the result is the same with and without log stabilization.
        @test d2 ≈ d5
        @test d3 ≈ d6
    end
end




@testset "Invariant axis" begin
    @info "Testing Invariant axis"
    w,h = 5,5
    for w = [5, 6, 20], h = [5,6,20]
        @show w,h
        β = 0.05
        D = [abs2((i-j)/(h-1)) for i = 1:h, j = 1:h]
        B = [zeros(h,w) for _ in 1:2]

        B[1][1,1] = 1
        B[1][1,5] = 1
        B[2][2,1] = 1
        B[2][2,5] = 1

        B = s1.(B)
        c,V,U = sinkhorn_convolutional(B[1], B[2], β=β, τ=1e20)
        c1,V1,U1 = sinkhorn_convolutional(B[1], B[1], β=β, τ=1e20)
        c2,V2,U2 = sinkhorn_convolutional(B[2], B[2], β=β, τ=1e20)
        c3 = sqrt((c - 0.5(c1+c2))) # The size of the summed over dimension needs to multiply here
        v,u = sum(B[1], dims=2)[:], sum(B[2], dims=2)[:] # Integrate over the dimension to be sensitive to (because we need to be invariant to this dimension in this step)
        @assert sum(u) ≈ 1
        @assert sum(v) ≈ 1
        Γ = discrete_grid_transportplan(v, u)    # Solve 1D OT between integrated measures
        invariant_cost = sqrt(dot(Γ, D))

        @test c3 ≈ invariant_cost rtol=0.15

        if isinteractive()
            Plots.heatmap(Γ, layout=5, title="Transport plan")
            Plots.heatmap!(V, sp=2, title="V")
            Plots.heatmap!(U, sp=3, title="U")
            Plots.plot!(v, sp=4, lab=invariant_cost)
            Plots.plot!(u, sp=5, lab=c3)
        end


        invariant_cost = dot(Γ, D)
        di = ConvOptimalTransportDistance(β=β, invariant_axis=1)
        d = ConvOptimalTransportDistance(β=β, invariant_axis=0)
        cdist = d(B[1], B[2]) - 0.5*(d(B[1], B[1]) + d(B[2], B[2]))
        @test cdist ≈ c3^2
        ci = (di(B[1], B[2]) - 0.5*(di(B[1], B[1]) + di(B[2], B[2])))

        @test ci ≈ cdist-invariant_cost rtol=0.15 atol=sqrt(eps())

        @test di(B[1], B[2]) ≈ d(B[1], B[2])-invariant_cost rtol=0.15

    end
end



@testset "Misc convolutional" begin
    @info "Testing Misc convolutional"

    @test -0.72 < SpectralDistances.default_dynamic_floor(rand(100,100)) < -0.6
    a1 = zeros(Float64, 10,10)
    a1[2,2] = 1
    a2 = zeros(Float64, 10,10)
    a2[6,6] = 1
    A = [a1,a2]
    β = 0.001
    λ = [0.5, 0.5]
    b = barycenter_convolutional(A,λ,β=β)
    @test maximum(b) == b[4,4]
    @test sum(b) ≈ 1

    d = ConvOptimalTransportDistance(β=0.01)
    l = barycentric_coordinates(d, A, a1)[1]
    @test l ≈ [1, 0]
    @test sum(l) ≈ 1
    l = barycentric_coordinates(d, A, a2)[1]
    @test l ≈ [0, 1]

    l = barycentric_coordinates(d, A, b)[1]
    @test l ≈ [0.5, 0.5]


    A1 = spectrogram(sin.(1:10000) .+ 0.1randn(10000), 256, window=hanning)
    A2 = spectrogram(sin.(1:10000) .+ 0.1randn(10000), 256, window=hanning)
    A = [A1, A2]

    @test -5 < SpectralDistances.default_dynamic_floor(A1) < -4
    @test -5 < SpectralDistances.default_dynamic_floor(A) < -4

    o = similar(A1.power)
    @test normalize_spectrogram!(o,A1) == normalize_spectrogram(A1)


    B = barycenter_convolutional(A,λ,β=β, dynamic_floor=-3)
    @test B isa typeof(A1)

    d = ConvOptimalTransportDistance(β=β, dynamic_floor=-3.0)
    B2 = barycenter(d, A)
    @test power(B) == power(B2)
    @test time(B) == time(B2)

    @test d(A1,A2) > d(A1,A1)
    @test d(A1,A2) > d(A2,A2)
    @test d(A1,A2) - 0.5*(d(A1,A1) + d(A2,A2)) > 0

    A3 = spectrogram(sin.(LinRange(0.8,1.2,300_000) .* (1:300_000)) .+ 0.01randn(300_000), 256, window=hanning)

    d = ConvOptimalTransportDistance(β=0.05, dynamic_floor=-3.0)
    @time D = distance_profile(d, A1, A3, tol=1e-6, stride=15)
    isinteractive() && plot(D)
    @test 40 <= argmin(D) <= 70





end


# @test B.power == barycenter_convolutional([A1.power, A2.power],λ,β=β)


# w = SpectralDistances.BCWorkspace(A, β)
# @btime SpectralDistances.barycenter_convolutional($w,$A,$λ)
# 1.198 ms (46 allocations: 1.26 MiB)
# 585.176 μs (46 allocations: 1.26 MiB) # add @avx
# 469.132 μs (16 allocations: 313.13 KiB) # operate inplace
# 387.881 μs (0 allocations: 0 bytes) # views and generator in sum




##
# m,n = 1280,1280
# x = zeros(m,n)
# x[10,10] = 1
# β = 0.0003 # The threshold for imfilter to be faster is around 0.0003 on size 128
#
# t = LinRange(0, 1, m)
# Y, X = meshgrid(t, t)
# xi1 = @. exp(-(X - Y)^2 / β)
#
# t = LinRange(0, 1, n)
# Y, X = meshgrid(t, t)
# xi2 = @. exp(-(X - Y)^2 / β)
#
# K1(x) = xi1 * x * xi2
#
#
# using ImageFiltering
#
# t = LinRange(-0.5, 0.5, m)
# k1 = @. exp(-t^2 / β)
# t = LinRange(-0.5, 0.5, n)
# k2 = @. exp(-t^2 / β)
#
# function kfg(x)
#     l = round(Int,5x)
#     l = isodd(l) ? l : l+1
#     KernelFactors.gaussian(x,l)
# end
#
# kk1 = kfg(sqrt(0.5β)*m)
# kk2 = kfg(sqrt(0.5β)*n)
# kk1,kk2 = kk1 ./ maximum(kk1), kk2 ./ maximum(kk2)
# # kernf = kernelfactors((centered(k1), centered(k2)))
#
# # kernf = KernelFactors.gaussian((sqrt(β)*m, sqrt(β)*m))
# kernf = kernelfactors((kk1, kk2))
# kernp = broadcast(*, kernf...)
#
# CPU1(Algorithm.FIR())
# K2(x) = imfilter(x, kernf)
#
# plot(heatmap(K1(x)), heatmap(K2(x)), heatmap(K2(x)-K1(x)))
#
# using FFTW
# FFTW.set_num_threads(2)
#
#
# using ImageFiltering.ComputationalResources
# @btime $xi1 * $x * $xi2
# @btime imfilter!($(similar(x)), $x, $kernf)
#
# using LoopVectorization
# function filter2davx!(out::AbstractMatrix, A::AbstractMatrix, kerns)
#     for kern in kerns
#         @avx for J in CartesianIndices(out)
#             tmp = zero(eltype(out))
#             for I ∈ CartesianIndices(kern)
#                 tmp += A[I + J] * kern[I]
#             end
#             out[J] = tmp
#         end
#     end
#     out
# end
#
# out = similar(x)
#
# out = ImageFiltering.OffsetArray(similar(x, size(x).-2), 1, 1)
#
# ka = convert(AbstractArray, kernf[1])
# kb = convert(AbstractArray, kernf[2])
#
# @btime filter2davx!($out, $x, $((ka, kb)));
#
# plot(heatmap(K1(x)), heatmap(K2(x)), heatmap(K2(x)-K1(x)), heatmap(out-K1(x)))
