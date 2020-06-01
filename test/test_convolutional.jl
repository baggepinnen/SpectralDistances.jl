using SpectralDistances, Distances, DSP


@testset "Sinkhorn convolutional" begin
    @info "Testing Sinkhorn convolutional"
    m,n,T = 8, 10, Float64
    for m = 8:12, n = 8:12, T = (Float64, )
        # @show m,n,T

        a = zeros(T, m, n)
        a[2,2] = 1
        b = zeros(T, m, n)
        b[3,3] = 1
        d1 = sinkhorn_convolutional(a,b,τ=1e2)
        b = zeros(T, m, n)
        b[4,4] = 1
        d2 = sinkhorn_convolutional(a,b,τ=1e2)
        @test 2sqrt(d1) ≈ sqrt(d2) rtol = 0.1
        b = zeros(T, m, n)
        b[6,6] = 1
        d3 = sinkhorn_convolutional(a,b,τ=1e2)
        @test 2sqrt(d2) ≈ sqrt(d3) rtol = 0.1

        # Do the same as above but with different log stabilization
        a = zeros(T, m, n)
        a[2,2] = 1
        b = zeros(T, m, n)
        b[3,3] = 1
        d4 = sinkhorn_convolutional(a,b,τ=1e5)
        b = zeros(T, m, n)
        b[4,4] = 1
        d5 = sinkhorn_convolutional(a,b,τ=1e5)
        @test 2sqrt(d4) ≈ sqrt(d5) rtol = 0.1
        b = zeros(T, m, n)
        b[6,6] = 1
        d6 = sinkhorn_convolutional(a,b,τ=1e5)
        @test 2sqrt(d5) ≈ sqrt(d6) rtol = 0.1

        @test d1 ≈ d4 # Test that the result is the same with and without log stabilization.
        @test d2 ≈ d5
        @test d3 ≈ d6
    end
end


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

A1 = spectrogram(randn(1000), 128)
A2 = spectrogram(randn(1000), 128)
A = [A1, A2]
B = barycenter_convolutional(A,λ,β=β, dynamic_floor=-2)
@test B isa typeof(A1)

d = ConvOptimalTransportDistance(β=β, dynamic_floor=-2.0)
B2 = barycenter(d, A)
@test power(B) == power(B2)
@test time(B) == time(B2)

@test d(A1,A2) > d(A1,A1)
@test d(A1,A2) > d(A2,A2)
@test d(A1,A2) - 0.5*(d(A1,A1) + d(A2,A2)) > 0

A3 = spectrogram(randn(10000), 128)

@time D = distance_profile(d, A1, A3, tol=1e-4)
# TODO: write some tests for D

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
