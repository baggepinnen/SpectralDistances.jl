# This example loads 4 short sound clips, each containing a whistle. It then forms the barycenter of those clips with varying barycentric coordinates and plot all barycenters in a grid.
using SpectralDistances, WAV, LPVSpectral
cd(@__DIR__)
y = map(1:4) do i
    filename = "whistle_$i.wav"
    sound, fs = wavread(filename)
    @show fs
    sound = vec(mean(sound, dims = 2)) .+ 0.005 .* randn.()
end

ml = minimum(length, y)
y  = [y[1:ml] for y in y] # Make sure they are all of equal length
fs = 44100
S  = melspectrogram.(y, 512, nmels = 64, fs = fs, fmin = 2500, fmax = 7000)
plot(plot.(S)..., colorbar = false)
EM = melspectrogram(mean(y), 512, nmels = 64, fs = fs, fmin = 2500, fmax = 7000)

## Plot the barycenter when all weights are equal (the Wasserstein mean)
λ = s1([1, 1, 1, 1])
B = barycenter_convolutional(
    S,
    λ,
    β             = 0.0005,
    tol           = 5e-6,
    iters         = 5000,
    ϵ             = 1e-100,
    dynamic_floor = -25,
    verbose       = true,
)
plot(
    plot(B, title = "Barycenter"),
    plot(EM, title = "Euclidean mean"),
    plot.(S, title = "")...,
    colorbar = false,
    layout   = (3, 2),
    xlabel   = false,
)



## Plot a 5×5 grid of barycenters with different barycentric coordinates.
N_fig = 5
plotopts = (
    colorbar  = false,
    titlefont = font(11),
    xlabel    = false,
    axis      = false,
    ylabel    = false,
)
plot(layout = (N_fig, N_fig))
v = I(4)
for i = 0:N_fig-1, j = 0:N_fig-1

    tx = i / (N_fig - 1)
    ty = j / (N_fig - 1)

    tmp1 = (1 - tx) * v[:, 1] + tx * v[:, 2]
    tmp2 = (1 - tx) * v[:, 3] + tx * v[:, 4]
    λ = Vector((1 - ty) * tmp1 + ty * tmp2)
    sp = i * N_fig + j + 1

    i == 0 && j == 0 && (plot!(S[1], sp=sp; plotopts...); continue)
    i == 0 && j == (N_fig - 1) && (plot!(S[3], sp=sp; plotopts...); continue)
    i == (N_fig - 1) && j == 0 && (plot!(S[2], sp=sp; plotopts...); continue)
    i == (N_fig - 1) && j == (N_fig - 1) && (plot!(S[4], sp=sp; plotopts...); continue)

    B = barycenter_convolutional(
        S,
        λ,
        β             = 0.0005,
        tol           = 1e-4,
        iters         = 5000,
        ϵ             = 1e-100,
        dynamic_floor = -25,
        verbose       = false,
    )
    plot!(
        B;
        sp        = sp,
        title     = @sprintf("%G  %G  %G  %G" , λ...),
        plotopts...
    )
end
current()
