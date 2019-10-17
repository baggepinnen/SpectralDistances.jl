@recipe function plot(s::DSP.Periodograms.Spectrogram)
    seriestype := :heatmap
    xlabel --> "Time"
    ylabel --> "Freq"
    s.time, s.freq, log.(s.power)
end

@recipe function plot(s::DSP.Periodograms.Periodogram; normalize=true)
    ylabel --> "Power"
    xlabel --> "Freq"
    xscale --> :log10
    yscale --> :log10
    s.freq .+ s.freq[2], s.power ./ (normalize*std(s.power))
end

function Base.show(io::IO, m::MIME"image/png", anim::Plots.Animation)
    path = mktemp()[1]
    path = path*".gif"
    g = gif(anim, path, fps=15)
    show(io, MIME"image/gif"(), g)
end
Base.showable(::MIME"image/gif", agif::Plots.Animation) = true
Base.showable(::MIME"text/html", agif::Plots.Animation) = true

@recipe function plot(v::Vector{<:Complex})
    @series begin
        primary := true
        seriestype := :scatter
        real.(v), imag.(v)
    end
    d = determine_domain(v)
    if d isa Discrete
        t = LinRange(0, 2pi, 100)
        @series begin
            primary := false
            linestyle := :dash
            linecolor := :black
            cos.(t), sin.(t)
        end
    end

end

using Plots
"""
    plot_assignment(m1, m2)

Plots the poles of two models and the optimal assignment between them
"""
function plot_assignment(m1,m2)
    plot(cos.(0:0.1:2pi),sin.(0:0.1:2pi),l=(:dash, :black), subplot=1, layout=1, lab="")
    scatter!(real.(roots(m1)), imag.(roots(m1)), c=:blue, subplot=1, markerstrokealpha=0.1, label="m1")
    scatter!(real.(roots(m2)), imag.(roots(m2)), c=:red, subplot=1, markerstrokealpha=0.1, label="m2")
    hungariansort(roots(m1), roots(m2))
    xc = [real(roots(m1)) real(roots(m2)) fill(Inf, length(m1.p))]'[:]
    yc = [imag(roots(m1)) imag(roots(m2)) fill(Inf, length(m1.p))]'[:]
    plot!(xc,yc, subplot=1, lab="") |> display
end
