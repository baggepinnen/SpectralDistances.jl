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
    seriestype := :scatter
    real.(v), imag.(v)
end
