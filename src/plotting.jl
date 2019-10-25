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

@recipe function plot(v::Vector{<:Complex}, d = determine_domain(v); circle=true)
    @series begin
        primary := true
        seriestype := :scatter
        real.(v), imag.(v)
    end
    if circle && d isa Discrete
        t = LinRange(0, 2pi, 100)
        @series begin
            seriestype := :path
            marker := false
            primary := false
            linestyle := :dash
            linecolor := :black
            cos.(t), sin.(t)
        end
    end
end

@recipe function plot(r::AbstractRoots)
    r.r, domain(r)
end

using Plots
"""
    assignmentplot(m1, m2)

Plots the poles of two models and the optimal assignment between them
"""
assignmentplot
@userplot AssignmentPlot

@recipe function assignmentplot(h::AssignmentPlot; d=Discrete(),p=2)
    m1,m2 = h.args[1:2]
    rf(a) = roots(d, a)
    r1 = (rf(m1))
    r2 = hungariansort(r1, (rf(m2)), p)
    @series begin
        markerstrokealpha --> 0.1
        label --> "m1"
        seriestype := :scatter
        real.(r1), imag.(r1)
    end
    @series begin
        markerstrokealpha --> 0.1
        label --> "m2"
        seriestype := :scatter
        real.(r2), imag.(r2)
    end
    @series begin
        primary := false
        seriestype := :path
        xc = [real.(r1) real.(r2) fill(Inf, length(m1.p))]'[:]
        yc = [imag.(r1) imag.(r2) fill(Inf, length(m1.p))]'[:]
        xc,yc
    end
    d isa Discrete && @series begin
        l := (:dash, :black)
        primary := false
        cos.(0:0.1:2pi),sin.(0:0.1:2pi)
    end
end


@recipe function plot(m::AbstractModel, w=exp10.(LinRange(-2, log10(pi), 200)))
    mag = evalfr.(Continuous(), Identity(), w, m)
    yscale --> :log10
    xscale --> :log10
    w, sqrt.(mag)
end
