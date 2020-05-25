@recipe function plot(s::DSP.Periodograms.Spectrogram)
    seriestype := :heatmap
    xguide --> "Time"
    yguide --> "Freq"
    s.time, s.freq, log.(s.power)
end

@recipe function plot(s::DSP.Periodograms.Periodogram; normalize=true)
    yguide --> "Power"
    xguide --> "Freq"
    xscale --> :log10
    yscale --> :log10
    if normalize
        y =  s.power[2:end] ./ std(s.power)
    else
        y = s.power[2:end]
    end
    s.freq[2:end], y
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
    xguide --> "Re"
    yguide --> "Im"
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
    title --> "Root map"
    xguide --> "Re"
    yguide --> "Im"
    r.r, domain(r)
end

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

"""
    flowplot(dist,m1, m2)

Plots the poles of two models and the optimal mass flow between them
"""
flowplot
@userplot FlowPlot

@recipe function flowplot(h::FlowPlot)
    dist,m1,m2 = h.args[1:3]

    Γ = transport_plan(dist,m1,m2;d=d)
    e1,e2 = roots.(domain(dist), (m1,m2))
    # display(Γ)

    @series begin
        markerstrokealpha --> 0.1
        label --> "m1"
        seriestype := :scatter
        real.(e1), imag.(e1)
    end
    @series begin
        markerstrokealpha --> 0.1
        label --> "m2"
        seriestype := :scatter
        real.(e2), imag.(e2)
    end
    for i in eachindex(e1), j in eachindex(e2)
        @series begin
            primary := false
            seriestype := :path
            coords = [e1[i], e2[j]]
            linewidth --> 10Γ[i,j]
            linealpha --> sqrt(10Γ[i,j])
            real(coords), imag(coords)
        end
    end
    domain(dist) isa Discrete && @series begin
        l := (:dash, :black)
        primary := false
        cos.(0:0.1:2pi),sin.(0:0.1:2pi)
    end
end


@recipe function plot(m::AbstractModel, w=exp10.(LinRange(-2, log10(pi), 200)); rad=true)
    mag = evalfr.(Continuous(), Identity(), w, m, m.b)
    yscale --> :log10
    xscale --> :log10
    rad || (w = w .* 2π)
    w, sqrt.(mag)
end


@recipe function plot(
    m::TimeVaryingAR,
    w = exp10.(LinRange(-2, log10(pi), 200));
    rad = true,
)
    rad || (w = w .* 2π)
    mag =
        evalfr.(
            Continuous(),
            Identity(),
            w,
            reshape(m.models, 1, :),
            getfield.(m.models, :b)',
        )
    seriestype --> :heatmap
    log.(mag)
end
