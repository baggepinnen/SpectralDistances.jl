"""
    twoD(X)

Project `X` to two dmensions using PCA
"""
function twoD(X)
    X = X .- mean(X,dims=1)
    X = X ./ std(X,dims=1)
    s = svd(X)
    Y = s.U[:,1:2]#.*s.S[1:2]'
    Y[:,1], Y[:,2]
end

"""
    threeD(X)

Project `X` to three dimensions using PCA
"""
function threeD(X)
    X = X .- mean(X,dims=1)
    X = X ./ std(X,dims=1)
    s = svd(X)
    Y = s.U[:,1:3]#.*s.S[1:2]'
    Y[:,1], Y[:,2], Y[:,3]
end

"""
    s1(x, dims=:)

normalize x sums to 1
"""
s1(x, dims=:) = x./sum(x, dims=dims)
"""
    n1(x)

normalize x norm 1
"""
n1(x) = x./norm(x)
n1(x::AbstractMatrix, dims=:) = mapslices(n1, x, dims=dims)
"""
    v1(x, dims=:)

normalize x var 1
"""
function v1(x, dims=:)
    x = x .- mean(x, dims=dims)
    x .= x./std(x, dims=dims)
end

"""
    median1(x, dims=:)

normalize x median 0, median absolute deviation = 1
"""
function m1(x, dims=:)
    x = x .- median(x, dims=dims)
    # x .= x./mapslices(mad, x, dims=dims)
end
function m1(x::AbstractVector)
    x = x .- median(x)
    x .= x./mad(x, normalize=true)
end


"""
    bp_filter(x, passband)

Band-pass filter, passband is tuple, `fs` assumed = 1
"""
function bp_filter(x, passband)
    responsetype = Bandpass(passband..., fs=1)
    designmethod = Butterworth(2)
    filt(digitalfilter(responsetype, designmethod), x)
end

"""
    lp_filter(x, cutoff)

Low-pass filter, `fs` assumed = 1
"""
function lp_filter(x, cutoff)
    responsetype = Lowpass(cutoff, fs=1)
    designmethod = Butterworth(2)
    filt(digitalfilter(responsetype, designmethod), x)
end
