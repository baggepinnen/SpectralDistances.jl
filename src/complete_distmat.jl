import .Convex
using .SCS

"""
    D̃, S = complete_distmat(D, W, λ = 2)

Takes an incomplete squared Euclidean distance matrix `D` and fills in the missing entries indicated by the mask `W`. `W` is a `BitArray` or array of {0,1} with 0 denoting a missing value. Returns the completed matrix and an SVD object that allows reconstruction of the generating point set `X`.

*NOTE* This function is only available after `using Convex, SCS`.

# Arguments:
- `D`: The incomplete matrix
- `W`: The mask
- `λ`: Regularization parameter. A higher value enforces better data fitting, which might be required if the number of entries in `D` is very small.

# Example:
```julia
using Distances
P = randn(2,40)
D = pairwise(SqEuclidean(), P)
W = rand(size(D)...) .> 0.3 # Create a random mask
W = (W + W') .> 0           # It makes sense for the mask to be symmetric
W[diagind(W)] .= true
D0 = W .* D                 # Remove missing entries

D2, S = complete_distmat(D0, W)

@show (norm(D-D2)/norm(D))
@show (norm(W .* (D-D2))/norm(D))
```

The set of points that created `D` can be reconstructed up to an arbitrary rotation and translation, `X` contains the reconstruction in the `d` first rows, where `d` is the dimension of the point coordinates. To reconstruct `X` using `S`, do
```julia
X  = Diagonal(sqrt.(S.S))*S.Vt

# Verify that reconstructed `X` is correct up to rotation and translation
A = [X[1:2,:]' ones(size(D,1))]
P2 = (A*(A \\ P'))'
norm(P-P2)/norm(P) # Should be small
```

Ref: Algorithm 5 from "Euclidean Distance Matrices: Essential Theory, Algorithms and Applications"
Ivan Dokmanic, Reza Parhizkar, Juri Ranieri and Martin Vetterli https://arxiv.org/pdf/1502.07541.pdf
"""
function complete_distmat(D, W, λ=2)
    @assert all(==(1), diag(W)) "The diagonal is always observed and equal to 0. Make sure the diagonal of W is true"
    @assert all(iszero, diag(D)) "The diagonal of D is always 0"
    n = size(D, 1)
    x = -1/(n + sqrt(n))
    y = -1/sqrt(n)
    V = [fill(y, 1, n-1); fill(x, n-1,n-1) + I(n-1)]
    e = ones(n)
    G = Convex.Variable((n-1, n-1))
    B = V*G*V'
    E = diag(B)*e' + e*diag(B)' - 2*B
    problem = Convex.maximize(tr(G)- λ * norm(vec(W .* (E - D))), [G ∈ :SDP])
    Convex.solve!(problem, SCS.Optimizer)
    if Int(problem.status) != 1
        @error problem.status
    end
    B  = Convex.evaluate(B)
    D2 = diag(B)*e' + e*diag(B)' - 2*B
    @info "Data fidelity (norm(W .* (D-D̃))/norm(D))", (norm(W .* (D-D2))/norm(D))
    s  = svd(B)
    D2, s
end
