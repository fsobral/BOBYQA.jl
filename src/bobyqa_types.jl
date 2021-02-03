"""

    BOBYQA_Hessian{T}

Immutable type representing the way that BOBYQA stores the Hessian of
quadratic models, that is

    Q = M + ∑ μ_i (y_i - x0) (y_i - x0)^T

where `M` is a full `n x n` matrix, `μ` represents the `m` Lagrange
multipliers associated with the solution of the minimum Frobenius-norm
interpolation problem, `Y = [y_1 ... y_n]` is a matrix whose columns
are the interpolation points of the model and `x0` is the shift (not
necessarily the current point).

"""
struct BOBYQA_Hessian

    # Matrix M is a full matrix
    M :: AbstractMatrix
    # Lagrange multipliers for MFN-models
    μ :: AbstractVector
    # List of vectors Y
    Y :: AbstractMatrix
    # Center of the model
    x0 :: AbstractVector

    # Constructor for checking dimensions
    BOBYQA_Hessian(M, μ, Y, x0) = begin

        nm, mm = size(M)

        (nm != mm) && throw(DimensionMismatch("Matrix M should be square."))
        
        mmu = length(μ)

        ((size(Y) != (nm, mmu)) || (length(x0) != nm)) && throw(DimensionMismatch("The dimension of the elements does not match."))

        new(M, μ, Y, x0)
        
    end
                               
end

"""

    BOBYQA_Hessian(n, m)

Creates an empty structure, ready to be filled.

"""
BOBYQA_Hessian(n, m) = BOBYQA_Hessian(Matrix{Float64}(undef, n, n),
                                      Vector{Float64}(undef, m),
                                      Matrix{Float64}(undef, n, m),
                                      zeros(n))

"""

    *(Q::BOBYQA_Hessian, v::AbstractVector)

Calculates the matrix-vector product `Q*v`, where `Q` is a matrix
stored using the BOBYQA format for Hessians. Returns a new vector.

"""
function Base.:*(Q::BOBYQA_Hessian, v::AbstractVector)

    # Usual multiplication
    qv = Q.M * v

    return add_rank1_mul!(qv, Q, v)

end

"""

    mul!(Y::AbstractVector, A::BOBYQA_Hessian, b::AbstractVector)

Calculates the matrix-vector product `A*b`, where A is a matrix stored
using the BOBYQA format for Hessians, and stores the result in vector
`Y`. `Y` **cannot point** to any part of `A` or `b`.

"""
function LinearAlgebra.mul!(Y::AbstractVector, A::BOBYQA_Hessian, b::AbstractVector)
    
    LinearAlgebra.mul!(Y, A.M, b)

    return add_rank1_mul!(Y, A, b)
    
end

"""

    add_rank1_mul!(Y, Q, v)

Auxiliary function which computes product of `v` with the sum of
rank-1 matrices given by the shifted interpolation points in `Q.Y`, as
described in (`mul!`)[ref] and [`*`](ref).

This function assumes that vector `Y` has already been initialized
with well defined values, since it is added inside.

"""
function add_rank1_mul!(Y, Q, v)

    # Store x0^T * v
    x0tv = dot(Q.x0, v)
    
    # Perform μ_j * y_j * y_j^T * v
    @views for j = 1:length(Q.μ)
        y = Q.Y[:, j]
        Y .= Y .+ (Q.μ[j] * (dot(y, v) - x0tv)) .* (y .- Q.x0)
    end

    return Y

end

# Export functions
export BOBYQA_Hessian, mul!
