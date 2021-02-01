"""

    BOBYQA_Hessian{T}

Immutable type representing the way that BOBYQA stores the Hessian of
quadratic models, that is

    Q = M + ∑ μ_i y_i y_i^T

where `M` is a full `n x n` matrix, `μ` represents the `m` Lagrange
multipliers associated with the solution of the minimum Frobenius-norm
interpolation problem and `Y = [y_1 ... y_n]` is a matrix whose
columns are the interpolation points of the model.

"""
struct BOBYQA_Hessian

    # Matrix M is a full matrix
    M :: AbstractMatrix
    # Lagrange multipliers for MFN-models
    μ :: AbstractVector
    # List of vectors Y
    Y :: AbstractMatrix

    # Constructor for checking dimensions
    BOBYQA_Hessian(M, μ, Y) = size(Y) != (size(M)[1], length(μ)) ? throw(DimensionMismatch("The dimension of the elements does not match.")) : new(M, μ, Y)
                               
end

BOBYQA_Hessian(n, m) = BOBYQA_Hessian(Matrix{Float64}(undef, n, n),
                                      zeros(Float64, m),
                                      Matrix{Float64}(undef, n, m))

"""

    *(Q::BOBYQA_Hessian, v::AbstractVector)

Calculates the matrix-vector product `Q*v`, where `Q` is a matrix
stored using the BOBYQA format for Hessians. Returns a new vector.

"""
function Base.:*(Q::BOBYQA_Hessian, v::AbstractVector)

    # Usual multiplication
    qv = Q.M * v
    
    # Perform μ_j * y_j * y_j^T * v
    @views for j = 1:length(Q.μ)
        y = Q.Y[:, j]
        qv .= qv .+ Q.μ[j] .* dot(y, v) .* y
    end

    return qv

end


"""

    mul!(Y::AbstractVector, A::BOBYQA_Hessian, b::AbstractVector)

Calculates the matrix-vector product `A*b`, where A is a matrix stored
using the BOBYQA format for Hessians, and stores the result in vector
`Y`. `Y` **cannot point** to any part of `A` or `b`.

"""
function LinearAlgebra.mul!(Y::AbstractVector, A::BOBYQA_Hessian, b::AbstractVector)
    
    LinearAlgebra.mul!(Y, A.M, b)

    # Perform μ_j * y_j * y_j^T * v
    @views for j = 1:length(A.μ)
        y = A.Y[:, j]
        Y .+= A.μ[j] * dot(y, b) .* y
    end

    return Y

end

# Export functions
export BOBYQA_Hessian, mul!
