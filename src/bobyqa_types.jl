export BOBYQA_Hessian, mul!

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
struct BOBYQA_Hessian{T<:Number}

    # Matrix M is a full matrix
    M :: AbstractMatrix{T}
    # List of vectors Y
    Y :: AbstractMatrix{T}
    # Lagrange multipliers for MFN-models
    μ :: AbstractVector{T}

end

"""

    *(Q::BOBYQA_Hessian, v::AbstractVector)

Calculates the matrix-vector product `Q*v`, where `Q` is a matrix
stored using the BOBYQA format for Hessians. Returns a new vector.

"""
function Base.:*(Q::BOBYQA_Hessian{T}, v::AbstractVector{T}) where {T}

    # Usual multiplication
    qv = Q.M * v

    # Perform μ_j * y_j * y_j^T * v
    @views for j = length(Q.μ)
        y = Q.Y[:, j]
        qv .+= μ[j] * dot(y, v) .* y
    end

    return qv

end


"""

    mul!(Y::AbstractVector, A::BOBYQA_Hessian, b::AbstractVector)

Calculates the matrix-vector product `A*b`, where A is a matrix stored
using the BOBYQA format for Hessians, and stores the result in vector
`Y`. `Y` **cannot point** to any part of `A` or `b`.

"""
function mul!(Y::AbstractVector{T}, A::BOBYQA_Hessian{T}, b::AbstractVector{T}) where {T}

    
    mul!(Y, Q.M, b)

    # Perform μ_j * y_j * y_j^T * v
    @views for j = length(Q.μ)
        y = Q.Y[:, j]
        Y .+= μ[j] * dot(y, v) .* y
    end

    return Y

end
