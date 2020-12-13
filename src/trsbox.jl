# TRSBOX: Versão inicial

using LinearAlgebra

include("auxiliary_functions.jl")

"""

    tcg_active_set(gk, Gk, xk, Δ, a, b)

    A version of a Truncated Conjugate Gradient with Active Set algorithm for the
    Trust-Region Subproblem

    - 'gk': n-dimensional vector (gradient of the model calculated in xk)
    - 'Gk': (n × n)-dimensional matrix (hessian of the model calculated in xk)
    - 'xk': n-dimensional vector (current iterate) 
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds 
    
    Returns a n-dimensional vector.

"""
function tcg_active_set(gk, Gk, xk, Δ, a, b)
    n = length(xk)
    α = 0.0 
    index_α = 0 
    index = 0
    aux_1 = 0.0
    aux_2 = 0.0
    aux_3 = 0.0
    aux_4 = 0.0
    aux_vector_1 = zeros(n)
    aux_vector_2 = zeros(n)
    aux_vector_3 = zeros(n)
    aux_vector_4 = zeros(n)

    I = active_set(gk, xk, a, b)
    d = zeros(n)
    s = - projection_active_set(gk, I)

    while true      

        if length(I) == n
            return d
        end

        α, index_α, index = calculate_alpha(gk, Gk, xk, d, s, Δ, a, b)

        d .= d .+ α .* s

        mul!(aux_vector_1, Gk, d)
        aux_vector_1 .+= gk 
        aux_vector_2 .= projection_active_set(aux_vector_1, I)
        aux_1 = dot(aux_vector_2, aux_vector_2)
        aux_2 = dot(d, gk) + dot(d, Gk * d) / 2.0
        
        # α_Delta is chosen
        if index_α == 1
            aux_vector_4 .= projection_active_set(d, I)
            aux_4 = dot(aux_vector_4, aux_vector_4)
            
            if aux_4 * aux_1 - dot(aux_vector_4, aux_vector_2) ^ 2.0 + 1.0e-4 * aux_2 <= 0.0
                return d
            else
                s .= new_search_direction(aux_vector_4, aux_vector_2)
                #Ver
            end

        end

        # α_B is chosen
        if index_α == 2
            push!(I, index)
            
            if sqrt(aux_1) * Δ + 1.0e-2 * aux_2 <= 0.0
                return d
            else
                s .= .- aux_vector_2
                continue
            end

        end

        # α_Q is chosen
        if index_α == 3
            aux_vector_3 .= projection_active_set(gk, I)
            aux_3 = dot(aux_vector_3, aux_vector_3)

            if (sqrt(aux_1) * Δ + 1.0e-2 * aux_2 <= 0.0) || (sqrt(aux_1) - sqrt(aux_3) + 1.0e-2 * aux_2 <= 0.0)
                return d
            else
                #Ver
            end

        end

    end

    return d
end
