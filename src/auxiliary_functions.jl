
"""

    active_set(g, x, a, b)

    Constructs the set of active constraints

    - 'n': dimension of the search space
    - 'g': n-dimensional vector (gradient of the model calculated in xk)
    - 'x': n-dimensional vector (current iterate)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds

    Returns a list with the indices that are fixed at the bounds

"""
function active_set(n, g, x, a, b)
    index_list = []

    for i=1:n
        if ( x[i] == a[i] && g[i] >= 0.0 ) || ( x[i] == b[i] && g[i] <= 0.0 )
            push!(index_list, i)
        end
    end

    return index_list

end

"""

    projection_active_set!(v, index_set, proj_v)

    Constructs the projection operator from the set of active constraints
    
    - 'v': n-dimensional vector 
    - 'index_set': list with the indices of the active constraints
    - 'proj_v': n-dimensional auxiliary vector

    Modifies proj_v to become the projection of vector v by the set of active constraints

"""
function projection_active_set!(v, index_set, proj_v)
    m = length(index_set)
    copyto!(proj_v, v)

    for i=1:m
        proj_v[index_set[i]] = 0.0
    end

end

"""

    calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs)

    Determines the stepsize α, which will be the minimum value between α_Δ, α_B and α_Q

    - 'n': dimension of the search space
    - 'x': n-dimensional vector (current iterate) 
    - 'd': n-dimensional vector (direction) 
    - 's': n-dimensional vector (new search-direction)
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds
    - 'sg': pre-calculated value of s'g
    - 'sGd': pre-calculated value of s'Gd
    - 'sGs': pre-calculated value of s'Gs
    
    Returns the real value α, an index that indicates which value was chosen, and a list 
    of indexes that violate the bound restrictions (in the case α = α_B) or an empty list for the other cases

"""
function calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs)
    α_values = Inf * ones(3)
    Δ_i = 0.0
    B_i = 0.0
    index_list = []

    # Computes α_Δ and α_B
    for i=1:n
        if s[i] > 0.0
            Δ_i = (Δ - d[i]) / s[i]
            B_i = (b[i] - x[i] - d[i]) / s[i]
        elseif s[i] < 0.0
            Δ_i = (- Δ - d[i]) / s[i]
            B_i = (a[i] - x[i] - d[i]) / s[i]
        else
            Δ_i = Inf
            B_i = Inf
        end
        if Δ_i < α_values[1]
            α_values[1] = Δ_i
        end
        if B_i < α_values[2]
            α_values[2] = B_i
        end
    end

    # Compute α_Q
    if sGs > 0.0
        α_values[3] = - (sg + sGd) / sGs
    end

    # Determines the value of α and the correspond index in the vector α_values
    α, index_α = findmin(α_values)

    # Computes the indexes of the fixed bounds, for the case that α_B is chosen for α.
    if index_α == 2
        for i = 1:n
            if ((a[i] - x[i] - d[i]) == α * s[i]) || ((b[i] - x[i] - d[i]) == α * s[i])
                push!(index_list, i)
            end
        end
    end
 
    return α, index_α, index_list

end

"""

    new_search_direction!(proj_d, proj_grad, norm2_proj_d, norm2_proj_grad, v)

    Calculate a new search direction for the case α = α_Δ

    - 'proj_d': n-dimensional vector (projection of the direction d)
    - 'proj_grad': n-dimensional vector (projection of gradient of the model calculated in xk + d) 
    - 'norm2_proj_d': value of 2-norm of vector proj_d squared.
    - 'norm2_proj_grad': value of 2-norm of vector proj_grad squared.
    - 'v': n-dimensional auxiliary vector

    Modifies v to become the new search direction

"""
function new_search_direction!(proj_d, proj_grad, norm2_proj_d, norm2_proj_grad, v)
    pdpg = dot(proj_d, proj_grad)
    α = 0.0
    β = 0.0
    aux = 0.0

    if pdpg == 0
        β = - sqrt(norm2_proj_d / norm2_proj_grad)
        v .= β .* proj_grad
    else
        aux = ((norm2_proj_d ^ 2.0 * norm2_proj_grad) / (pdpg)) - norm2_proj_d
        α = - sqrt(aux * norm2_proj_d) / aux
        β = - α * norm2_proj_d / pdpg

        # Checks if β < (- α * P_I(d)'P_I(∇Q(xk+d)) / ||P_I(∇Q(xk+d))||^2) 
        if β >= (- α * pdpg / norm2_proj_grad)
            α = norm2_proj_d / sqrt(aux * norm2_proj_d)
            β = - α * norm2_proj_d / pdpg
        end
        
        v .= α .* proj_d .+ β .* proj_grad
    end
    
end

"""

    calculate_theta!(n, x, proj_d, s, a, b, pdGpd, sGs, dGs, dGpd, sGpd, d)

    Calculate a rotation and a new direction for the case α = α_Δ

    - 'n': dimension of the search space
    - 'x': n-dimensional vector (current iterate)
    - 'proj_d': n-dimensional vector (projection of the direction d)
    - 's': n-dimensional vector (new search-direction)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds
    - 'sGs': pre-calculated value of s'Gs
    - 'dGs': pre-calculated value of d'Gs
    - 'dGpd': pre-calculated value of d'Gpd, where pd is the projection of d by I.
    - 'sGpd': pre-calculated value of s'Gpd, where pd is the projection of d by I.
    - 'pdGpd': pre-calculated value of s'Gpd, where pd is the projection of d by I.
    - 'd': n-dimensional vector (direction)

    Modifies d to become the new direction and return a boolean value (true if θ == θ_Q, false otherwise)

"""
function calculate_theta!(n, x, proj_d, s, a, b, sGs, dGs, dGpd, sGpd, pdGpd, d)
    θ_B = 0.0
    θ_Q = 0.0
    index_list = []
    aux_1 = a - x - d + proj_d
    aux_2 = b - x - d + proj_d
    
    # Computes θ_B
    θ = pi / 4
    cos_θ = cos(θ)
    sin_θ = sin(θ)
    while true
        for i = 1:n
            if (aux_1[i] > (cos_θ * proj_d[i] + sin_θ * s[i])) || (u[i] < (cos_θ * proj_d[i] + sin_θ * s[i]))
                θ *= 0.9
                cos_θ = cos(θ)
                sin_θ = sin(θ)
               continue
            end
        end
    end
    θ_B = θ

    # Computes θ_Q
    θ = pi / 4
    cos_θ = cos(θ)
    sin_θ = sin(θ)
    while true
        if (- sin_θ * dGpd + cos_θ * dGs - sin_θ * cos_θ * sGs - sin_θ * ( cos_θ - 1.0 ) * pdGpd + ( - sin_θ ^ 2.0 + cos_θ ^ 2.0 - cos_θ) * sGpd) >= 0.0
            θ *= 0.9
            cos_θ = cos(θ)
            sin_θ = sin(θ)
            continue
        end
    end
    θ_Q = θ

    # Determines the value of Θ
    θ = min(θ_B, θ_Q)

    # Computes d(Θ)
    d .= d .+ x .- proj_d + cos(θ) * proj_d + sin(θ) * s

    # Computes the indexes of the fixed bounds,
    for i = 1:n
        if ((a[i] - x[i]) == d[i]) || ((b[i] - x[i]) == d[i])
            push!(index_list, i)
        end
    end

    if θ == θ_B
        return index_list, false
    else
        return index_list, true
    end

end

"""

    stopping_criterion_34(Δ, norm2_proj_grad, dg, dGd)

    Checks the stopping criterion for choice α = α_B and α = α_Q, present in equation 3.4 of the main reference.

    - 'Δ': positive real value (trust-region radius)
    - 'norm2_proj_grad': value of 2-norm of vector proj_grad squared.
    - 'dg': pre-calculated value of d'g
    - 'dGd': pre-calculated value of d'Gd

    Return a boolean value

"""
function stopping_criterion_34(Δ, norm2_proj_grad, dg, dGd)
    return isapprox(sqrt(norm2_proj_grad) * Δ + 1.0e-2 * (dg + dGd), 0.0; atol = eps(Float64), rtol = 0.0)
end

"""

    stopping_criterion_34_B(dg, dog, dGd, doGdo)

    Checks the additional stopping criterion for choice α = α_Q and α = α_Δ, present in pages 10 and 11 of the main reference.

    - 'dg': pre-calculated value of d'g
    - 'dog': pre-calculated value of do'g, where do is the old value of d.
    - 'dGd': pre-calculated value of d'Gd
    - 'doGdo': pre-calculated value of do'Gdo, where do is the old value of d.

    Return a boolean value

"""
function stopping_criterion_34_B(dg, dog, dGd, doGdo)
    return isapprox(1.01 * (dog + doGdo) - dg - dGd, 0.0; atol = eps(Float64), rtol = 0.0)
end

"""

    stopping_criterion_35(proj_d, proj_grad, norm2_proj_d, norm2_proj_grad, dg, dGd)

    Checks the stopping criterion for choice α = α_Δ, present in equation 3.5 of the main reference.

    - 'proj_d': n-dimensional vector (projection of the direction d)
    - 'proj_grad': n-dimensional vector (projection of gradient of the model calculated in xk + d) 
    - 'norm2_proj_d': value of 2-norm of vector proj_d squared.
    - 'norm2_proj_grad': value of 2-norm of vector proj_grad squared.
    - 'dg': pre-calculated value of d'g
    - 'dGd': pre-calculated value of d'Gd

    Return a boolean value

"""
function stopping_criterion_35(proj_d, proj_grad, norm2_proj_d, norm2_proj_grad, dg, dGd)
    return isapprox(norm2_proj_d * norm2_proj_grad - dot(proj_d, proj_grad) ^ 2.0 + 1.0e-4 * (dg + dGd), 0.0; atol = eps(Float64), rtol = 0.0)
end