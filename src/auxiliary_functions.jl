# Auxiliary functions for BOBYQA algorithm

# Main reference:
# POWELL, M. J. D. (2009). The BOBYQA algorithm for bound constrained optimization without derivatives.
# Cambridge NA Report NA2009/06, University of Cambridge, Cambridge, 26-46.

"""

    active_set!(n, g, x, a, b, index_set)

    Constructs the set of active constraints

    - 'n': dimension of the search space
    - 'g': n-dimensional vector (gradient of the model calculated in xk)
    - 'x': n-dimensional vector (current iterate)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds
    - 'index_set': boolean array

    Modify index_set to return a boolean array whose entries will have the value "true" for the indexes 
    that are fixed at the limits 

"""
function active_set!(n, g, x, a, b, index_set)

    for i=1:n
        if ( x[i] == a[i] && g[i] >= 0.0 ) || ( x[i] == b[i] && g[i] <= 0.0 )
            index_set[i] = true
        end
    end

end

"""

    projection_active_set!(v, index_set, proj_v)

    Constructs the projection operator from the set of active constraints
    
    - 'v': n-dimensional vector 
    - 'index_set': boolean array with the indices of the active constraints
    - 'proj_v': n-dimensional auxiliary vector

    Modifies proj_v to become the projection of vector v by the set of active constraints

"""
function projection_active_set!(v, index_set, proj_v)

    copyto!(proj_v, v)
    proj_v[index_set] .= 0.0

end

"""

    update_active_set!(index_set, index_list)

    Checks whether the values in index_list are present in index_set and adds values that are not

    - 'index_set': boolean array with the indices of the active constraints
    - 'index_list': boolean array with the new indices that are active constraints

    Modify index_set to the new index boolean array

"""
function update_active_set!(index_set, index_list)
    
    index_set[index_list] .= true

end

"""

    calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs, index_list)

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
    - 'index_list': boolean array
    
    
    Modifies index_list to become a boolean array with the indexes that violate the bound restrictions 
    (in the case α = α_B) and, returns the real value α and an index that indicates which value was chosen

"""
function calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs, index_list)
    α_values = Inf * ones(3)
    Δ_i = 0.0
    B_i = 0.0
    index_list .= false

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
                index_list[i] = true
            end
        end
    end
 
    return α, index_α

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

    # If P_I(d)'P_I(∇Q(xk+d)) = 0, then the new direction is a multiple of P_I(∇Q(xk+d))
    if isapprox(pdpg, 0.0; atol = eps(Float64), rtol = 0.0)
        β = - sqrt(norm2_proj_d / norm2_proj_grad)
        v .= β .* proj_grad
    else
        aux = ((norm2_proj_d ^ 2.0 * norm2_proj_grad) / (pdpg ^ 2.0)) - norm2_proj_d
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

    stop_condition_theta_B(θ, n, x, d, s, proj_d, a, b)

    Checks the stop criterion for choosing θ, assisting in the approximate calculation of θ_B

    - 'θ': real value (to be tested)
    - 'n': dimension of the search space
    - 'x': n-dimensional vector (current iterate)
    - 'd': n-dimensional vector (direction)
    - 's': n-dimensional vector (new search-direction)
    - 'proj_d': n-dimensional vector (projection of the direction d)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds    

    Return a boolean value

"""
function stop_condition_theta_B(θ, n, x, d, s, proj_d, a, b)
    sin_θ = sin(θ)
    cos_θ = cos(θ)
    
    for i=1:n
        if ((a[i] - x[i] - d[i] + proj_d[i]) > (cos_θ * proj_d[i] + sin_θ * s[i])) || ((b[i] - x[i] - d[i] + proj_d[i]) < (cos_θ * proj_d[i] + sin_θ * s[i]))
            return false
        end
    end

    return true
end

"""

    stop_condition_theta_Q(θ, sg, pdg, sGs, dGs, dGpd, sGpd, pdGpd)

    Checks the stop criterion for choosing θ, assisting in the approximate calculation of θ_Q

    - 'θ': real value (to be tested)
    - 'sg': pre-calculated value of s'g
    - 'pdg': pre-calculated value of pd'g, where pd is the projection of d by I
    - 'sGs': pre-calculated value of s'Gs
    - 'dGs': pre-calculated value of d'Gs
    - 'dGpd': pre-calculated value of d'Gpd, where pd is the projection of d by I
    - 'sGpd': pre-calculated value of s'Gpd, where pd is the projection of d by I
    - 'pdGpd': pre-calculated value of s'Gpd, where pd is the projection of d by I
    - 'd': n-dimensional vector (direction)

    Return a boolean value

"""
function stop_condition_theta_Q(θ, sg, pdg, sGs, dGs, dGpd, sGpd, pdGpd)
    sin_θ = sin(θ)
    cos_θ = cos(θ)

    if (- sin_θ * pdg + cos_θ * sg - sin_θ * dGpd + cos_θ * dGs + sin_θ * cos_θ * sGs - sin_θ * (cos_θ - 1.0) * pdGpd + (cos_θ ^ 2.0 - sin_θ ^ 2.0 - cos_θ) * sGpd) >= 0.0
        return false
    else
        return true
    end
end

"""

    binary_search(lower_value, upper_value, stop_condition, ε)

    performs a binary search in the interval [lower_value, upper_value] in order to satisfy stop_condition with tolerance ε

    - 'lower_value': real value (lower limit of search range)
    - 'upper_value': real value (upper limit of search range)
    - 'stop_condition': function that checks the stop condition
    - 'ε': real value (tolerance)

    Return a real value

"""
function binary_search(lower_value, upper_value, stop_condition, ε)
    if stop_condition(upper_value)
        return upper_value
    else
        while true
            mean_value = (lower_value + upper_value) / 2.0
            if stop_condition(mean_value)
                if (upper_value - mean_value) <= ε
                    return mean_value
                else
                    lower_value = mean_value
                end
            else
                if (mean_value - lower_value) <= ε
                    return mean_value
                else
                    upper_value = mean_value
                end
            end
        end
    end
end

"""

    calculate_theta!(n, x, proj_d, s, a, b, sg, pdg, sGs, dGs, dGpd, sGpd, pdGpd, d, index_list)

    Calculate a rotation and a new direction for the case α = α_Δ

    - 'n': dimension of the search space
    - 'x': n-dimensional vector (current iterate)
    - 'proj_d': n-dimensional vector (projection of the direction d)
    - 's': n-dimensional vector (new search-direction)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds
    - 'sg': pre-calculated value of s'g
    - 'pdg': pre-calculated value of pd'g, where pd is the projection of d by I
    - 'sGs': pre-calculated value of s'Gs
    - 'dGs': pre-calculated value of d'Gs
    - 'dGpd': pre-calculated value of d'Gpd, where pd is the projection of d by I
    - 'sGpd': pre-calculated value of s'Gpd, where pd is the projection of d by I
    - 'pdGpd': pre-calculated value of s'Gpd, where pd is the projection of d by I
    - 'd': n-dimensional vector (direction)
    - 'index_list': boolean array

    Modifies d to become the new direction and index_list to become a boolean array with the active constraints, 
    and return a boolean value (true if θ == θ_Q, false otherwise)

"""
function calculate_theta!(n, x, proj_d, s, a, b, sg, pdg, sGs, dGs, dGpd, sGpd, pdGpd, d, index_list)
    ε = 1.0e-1
    θ_B = 0.0
    θ_Q = 0.0
    index_list .= false
    
    # Computes θ_B
    cond_B(θ) = stop_condition_theta_B(θ, n, x, d, s, proj_d, a, b)
    θ_B = binary_search(0.0, pi / 4.0, cond_B, ε)

    # Computes θ_Q
    cond_Q(θ) = stop_condition_theta_Q(θ, sg, pdg, sGs, dGs, dGpd, sGpd, pdGpd)
    θ_Q = binary_search(0.0, pi / 4.0, cond_Q, ε)

    # Determines the value of Θ
    θ = min(θ_B, θ_Q)

    # Computes d(Θ)
    d .= d .- proj_d + cos(θ) * proj_d + sin(θ) * s

    # Computes the indexes of the fixed bounds,
    for i = 1:n
        if ((a[i] - x[i]) == d[i]) || ((b[i] - x[i]) == d[i])
            index_list[i] = true
        end
    end

    if θ == θ_B
        return false
    else
        return true
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
    if (sqrt(norm2_proj_grad) * Δ + 1.0e-2 * (dg + 0.5 * dGd)) <= 0.0
        return true
    else
        return false
    end
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
    if (1.01 * (dog + 0.5 * doGdo) - dg - 0.5 * dGd) <= 0.0
        return true
    else
        return false
    end
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
    if (norm2_proj_d * norm2_proj_grad - dot(proj_d, proj_grad) ^ 2.0 - 1.0e-4 * (- dg - 0.5 * dGd) ^ 2.0) <= 0.0
        return true
    else
        return false
    end
end