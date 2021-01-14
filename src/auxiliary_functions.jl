using LinearAlgebra

"""

    active_set(gk, xk, a, b)

    Constructs the set of active constraints

    - 'gk': n-dimensional vector (gradient of the model calculated in xk)
    - 'xk': n-dimensional vector (current iterate)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds

    Returns a list with the indices that are fixed at the bounds

"""
function active_set(gk, xk, a, b)
    n = length(xk)
    index_list = []

    for i=1:n
        if ( xk[i] == a[i] && gk[i] >= 0.0 ) || ( xk[i] == b[i] && gk[i] <= 0.0 )
            push!(index_list, i)
        end
    end

    return index_list

end

"""

    projection_active_set(v, index_set)

    Constructs the projection operator from the set of active constraints
    
    - 'v': n-dimensional vector 
    - 'index_set': list with the indices of the active constraints

    Returns a n-dimensional vector, which is the projection of v by the set of active constraints

"""
function projection_active_set(v, index_set)
    n = length(index_set)
    proj_v = copy(v)

    for i=1:n
        proj_v[index_set[i]] = 0.0
    end

    return proj_v

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
    fixed_bounds = []

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
                push!(fixed_bounds, i)
            end
        end
    end
 
    return α, index_α, fixed_bounds

end


"""

    new_search_direction(proj_d, proj_grad)

    Calculate a new search direction for the case α = α_Δ

    - 'proj_d': n-dimensional vector (projection of the direction d)
    - 'proj_grad': n-dimensional vector (projection of gradient of the model calculated in xk + d) 
    
    Returns a n-dimensional vector

"""
function new_search_direction(proj_d, proj_grad)
    dg = dot(proj_d, proj_grad)
    dd = dot(proj_d, proj_d)
    gg = dot(proj_grad, proj_grad)
    α = 0.0
    β = 0.0
    aux = 0.0

    if dg == 0
        β = - sqrt(dd / gg)
    else
        aux = ((dd ^ 2.0 * gg) / (dg ^ 2.0)) - dd
        α = - sqrt(aux * dd) / aux
        β = - α * dd / dg

        if β >= - α * dg / gg
            α = dd / sqrt(aux * dd)
            β = - α * dd / dg
        end
    end

    return α * proj_d + β * proj_grad
    
end

"""

    theta_B(xk, d, proj_d, s, a, b)

    Determines which is the largest value of θ such that xk + d(θ) satisfies the bounds

    - 'xk': n-dimensional vector (current iterate) 
    - 'd': n-dimensional vector (direction)
    - 'proj_d': n-dimensional vector (projection of the direction d) 
    - 's': n-dimensional vector (new search-direction)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds    
    
    Returns the real value θ (an angle), a n-dimensional vector d(θ), and a list of indexes that violate the bound restrictions

"""
function theta_B(xk, d, proj_d, s, a, b)
    θ = pi / 4
    n = length(xk)
    l = a - xk
    u = b - xk
    d_θ = zeros(n)
    index_list = []
    
    while true
        d_θ .= d .- proj_d .+ cos(θ) .* proj_d .+ sin(θ) .* s
        for i = 1:n
            if (l[i] > d_θ[i]) || (u[i] < d_θ[i])
               θ *= 0.9
               continue
            end
        end
    end
    for i = 1:n
        if (l[i] == d_θ[i]) || (u[i] == d_θ[i])
            push!(index_list, i)
        end
    end

    return θ, d_θ, index_list

end

"""

    theta_Q(gk, Gk, xk, d, proj_d, s, a, b)

    Determines which is the largest value of θ such that Q(xk + d(θ)) decreases monotonically

    - 'gk': n-dimensional vector (gradient of the model calculated in xk)
    - 'Gk': n × n matrix (hessian of the model calculated in xk)
    - 'xk': n-dimensional vector (current iterate) 
    - 'd': n-dimensional vector (direction)
    - 'proj_d': n-dimensional vector (projection of the direction d) 
    - 's': n-dimensional vector (new search-direction)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds   
    
    Returns the real value θ (an angle), a n-dimensional vector d(θ), and a list of indexes that violate the bound restrictions

"""
function theta_Q(gk, Gk, xk, d, proj_d, s, a, b)
    dGp = dot(d, Gk * proj_d)
    dGs = dot(d, Gk * s)
    sGs = dot(s, Gk * s)
    sGp = dot(s, Gk * proj_d)
    pGp = dot(proj_d, Gk * proj_d)
    θ = pi / 4
    index_list = []

    while true
        sin_θ = sin(θ)
        cos_θ = cos(θ)
        aux = - sin_θ * dGp + cos_θ * dGs - sin_θ * cos_θ * sGs - sin_θ * ( cos_θ - 1.0 ) * pGp + ( - sin_θ ^ 2.0 + cos_θ ^ 2.0 - cos_θ) * sGp

        if aux >= 0.0
            θ *= 0.9
            continue
        end
    end

    d_θ = d - proj_d + sin_θ * proj_d + cos_θ * s

    for i = 1:n
        if ((a[i] - x[i]) == d_θ[i]) || ((b[i] - x[i]) == d_θ[i])
            push!(index_list, i)
        end
    end

    return θ, d_θ, index_list

end

"""

    calculate_theta(gk, Gk, xk, d, proj_d, s, a, b)

    Determines the direction d_θ, which will be defined by the minimum value between θ_B and θ_Q

    - 'gk': n-dimensional vector (gradient of the model calculated in xk)
    - 'Gk': n × n matrix (hessian of the model calculated in xk)
    - 'xk': n-dimensional vector (current iterate) 
    - 'd': n-dimensional vector (direction)
    - 'proj_d': n-dimensional vector (projection of the direction d)
    - 's': n-dimensional vector (new search-direction)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds    
    
    Returns a n-dimensional vector d(θ), a list of indexes that violate the bound restrictions, and a boolean (true if θ = θ_Q or false otherwise)

"""
function calculate_theta(gk, Gk, xk, d, proj_d, s, a, b)
    θ_B, dθ_B, listθ_B = theta_B(xk, d, proj_d, s, a, b)
    θ_Q, dθ_Q, listθ_Q = theta_Q(gk, Gk, xk, d, proj_d, s, a, b)

    if min(θ_B, θ_Q) == θ_B
        return dθ_B, listθ_B, false
    else
        return dθ_Q, listθ_Q, true
    end
    
end