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

    alpha_Delta(d, s, Δ)

    Determines which is the largest value of α such that ||d + αs|| <= Δ is satisfied

    - 'd': n-dimensional vector (direction) 
    - 's': n-dimensional vector (new search-direction) 
    - 'Δ': positive real value (trust-region radius)

    Returns the real value α_Δ

"""
function alpha_Delta(d, s, Δ)
    n = length(xk)
    α = Inf
    α_i = 0.0

    for i=1:n
        if s[i] > 0.0
            α_i = (Δ - d[i]) / s[i]
        elseif s[i] < 0.0
            α_i = (-Δ - d[i]) / s[i]
        else
            α_i = Inf
        end
        if α_i < α
            α = α_i
        end
    end

    return α

end

"""

    alpha_B(xk, d, s, a, b)

    Determines which is the largest value of α such that xk + d + αs satisfies the bounds

    - 'xk': n-dimensional vector (current iterate)
    - 'd': n-dimensional vector (direction) 
    - 's': n-dimensional vector (new search-direction)    
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds

    Returns the real value α_B and the index for which the bound is achieved

"""
function alpha_B(xk, d, s, a, b)
    n = length(xk)
    α = Inf
    α_i = 0.0
    index = 0

    for i=1:n
        if s[i] > 0.0
            α_i = (b[i] - x[i] - d[i]) / s[i]
        elseif s[i] < 0.0
            α_i = (a[i] - x[i] - d[i]) / s[i]
        else
            α_i = Inf
        end
        if α_i < α
            α = α_i
            index = i
        end
    end

    return α, index

end

"""

    alpha_Q(gk, Gk, d, s)

    Determines which is the largest value of α such that Q(xk + d + αs) decreases monotonically

    - 'gk': n-dimensional vector (gradient of the model calculated in xk)
    - 'Gk': n × n matrix (hessian of the model calculated in xk)
    - 'd': n-dimensional vector (direction) 
    - 's': n-dimensional vector (new search-direction)    

    Returns the real value α_Q

"""
function alpha_Q(gk, Gk, d, s)
    sGd = dot(s, Gk * d)
    sGs = dot(s, Gk * s)
    sg = dot(s, gk)

    α_Q = Inf

    if sGs > 0.0
        α_Q = (- sg - sGd) / sGs
    end

    return α_Q

end

"""

    calculate_alpha(gk, Gk, xk, d, s, Δ, a, b)

    Determines the stepsize α, which will be the minimum value between α_Δ, α_B and α_Q

    - 'gk': n-dimensional vector (gradient of the model calculated in xk)
    - 'Gk': n × n matrix (hessian of the model calculated in xk)
    - 'xk': n-dimensional vector (current iterate) 
    - 'd': n-dimensional vector (direction) 
    - 's': n-dimensional vector (new search-direction)
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds    
    
    Returns the real value α, an index that indicates which value was chosen, and the index
    for which the bound is achieved (in the case α = α_B) or 0 for the other cases

"""
function calculate_alpha(gk, Gk, xk, d, s, Δ, a, b)
    v = zeros(3)
    v[1] = alpha_Delta(d, s, Δ)
    v[2], index = alpha_B(xk, d, s, a, b)
    v[3] = alpha_Q(gk, Gk, d, s)
    values = findmin(v)
    if values[2] != 2
        index = 0
    end

    return values[1], values[2], index

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
    
    Returns the real value θ (an angle), and a n-dimensional vector d(θ)

"""
function theta_B(xk, d, proj_d, s, a, b)
    θ = pi / 4
    n = length(xk)
    l = a - xk
    u = b - xk
    d_θ = zeros(n)
    
    while true
        d_θ .= d .- proj_d .+ cos(θ) .* proj_d .+ sin(θ) .* s
        for i = 1:n
            if (l[i] > d_θ[i]) || (u[i] < d_θ[i])
               θ *= 0.9
               continue
            end
        end
    end

    return θ, d_θ
    
end

function theta_Q(gk, Gk, d, proj_d, s)
    dGp = dot(d, Gk * proj_d)
    dGs = dot(d, Gk * s)
    sGs = dot(s, Gk * s)
    sGp = dot(s, Gk * proj_d)
    pGp = dot(proj_d, Gk * proj_d)
    θ = pi / 4

    while true
        sin_θ = sin(θ)
        cos_θ = cos(θ)
        aux = - sin_θ * dGp + cos_θ * dGs - sin_θ * cos_θ * sGs - sin_θ * ( cos_θ - 1.0 ) * pGp + ( - sin_θ ^ 2.0 + cos_θ ^ 2.0 - cos_θ) * sGp

        if aux >= 0.0
            θ *= 0.9
            continue
        end
    end

    return θ

end