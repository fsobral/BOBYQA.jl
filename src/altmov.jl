# Preliminary functions for BOBYQA algorithm

# Main reference:
# POWELL, M. J. D. (2009). The BOBYQA algorithm for bound constrained optimization without derivatives.
# Cambridge NA Report NA2009/06, University of Cambridge, Cambridge, 26-46

"""

    altmov!(n, m, t, xk, xo, BMAT, ZMAT, Δ, a, b, d, c)

    A version of a Truncated Conjugate Gradient with Active Set algorithm for the
    Trust-Region Subproblem

    - 'n': dimension of the search space
    - 'm': number of interpolation points
    - 'xk': n-dimensional vector (current iterate)
    - 'xo': n-dimensional vector (current origin)
    - 'ik': index of current iterate
    - 't': index of the point that should leave the interpolation set
    - 'BMAT': ((m + n) × n)-dimensional matrix (with information about Ξ and Υ)
    - 'ZMAT': (m × (m - n - 1)-dimensional matrix (with information about Z)
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds 
    
    Modifies the vectors c and d to be the alternative Cauchy and usual directions, respectively.

"""
function altmov!(n, m, xk, xo, ik, t, BMAT, ZMAT, Δ, a, b, d, c)

    # Initializes some variables and vectors
    λ_vec = zeros(m)
    grad_lag = zeros(n)
    w = zeros(n)
    c2 = zeros(n)
    dif = xk - xo
    dif_grad_lag = 0.0
    dif_hess_lag_dif = 0.0
    best_j = 0
    best_α = 0.0
    best_ϕ = 0.0
    best_cond = 0.0

    # Saves g entries from BMAT matrix
    for i = 1:n
        grad_lag[i] = BMAT[t, i]
    end

    # Calculates the lambda values and the gradient of the t-th Lagrange function ∇Λ_{t}(x_{k})
    for i = 1:m
        λ_vec[i] = dot(ZMAT[t, :], ZMAT[i, :])
        grad_lag .+= λ_vec[i] * dot(set[:, i] - xo, dif) * (set[:, i] - xo) 
    end
    
    # Calculates the "Usual" Alternative Step d
    for j = 1:m
        if j == ik
            continue
        end
        dif = set[:, j] - xk

        # Calculates (y_{j} - x_{k})^{T}∇Λ_{t}(x_{k})  and (y_{j} - x_{k})^{T}∇Λ_{t}^{2}(x_{k})(y_{j} - x_{k})
        dif_grad_lag = dot(dif, grad_lag)
        dif_hess_lag_dif = 0.0
        for i = 1:m
            dif_hess_lag_dif += λ_vec[i] * dot(dif, set[:, j] - xo) ^ 2.0
        end

        # Calculates the value of the abscissa associated with the vertex of the parabola ϕ_{j}(α)
        α_vertex = ( - 2.0 * dif_grad_lag ) / dif_hess_lag_dif
        
        # Trust-region bounds
        α_upper = Δ / norm(dif)
        α_lower = - α_upper

        # Adjusts the trust region bounds to the box constraints.
        for i = 1:n
            if dif[i] > 0.0
                α_lower = max(α_lower, (a[i] - xk[i]) / (dif[i]))
                α_upper = min(α_upper, (b[i] - xk[i]) / (dif[i]))
            elseif dif[i] < 0.0
                α_lower = max(α_lower, (b[i] - xk[i]) / (dif[i]))
                α_upper = min(α_upper, (a[i] - xk[i]) / (dif[i]))
            end
        end

        # Evaluates the value of ϕ(α)
        ϕ_α_vertex = α_vertex * dif_grad_lag + ( α_vertex ^ 2.0 / 2.0 ) * dif_hess_lag_dif
        ϕ_α_lower = α_lower * dif_grad_lag + ( α_lower ^ 2.0 / 2.0 ) * dif_hess_lag_dif
        ϕ_α_upper = α_upper * dif_grad_lag + ( α_upper ^ 2.0 / 2.0 ) * dif_hess_lag_dif

        # Sets the optimal α value
        if ( α_lower < α_vertex ) && ( α_vertex < α_upper )
            if ( abs(ϕ_α_vertex) > abs(ϕ_α_lower) ) && ( abs(ϕ_α_vertex) > abs(ϕ_α_upper) )
                α_j = α_vertex
                ϕ_α_j = ϕ_α_vertex
            elseif ( abs(ϕ_α_lower) > abs(ϕ_α_upper) ) && ( abs(ϕ_α_lower) > abs(ϕ_α_vertex) )
                α_j = α_lower
                ϕ_α_j = ϕ_α_lower
            else
                α_j = α_upper
                ϕ_α_j = ϕ_α_upper
            end
        else
            if abs(ϕ_α_lower) > abs(ϕ_α_upper)
                α_j = α_lower
                ϕ_α_j = ϕ_α_lower
            else
                α_j = α_upper
                ϕ_α_j = ϕ_α_upper
            end
        end

        # Compares the best α_j so far
        cond_α = ( ϕ_α_j ^ 2.0 ) * ( 0.5 * λ_vec[t] * α_j ^ 2.0 * (1.0 - α_j) ^ 2.0 * norm(dif) ^ 4.0 + ϕ_α_j ^ 2.0 )
        if ( best_j == 0 ) || ( cond_α > best_cond )
            best_j = j
            best_α = α_j
            best_ϕ = ϕ_α_j
            best_cond = cond_α
        end 

    end

    # Saves the "Usual" Alternative Step d and the value of Λ_{t}(xk + d)
    d .= best_α * ( set[:, best_j] - xk )
    Λ_xk_d = best_ϕ

    # Calculates the "Cauchy" Alternative Step c

    w .= 0.0
    for i = 1:n
        if grad_lag[i] > 0.0
            w[i] = a[i] - xk[i]
        elseif grad_lag[i] < 0.0
            w[i] = b[i] - xk[i]
        end
    end

    if norm(w) <= Δ
        c .= w
    else
        construct_altmov_cauchy!(n, xk, grad_lag, Δ, a, b, w)
        c .= w
    end

    w .= 0.0
    for i = 1:n
        if grad_lag[i] < 0.0
            w[i] = a[i] - xk[i]
        elseif grad_lag[i] > 0.0
            w[i] = b[i] - xk[i]
        end
    end

    if norm(w) <= Δ
        c2 .= w
    else
        construct_altmov_cauchy!(n, xk, -grad_lag, Δ, a, b, w)
        c2 .= w
    end

    # Calculates which Cauchy step results in the highest value of |Λ_{t}(x_{k} + c)|

    dif_grad_lag = dot(c, grad_lag)
    dif_hess_lag_dif = 0.0
    for i = 1:m
        dif_hess_lag_dif += λ_vec[i] * dot(c, set[:, j] - xo) ^ 2.0
    end
    Λ_xk_c = dif_grad_lag + 0.5 * dif_hess_lag_dif

    dif_grad_lag = dot(c2, grad_lag)
    dif_hess_lag_dif = 0.0
    for i = 1:m
        dif_hess_lag_dif += λ_vec[i] * dot(c2, set[:, j] - xo) ^ 2.0
    end
    Λ_xk_c2 = dif_grad_lag + 0.5 * dif_hess_lag_dif

    if abs(Λ_xk_c2) > abs(Λ_xk_c)
        c .= c2
        Λ_xk_c = Λ_xk_c2
    end
    
    return Λ_xk_d, Λ_xk_c

end