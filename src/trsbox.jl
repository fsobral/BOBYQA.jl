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
function trsbox(n, gk, Gk, xk, Δ, a, b)

    #Initializes some variables and vectors.
    α = 0.0
    β = 0.0
    index_α = 0
    dg = 0.0
    dog = 0.0
    sg = 0.0
    sGd = 0.0
    sGs = 0.0
    dGd = 0.0
    doGdo = 0.0
    norm2_proj_d = 0.0
    norm2_proj_grad_xd = 0.0
    #index = 0
    #aux_1 = 0.0
    #aux_2 = 0.0
    #aux_3 = 0.0
    #aux_4 = 0.0
    #aux_vector_1 = zeros(n)
    #aux_vector_2 = zeros(n)
    #aux_vector_3 = zeros(n)
    #aux_vector_4 = zeros(n)
    d = zeros(n)
    s = zeros(n)
    d_old = zeros(n)
    Gd = zeros(n)
    Gdo = zeros(n)
    Gs = zeros(n)
    grad_xd = zeros(n)
    proj_d = zeros(n)
    proj_grad_xd = zeros(n)


    # Calculates the set of active restrictions and the first search direction
    I = active_set(n, gk, xk, a, b)
    projection_active_set!(gk, I, s)
    s .= .-s

    # Stop Criteria evaluation
    if length(I) == n
        msg = "All bounds restrictions are active."
        return d, msg
    end

    # Stop Criteria evaluation
    if dot(s, s) == 0.0
        msg = "The initial search direction is null."
        return zeros(n), msg
    end

    # Note that, for the first iteration, we have sGd = 0.0
    sg = dot(s, gk)
    mul!(Gs, Gk, s)
    sGs = dot(s, Gs)

    while true         

        # Computes the index α
        α, index_α, indexes = calculate_alpha(n, xk, d, s, Δ, a, b, sg, sGd, sGs)

        # Computes the new direction d
        d_old .= d
        d .+= α .* s

        # Computes some curvatures and save old information about it
        dog = dg
        Gdo .= Gd
        doGdo = dGd
        dg = dot(d, gk)
        mul!(Gd, Gk, d)
        sGd = dot(s, Gd)
        dGd = dot(d, Gd)

        # Computes ∇Q(xk + d), P_I(∇Q(xk + d)) and ||P_I(∇Q(xk + d))||^2     
        grad_xd .= gk .+ Gd
        projection_active_set!(grad_xd, I, proj_grad_xd)
        norm2_proj_grad_xd = dot(aux_vector_2, aux_vector_2)

        # Computes P_I(d) and ||P_I(d)||^2
        projection_active_set!(d, I, proj_d)
        norm2_proj_d = dot(proj_d, proj_d)
        
        # α_Δ is chosen
        if index_α == 1
            while true

                if (norm2_proj_d * norm2_proj_grad_xd - dot(proj_d, proj_grad_xd) ^ 2.0 + 1.0e-4 * (dg + dGd)) <= 0.0
                    msg = "Stopping criteria for α_Δ have been achieved"
                    return d, msg
                else # ver daqui em diante...
                    d_old .= d
                    s .= new_search_direction(aux_vector_4, aux_vector_2)
                    d, indexes, value = calculate_theta(gk, Gk, xk, d, aux_vector_4, s, a, b)
                    push!(I, indexes)

                    if value == true
                        doGdo = dot(d_old, Gk * d_old)
                        dGd = dot(d, Gk * d)
                        dogk = dot(d_old, gk)
                        if (dogk + 0.5 * doGdo - dot(d, gk) - 0.5 * dGd + 1.0e-2 * (dogk + 0.5 * doGdo)) <= 0.0
                            return d
                        end
                    end
                end
            end          
        end

        # α_B is chosen
        if index_α == 2
            push!(I, indexes)
            
            if sqrt(aux_1) * Δ + 1.0e-2 * aux_2 <= 0.0
                return d
            else
                s .= .- aux_vector_2
                continue
            end

        end

        # α_Q is chosen
        if index_α == 3
            aux_vector_3 .= d - α * s
            aux_3 = dot(aux_vector_3, Gk * aux_vector_3)
            aux_5 = dot(d, Gk * d)
            aux_6 = - α * dot(s, gk)

            if (sqrt(aux_1) * Δ + 1.0e-2 * aux_2 <= 0.0) || (aux_6 + aux_3 - aux_5 + 1.0e-2 * aux_2 <= 0.0)
                return d
            else
                mul!(aux_vector_5, Gk, s)
                aux_vector_5 *= - α
                mul!(aux_vector_6, Gk, d - \alpha * s)
                aux_vector_6 .+= gk
                aux_vector_7 = projection_active_set(aux_vector_6)
                β = - dot(aux_vector_5, aux_vector_7) / dot(aux_vector_5, s)
                mul!(aux_vector_8, Gk, d)
                aux_vector_8 .+= gk
                if β * dot(s, aux_vector_8) < 0.0
                    s .*= β
                else
                    s .*= -β
                    continue
                end
            end
        end
    end

    return d

end

