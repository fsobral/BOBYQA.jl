# TRSBOX - Trust Region Step in the Box

# Main reference:
# POWELL, M. J. D. (2009). The BOBYQA algorithm for bound constrained optimization without derivatives.
# Cambridge NA Report NA2009/06, University of Cambridge, Cambridge.

"""

    trsbox!(n, gk, Gk, xk, Δ, a, b, d)

    A version of a Truncated Conjugate Gradient with Active Set algorithm for the
    Trust-Region Subproblem

    - 'n': dimension of the search space
    - 'gk': n-dimensional vector (gradient of the model calculated in xk)
    - 'Gk': (n × n)-dimensional matrix (hessian of the model calculated in xk)
    - 'xk': n-dimensional vector (current iterate) 
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds 
    
    Modifies d to be the new direction and returns a message about the chosen direction

"""
function trsbox!(n, gk, Gk, xk, Δ, a, b, d)

    #Initializes some variables and vectors.
    α = 0.0
    β = 0.0
    index_α = 0
    dg = 0.0
    dog = 0.0
    sg = 0.0
    sGd = 0.0
    sGs = 0.0
    sGdo = 0.0
    dGd = 0.0
    doGdo = 0.0
    pgGd = 0.0
    pgGdo = 0.0
    norm2_proj_d = 0.0
    norm2_proj_grad_xd = 0.0
    s = zeros(n)
    d_old = zeros(n)
    Gd = zeros(n)
    Gdo = zeros(n)
    Gs = zeros(n)
    Gpd = zeros(n)
    grad_xd = zeros(n)
    proj_d = zeros(n)
    proj_grad_xd = zeros(n)

    # Set the entries of d to null
    d .= 0.0

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
    if isapprox(dot(s, s), 0.0; atol = eps(Float64), rtol = 0.0)
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
        norm2_proj_grad_xd = dot(proj_grad_xd, proj_grad_xd)

        # Computes P_I(d) and ||P_I(d)||^2
        projection_active_set!(d, I, proj_d)
        norm2_proj_d = dot(proj_d, proj_d)
        
        # α_Δ is chosen
        if index_α == 1
            while true

                # Stop Criteria evaluation (inequality 3.5)
                if stopping_criterion_35(proj_d, proj_grad_xd, norm2_proj_d, norm2_proj_grad_xd, dg, dGd)
                    msg = "Stopping criterion for α_Δ has been reached"
                    return d, msg
                else

                    # Save old information about direction d
                    d_old .= d
                    dog = dg
                    Gdo .= Gd
                    doGdo = dGd

                    # Calculate the new search direction s
                    new_search_direction!(proj_d, proj_grad_xd, norm2_proj_d, norm2_proj_grad_xd, s)

                    # Calculate some curvature information
                    mul!(Gs, Gk, s)
                    sGs = dot(s, Gs)
                    dGs = dot(d, Gs)
                    sg = dot(s, gk)
                    pdg = dot(proj_d, gk)
                    mul!(Gpd, Gk, proj_d)
                    dGpd = dot(d, Gpd)
                    sGpd = dot(s, Gpd)
                    pdGpd = dot(proj_d, Gpd)

                    # Calculate the new direction d
                    indexes, value = calculate_theta!(n, xk, proj_d, s, a, b, sg, pdg, sGs, dGs, dGpd, sGpd, pdGpd, d)

                    # Updates the set of indexes of the fixed bounds.
                    push!(I, indexes)

                    # Calculate some curvature information
                    dg = dot(d, gk)
                    mul!(Gd, Gk, d)
                    dGd = dot(d, Gd)

                    # Stop Criteria evaluation (inequality 3.4 modified)
                    if value == true                       
                        if stopping_criterion_34_B(dg, dog, dGd, doGdo) 
                            msg = "Stopping criterion for α_Δ with the choice of θ_Q has been reached"
                            return d, msg
                        end
                    end

                    # Computes ∇Q(xk + d), P_I(∇Q(xk + d)) and ||P_I(∇Q(xk + d))||^2     
                    grad_xd .= gk .+ Gd
                    projection_active_set!(grad_xd, I, proj_grad_xd)
                    norm2_proj_grad_xd = dot(proj_grad_xd, proj_grad_xd)

                    # Computes P_I(d) and ||P_I(d)||^2
                    projection_active_set!(d, I, proj_d)
                    norm2_proj_d = dot(proj_d, proj_d)
                end
            end          
        end

        # α_B is chosen
        if index_α == 2

            # Updates the set of indexes of the fixed bounds.
            push!(I, indexes)

            # Computes ∇Q(xk + d), P_I(∇Q(xk + d)) and ||P_I(∇Q(xk + d))||^2     
            grad_xd .= gk .+ Gd
            projection_active_set!(grad_xd, I, proj_grad_xd)
            norm2_proj_grad_xd = dot(proj_grad_xd, proj_grad_xd)

            # Computes P_I(d) and ||P_I(d)||^2
            projection_active_set!(d, I, proj_d)
            norm2_proj_d = dot(proj_d, proj_d)
            
            # Stop Criteria evaluation (inequality 3.4)
            if stopping_criterion_34(Δ, norm2_proj_grad_xd, dg, dGd)
                msg = "Stopping criterion for α_B has been reached"
                return d, msg
            else
                # New search direction s
                s .= .- proj_grad_xd
                continue
            end
        end

        # α_Q is chosen
        if index_α == 3

            # Stop Criteria evaluation (inequality 3.4 and inequality 3.4 modified)
            if stopping_criterion_34(Δ, norm2_proj_grad_xd, dg, dGd) || stopping_criterion_34_B(dg, dog, dGd, doGdo)
                msg = "Stopping criterion for α_Q has been reached"
                return d, msg
            else
                # Calculate some curvature information
                pgGd = dot(proj_grad_xd, Gd)
                pgGdo = dot(proj_grad_xd, Gdo)
                sGdo = dot(s, Gdo)

                # New search direction s
                s .= - ((pgGdo - pgGd) / (sGdo - sGd)) .* s
                s .+= proj_grad_xd
                continue
            end
        end
    end
end