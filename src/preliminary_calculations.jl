# Preliminary functions for BOBYQA algorithm

# Main reference:
# POWELL, M. J. D. (2009). The BOBYQA algorithm for bound constrained optimization without derivatives.
# Cambridge NA Report NA2009/06, University of Cambridge, Cambridge, 26-46.

"""

    check_initial_room(n, Δ, a, b)

    Checks whether the limits satisfy the conditions b[i] >= a[i]+2*Δ

    - 'n': dimension of the search space
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds

    Returns a boolean (true if satisfy the conditions above or false otherwise)

"""
function check_initial_room(n, Δ, a, b)

    for i = 1:n
        if b[i] < (a[i] + 2.0 * Δ)
            return false
        end
    end

    return true

end

"""

    correct_initial_guess!(n, Δ, a, b, x)

    Checks whether the limits satisfy the conditions b[i] >= a[i]+2*Δ

    - 'n': dimension of the search space
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds
    - 'x': n-dimensional vector (first iterate)

    Returns a n-dimensional vector

"""
function correct_initial_guess!(n, Δ, a, b, x)

    for i = 1:n
        if x[i] < a[i]
            x[i] = a[i]
        elseif x[i] > b[i]
            x[i] = b[i]
        elseif (a[i] < x[i]) && (x[i] < a[i] + Δ)
            x[i] = a[i] + Δ
        elseif (b[i] - Δ < x[i]) && (x[i] < b[i])
            x[i] = b[i] - Δ
        end
    end

end

"""

    construct_set!(x0, Δ, a, b, p, set, α_values, β_values)

    Partially builds the set of interpolation points of the first model 
    and two vectors with some important constants.

    - 'x0': n-dimensional vector (first iterate)
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds
    - 'p': integer (number of pairs of points to be generated)
    - 'set': n × m matrix (set of interpolation points)
    - 'α_values': n-dimensional vector with some constants related with the first n points
    - 'β_values': n-dimensional vector with some constants related with the last n points

    Returns a modified version of the matrix set and two other vectors.

"""
function construct_set!(x0, Δ, a, b, p, set, α_values, β_values)
    n = length(x0)

    for i=1:p
        if x0[i] == a[i]
            set[:, i + 1] = x0
            set[i, i + 1] += Δ
            α_values[i] = Δ
            set[:, n + i + 1] = x0
            set[i, n + i + 1] += 2.0 * Δ
            β_values[i] = 2.0 * Δ
        elseif x0[i] == b[i]
            set[:, i + 1] = x0
            set[i, i + 1] += -Δ
            α_values[i] = -Δ
            set[:, n + i + 1] = x0
            set[i, n + i + 1] += -2.0 * Δ
            β_values[i] = -2.0 * Δ
        else
            set[:, i + 1] = x0
            set[i, i + 1] += Δ
            α_values[i] = Δ
            set[:, n + i + 1] = x0
            set[i, n + i + 1] += -Δ
            β_values[i] = -Δ
        end
    end

end

"""

    construct_set_aux!(x0, Δ, a, b, q, set, α_values, β_values)

    Changes the qth point of the set of interpolation points and the qth value of the 
    vectors α_values and β_values

    - 'x0': n-dimensional vector (first iterate)
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds
    - 'q': point index to be changed
    - 'set': n × m matrix (set of interpolation points)
    - 'α_values': n-dimensional vector with some constants related with the first n points
    - 'β_values': n-dimensional vector with some constants related with the last n points

    Returns a modified version of the matrix set and two other vectors.

"""
function construct_set_aux!(x0, Δ, a, b, q, set, α_values, β_values)

    if x0[q] == a[q]
        set[:, q + 1] = x0
        set[q, q + 1] += Δ
        α_values[q] = Δ
        β_values[q] = 0.0
    elseif x0[p] == b[p]
        set[:, q + 1] = x0
        set[q, q + 1] += -Δ
        α_values[q] = -Δ
        β_values[q] = 0.0
    else
        set[:, q + 1] = x0
        set[q, q + 1] += Δ
        α_values[q] = Δ
        β_values[q] = 0.0
    end

end

"""

    initial_set(f, x0, Δ, a, b, m)

    Constructs the set of interpolation points of the first model,
    calculates the function values at these points, some important
    constants related to the interpolation points and the values
    of p(j) and q(j)
    
    - 'f': objective function
    - 'x0': n-dimensional vector (first iterate)
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds
    - 'm': integer (number of interpolation conditions)

    Returns an n × m matrix, a m-dimensional vector,
    two n-dimensional vectors and two index lists.

"""
function initial_set(f, x0, Δ, a, b, m)
    n = length(x0)
    set = zeros(n, m)
    α_values = zeros(n)
    β_values = zeros(n)
    f_values = zeros(m)
    p_indexes = []
    q_indexes = []
    set[:, 1] = x0
    aux_vector = zeros(n)
    aux = 0.0
    p = 0
    q = 0

    if m <= (2*n + 1)
        if mod(m-1, 2)
            p = convert(Int, (m - 1) / 2)
            construct_set!(x0, Δ, a, b, p, set, α_values, β_values)
        else
            p = convert(Int, (m - 2) / 2)
            construct_set!(x0, Δ, a, b, p, set, α_values, β_values)
            construct_set_aux!(x0, Δ, a, b, p + 1, set, α_values, β_values)
        end

        for i=1:m
            f_values[i] = f(set[:, i])
        end

    else
        construct_set!(x0, Δ, a, b, n, set, α_values, β_values)
        for i=1:convert(Int, 2 * n + 1)
            f_values[i] = f(set[:, i])
        end

        for i = 1:n
            if (a[i] < x0[i]) && (x0[i] < b[i]) && (f_values[n + i + 1] < f_values[i + 1])
                aux_vector .= set[:, i + 1]
                set[:, i + 1] .= set[:, n + i + 1]
                set[:, n + i + 1] .= aux_vector
                aux = f_values[i + 1]
                f_values[i + 1] = f_values[n + i + 1]
                f_values[i + 1] = aux
                aux = α_values[i]
                α_values[i] = β_values[i]
                β_values[i] = aux
            end
        end

        c = convert(Int, div(m - 2 * n - 1, n) - 1)

        for l = 0:c
            for j = ((2 + l) * n + 2):((3 + l) * n + 1)
                p = convert(Int, j - (l + 2) * n - 1)
                if (1 <= p + l + 1) && (p + l + 1 <= n)
                    q = p + l + 1
                else
                    q = p + l + 1 - n
                end
                push!(p_indexes, p)
                push!(q_indexes, q)
                set[:, j] .= set[:, p + 1] .+ set[:, q + 1] .- x0

            end
        end
        for j = ((3 + c) * n + 2 ):m
            p = convert(Int, j - (3 + c) * n - 1)
            if (1 <= p + c + 2) && (p + c + 2 <= n)
                q = p + c + 2
            else
                q = p + c + 2 - n
            end
            push!(p_indexes, p)
            push!(q_indexes, q)
            set[:, j] .= set[:, p + 1] .+ set[:, q + 1] .- x0

        end

        for i=convert(Int, 2 * n + 2):m
           f_values = f(set[:, j]) 
        end
        
    end
 
    return set, f_values, α_values, β_values, p_indexes, q_indexes

end

"""

    initial_model(set, f_values, α_values, β_values, p_indexes, q_indexes)

    Constructs the gradient vector and the Hessian matrix of the first model
    
    - 'f_values': m-dimensional vector with the function values at the interpolation points
    - 'α_values': n-dimensional vector with some constants related with the first n points
    - 'β_values': n-dimensional vector with some constants related with the last n points
    - 'p_indexes': list containing the values of p(j), for the case 2n + 2 <= j <= leq m
    - 'q_indexes': list containing the values of q(j), for the case 2n + 2 <= j <= leq m

    Returns a n-dimensional vector g = ∇Q and a n × m matrix G = ∇^2Q

"""
function initial_model(f_values, α_values, β_values, p_indexes, q_indexes)
    n = length(α_values)
    m = length(f_values)
    n1 = length(p_index)
    g .= 0.0
    G .= 0.0
    aux_matrix = zeros(2, 2)
    aux_vector_01 = zeros(2)
    aux_vector_02 = zeros(2)

    for i=1:min(n, m-n-1)
        aux_matrix[1, 1] = α_values[i]
        aux_matrix[1, 2] = α_values[i] ^ 2.0 / 2.0
        aux_matrix[2, 1] = β_values[i]
        aux_matrix[2, 2] = β_values[i] ^ 2.0 / 2.0
        aux_vector_01[1] = f_values[i + 1] - f_values[1]
        aux_vector_01[2] = f_values[n + i + 1] - f_values[1]

        aux_vector_02 = aux_matrix \ aux_vector_01
        g[i] = aux_vector_02[1]
        G[i, i] = aux_vector_02[2]
    end
    if m < 2*n+1
        for i=(m-n):n
            g[i] = (f_values[i + 1] - f_values[1]) / α_values[i]
        end
    else
        for i = 1:n1
            j = 2*n + 1 + i
            p_j = p_indexes[i]
            q_j = q_indexes[i]
            G[p_j, q_j] = (f_values[j] - f_values[1] - α_values[p_j] * g[p_j] - α_values[q_j] * g[q_j] 
                                - (α_values[p_j] ^ 2.0 / 2.0) * G[p_j, p_j] 
                                - (α_values[q_j] ^ 2.0 / 2.0) * G[q_j, q_j] ) / (α_values[p_j] * α_values[q_j])
            G[q_j, p_j] = G[p_j, q_j]
        end
    end

    return g, G

end