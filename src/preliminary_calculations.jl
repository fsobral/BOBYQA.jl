using LinearAlgebra

"""

    check_initial_room(Δ, a, b)

    Checks whether the limits satisfy the conditions b[i] >= a[i]+2*Δ

    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds

    Returns a boolean (true if satisfy the conditions above or false otherwise)

"""
function check_initial_room(Δ, a, b)
    n = length(a)

    for i = 1:n
        if b[i] < (a[i] + 2.0 * Δ)
            return false
        end
    end

    return true

end

"""

    correct_initial_guess(x0, Δ, a, b)

    Checks whether the limits satisfy the conditions b[i] >= a[i]+2*Δ

    - 'x0': n-dimensional vector (first iterate)
    - 'Δ': positive real value (trust-region radius)
    - 'a': n-dimensional vector with the lower bounds
    - 'b': n-dimensional vector with the upper bounds

    Returns a n-dimensional vector

"""
function correct_initial_guess(x0, Δ, a, b)
    n = length(x0)
    x = copy(x0)

    for i = 1:n
        if x0[i] < a[i]
            x[i] = a[i]
        elseif x0[i] > b[i]
            x[i] = b[i]
        elseif (a[i] < x0[i]) && (x0[i] < a[i] + Δ)
            x[i] = a[i] + Δ
        elseif (b[i] - Δ < x0[i]) && (x0[i] < b[i])
            x[i] = b[i] - Δ
        end
    end

    return x

end

function initial_set(x0, Δ, a, b, m)
    n = length(x0)
    set = zeros(n, m)
    set[:, 1] = x0
    
    for i=1:n
        if x0[i] == a[i]
            set[:, i+1] = x0
            set[i, i+1] += Δ
            set[:, n + i + 1] = x0
            set[i, n + i + 1] += 2.0 * Δ
        elseif x0[i] == b[i]
            set[:, i+1] = x0
            set[i, i+1] += -Δ
            set[:, n + i + 1] = x0
            set[i, n + i + 1] += -2.0 * Δ
        else
            set[:, i+1] = x0
            set[i, i+1] += Δ
            set[:, n + i + 1] = x0
            set[i, n + i + 1] += -Δ
        end
    end

end