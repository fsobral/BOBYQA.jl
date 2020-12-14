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

