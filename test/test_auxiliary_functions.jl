@testset "Auxiliary" begin

    @testset "active_set" begin

        n = 2
        
        g = ones(n)
        a = ones(n)
        b = 2 * ones(n)

        x = ones(n)
        indexes = zeros(Bool, n)
        sol_indexes = zeros(Bool, n)
        sol_indexes .= true
        BOBYQA.active_set!(n, g, x, a, b, indexes)
        @test( indexes == sol_indexes )

        x = 2.0 * ones(2)
        indexes = zeros(Bool, n)
        sol_indexes = zeros(Bool, n)
        BOBYQA.active_set!(n, g, x, a, b, indexes)
        @test( indexes == sol_indexes )

        x = [1.5, 1]
        indexes = zeros(Bool, n)
        sol_indexes = zeros(Bool, n)
        sol_indexes[2] = true
        BOBYQA.active_set!(n, g, x, a, b, indexes)
        @test( indexes == sol_indexes )

        # Test the case where the variable is fixed (maybe remove the
        # variable before?)
        a[1] = b[1]

        x = 2.0 * ones(2)
        indexes = zeros(Bool, n)
        sol_indexes = zeros(Bool, n)
        sol_indexes[1] = true
        BOBYQA.active_set!(n, g, x, a, b, indexes)
        @test( indexes == sol_indexes )
        
    end

    @testset "projection_active_set!" begin

        n = 10
        
        v = rand(n)
        proj_v = Vector{Float64}(undef, n)

        acts = zeros(Bool, n)
        BOBYQA.projection_active_set!(v, acts, proj_v)

        @test(proj_v == v)

        acts = zeros(Bool, n)
        acts[1] = true
        BOBYQA.projection_active_set!(v, acts, proj_v)

        @test(proj_v[1] == 0.0)
        @test(proj_v[2:n] == v[2:n])

        acts = zeros(Bool, n)
        index = rand(1:n, 5)
        acts[index] .= true
        sol = zeros(n)
        copyto!(sol, v)
        sol[index] .= 0.0
        BOBYQA.projection_active_set!(v, acts, proj_v)
        @test(proj_v == sol)
        
    end

    @testset "update_active_set" begin
        n = 10

        index_set = zeros(Bool, n)
        index_list = zeros(Bool, n)
        sol_index = zeros(Bool, n)
        BOBYQA.update_active_set!(index_set, index_list)
        @test(index_set == sol_index)

        index_set = zeros(Bool, n)
        index_list = zeros(Bool, n)
        index_list[1] = true
        index_list[2] = true
        sol_index = zeros(Bool, n)
        sol_index[1] = true
        sol_index[2] = true
        BOBYQA.update_active_set!(index_set, index_list)
        @test(index_set == sol_index)

        index_set = zeros(Bool, n)
        index_set[3] = true
        index_set[6] = true
        index_set[8] = true
        index_list = zeros(Bool, n)
        sol_index = zeros(Bool, n)
        sol_index[3] = true
        sol_index[6] = true
        sol_index[8] = true
        BOBYQA.update_active_set!(index_set, index_list)
        @test(index_set == sol_index)

        index_set = zeros(Bool, n)
        index_set[2] = true
        index_set[4] = true
        index_set[6] = true
        index_list = zeros(Bool, n)
        index_list[7] = true
        index_list[9] = true
        sol_index = zeros(Bool, n)
        sol_index[2] = true
        sol_index[4] = true
        sol_index[6] = true
        sol_index[7] = true
        sol_index[9] = true
        BOBYQA.update_active_set!(index_set, index_list)
        @test(index_set == sol_index)

        index_set = zeros(Bool, n)
        index_set[1] = true
        index_set[2] = true
        index_set[3] = true
        index_set[4] = true
        index_set[5] = true
        index_list = zeros(Bool, n)
        index_list[2] = true
        index_list[4] = true
        index_list[5] = true
        index_list[6] = true
        index_list[8] = true
        sol_index = zeros(Bool, n)
        sol_index[1] = true
        sol_index[2] = true
        sol_index[3] = true
        sol_index[4] = true
        sol_index[5] = true
        sol_index[6] = true
        sol_index[8] = true
        BOBYQA.update_active_set!(index_set, index_list)
        @test(index_set == sol_index)
    
    end

    @testset "calculate_alpha" begin
    
        n = 10
        x = ones(n)
        d = 0.5 * ones(n)
        s = 0.5 * ones(n)
        Δ = 1.0
        a = ones(n)
        b = 2.0 * ones(n)
        sg = 1.0
        sGd = 1.0
        sGs = 1.0
        index_list = zeros(Bool, n)
        sol_index = zeros(Bool, n)

        α, index_α = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs, index_list)
        @test(α == - 2.0)
        @test(index_α == 3)
        @test(index_list == sol_index)

        n = 10
        x = ones(n)
        d = 0.5 * ones(n)
        s = - 0.5 * ones(n)
        Δ = 1.0
        a = ones(n)
        b = 2.0 * ones(n)
        sg = - 5.0
        sGd = 1.0
        sGs = 1.0
        index_list = zeros(Bool, n)

        α, index_α = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs, index_list)
        @test(α == 1.0)
        @test(index_α == 2)
        @test(sum(index_list) == n)

        n = 10
        x = ones(n)
        d = 0.5 * ones(n)
        s = - 0.5 * ones(n)
        Δ = 0.5
        a = ones(n)
        b = 2.0 * ones(n)
        sg = - 5.0
        sGd = 1.0
        sGs = 1.0
        index_list = zeros(Bool, n)
        
        α, index_α = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs, index_list)
        @test(α == 1.0)
        @test(index_α == 2)
        @test(sum(index_list) == n)

        n = 10
        x = ones(n)
        d = ones(n)
        s = ones(n)
        Δ = 1.0
        a = zeros(n)
        b = 5.0 * ones(n)
        sg = - 2.0
        sGd = 1.0
        sGs = 1.0
        index_list = zeros(Bool, n)
        sol_index = zeros(Bool, n)
        
        α, index_α = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs, index_list)
        @test(α == 0.0)
        @test(index_α == 1)
        @test(index_list == sol_index)

        n = 3
        x = ones(n)
        d = [0.5, 1.0, 0.5]
        s = [- 0.5, 1.0, 0.5]
        Δ = 1.0
        a = [1.0, 0.0, 1.0]
        b = [2.0, 5.0, 2.0]
        sg = - 5.0
        sGd = 1.0
        sGs = 1.0
        index_list = zeros(Bool, n)
        sol_index = zeros(Bool, n)
        
        α, index_α = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs, index_list)
        @test(α == 0.0)
        @test(index_α == 1)
        @test(index_list == sol_index)

    end

    @testset "new_search_direction" begin
        n = 2

        v = rand(n)
        pd = [1.0, 0.0]
        pg = [0.0, 1.0]
        n_pd = dot(pd, pd)
        n_pg = dot(pg, pg)
        
        BOBYQA.new_search_direction!(pd, pg, n_pd, n_pg, v)
        @test(isapprox(v, [0.0, - 1.0], atol = 1.0e-1))

        v = rand(n)
        pd = [1.0, 1.0]
        pg = [0.0, 1.0]
        n_pd = dot(pd, pd)
        n_pg = dot(pg, pg)
        
        BOBYQA.new_search_direction!(pd, pg, n_pd, n_pg, v)
        @test(isapprox(v, [1.0, - 1.0], atol = 1.0e-1))

    end
    
    @testset "stop_condition_theta_B" begin
        
        θ = pi / 4
        n = 2
        x = ones(n)
        d = ones(n)
        s = ones(n)
        pd = ones(n)
        a = ones(n)
        b = 3 * ones(n)

        @test(BOBYQA.stop_condition_theta_B(θ, n, x, d, s, pd, a, b) == true)

        θ = pi / 4
        n = 2
        x = ones(n)
        d = - 5.0 * ones(n)
        s = ones(n)
        pd = ones(n)
        a = ones(n)
        b = 3 * ones(n)
        
        @test(BOBYQA.stop_condition_theta_B(θ, n, x, d, s, pd, a, b) == false)

        θ = pi / 4
        n = 2
        x = ones(n)
        d = [1.0, - 5.0]
        s = ones(n)
        pd = ones(n)
        a = ones(n)
        b = 3 * ones(n)
        
        @test(BOBYQA.stop_condition_theta_B(θ, n, x, d, s, pd, a, b) == false)

    end

    @testset "stop_condition_theta_Q" begin

        θ = pi / 4
        @test(BOBYQA.stop_condition_theta_Q(θ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) == false)

        θ = pi / 6
        @test(BOBYQA.stop_condition_theta_Q(θ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) == false)

        θ = pi / 4
        @test(BOBYQA.stop_condition_theta_Q(θ, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) == false)

        θ = pi / 6
        @test(BOBYQA.stop_condition_theta_Q(θ, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) == false)

        θ = pi / 4
        @test(BOBYQA.stop_condition_theta_Q(θ, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0) == true)

        θ = pi / 6
        @test(BOBYQA.stop_condition_theta_Q(θ, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0) == true)

    end

    @testset "binary_search" begin
        function stop_cond(a)
            return a < 0.2
        end

        value = BOBYQA.binary_search(0.0, 1.0, stop_cond, 1.0e-1)
        @test(value == 0.1875)

        value = BOBYQA.binary_search(- 2.0, 1.0, stop_cond, 1.0e-1)
        @test(value == 0.15625)

        value = BOBYQA.binary_search(- 2.0, 0.0, stop_cond, 1.0e-1)
        @test(value == 0.0)

    end

    @testset "calculate_theta!" begin

        n = 2
        x = ones(n)
        d = ones(n)
        s = ones(n)
        pd = ones(n)
        a = ones(n)
        b = 3 * ones(n)
        index_list = zeros(Bool, n)
        sol_index = zeros(Bool, n)

        bool_value = BOBYQA.calculate_theta!(n, x, pd, s, a, b, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, d, index_list)
        @test(bool_value == false)
        @test(isapprox(d, sqrt(2.0) * ones(n), atol = 1.0e-1))
        @test(index_list == sol_index)

        n = 2
        x = ones(n)
        d = ones(n)
        s = ones(n)
        pd = ones(n)
        a = ones(n)
        b = 3 * ones(n)
        index_list = zeros(Bool, n)
        sol_index = zeros(Bool, n)

        bool_value = BOBYQA.calculate_theta!(n, x, pd, s, a, b, 1.0, - 1.0, 0.0, 0.0, - 1.0, 0.0, 0.0, d, index_list)
        @test(bool_value == true)
        @test(isapprox(d, (cos(pi / 32.0) + sin(pi / 32.0)) * ones(n), atol = 1.0e-1))
        @test(index_list == sol_index)

    end

    @testset "stopping_criterion_34" begin

        Δ = 5.0
        n2_pg = 1.0
        dg = 1.0
        dGd = 1.0

        @test(BOBYQA.stopping_criterion_34(Δ, n2_pg, dg, dGd) == false)

        Δ = 1.0
        n2_pg = 1.0
        dg = - 200.0
        dGd = 10.0

        @test(BOBYQA.stopping_criterion_34(Δ, n2_pg, dg, dGd) == true)

    end

    @testset "stopping_criterion_34_B" begin
        
        dg = 1.0
        dog = 1.0
        dGd = 1.0
        doGdo = 1.0

        @test(BOBYQA.stopping_criterion_34_B(dg, dog, dGd, doGdo) == false)

        dg = 1.0
        dog = 0.0
        dGd = 1.0
        doGdo = 0.0

        @test(BOBYQA.stopping_criterion_34_B(dg, dog, dGd, doGdo) == true)

    end

    @testset "stopping_criterion_35" begin
        
        pd = [1.0, 0.0]
        pg = [0.0, 1.0]
        n2_pd = dot(pd, pd)
        n2_pg = dot(pg, pg)
        dg = 1.0
        dGd = 1.0

        @test(BOBYQA.stopping_criterion_35(pd, pg, n2_pd, n2_pg, dg, dGd) == false)

        pd = [1.0, 0.0]
        pg = [1.0, 0.0]
        n2_pd = dot(pd, pd)
        n2_pg = dot(pg, pg)
        dg = - 1.0
        dGd = - 1.0

        @test(BOBYQA.stopping_criterion_35(pd, pg, n2_pd, n2_pg, dg, dGd) == true)

    end

end