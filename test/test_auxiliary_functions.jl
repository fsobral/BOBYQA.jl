@testset "Auxiliary" begin

    @testset "active_set" begin

        n = 2
        
        g = ones(n)
        a = ones(n)
        b = 2 * ones(n)

        x = [1, 1]
        @test( BOBYQA.active_set(n, g, x, a, b) == [1, 2] )

        x = [2, 2]
        @test( BOBYQA.active_set(n, g, x, a, b) == [] )

        x = [1.5, 1]
        @test( BOBYQA.active_set(n, g, x, a, b) == [2] )

        # Test the case where the variable is fixed (maybe remove the
        # variable before?)
        a[1] = b[1]

        x = [2, 2]
        @test( BOBYQA.active_set(n, g, x, a, b) == [1] )
        
    end

    @testset "projection_active_set!" begin

        n = 10
        
        v = rand(n)
        proj_v = Vector{Float64}(undef, n)

        acts = []
        BOBYQA.projection_active_set!(v, acts, proj_v)

        @test(proj_v == v)

        acts = [1]
        BOBYQA.projection_active_set!(v, acts, proj_v)

        @test(proj_v[1] == 0.0)
        @test(proj_v[2:n] == v[2:n])

        acts = rand(1:n, 5)
        BOBYQA.projection_active_set!(v, acts, proj_v)
        @test(proj_v[acts] == zeros(5))
        
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

        α, index_α, index_list = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs)
        @test(α == - 2.0)
        @test(index_α == 3)
        @test(index_list == [])

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

        α, index_α, index_list = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs)
        @test(α == 1.0)
        @test(index_α == 2)
        @test(length(index_list) == n)

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
        
        α, index_α, index_list = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs)
        @test(α == 1.0)
        @test(index_α == 2)
        @test(length(index_list) == n)

    end

    @testset "new_search_direction" begin
        n = 2

        v = rand(n)
        pd = [1.0, 0.0]
        pg = [1.0, 0.0]
        n_pd = dot(pd, pd)
        n_pg = dot(pg, pg)

        BOBYQA.new_search_direction!(pd, pg, n_pd, n_pg, v)
        @test(v == [0.0, 0.0])

        v = rand(n)
        pd = [1.0, 0.0]
        pg = [0.0, 1.0]
        n_pd = dot(pd, pd)
        n_pg = dot(pg, pg)
        
        BOBYQA.new_search_direction!(pd, pg, n_pd, n_pg, v)
        @test(v == [0.0, - 1.0])

        v = rand(n)
        pd = [1.0, 1.0]
        pg = [0.0, 1.0]
        n_pd = dot(pd, pd)
        n_pg = dot(pg, pg)
        
        BOBYQA.new_search_direction!(pd, pg, n_pd, n_pg, v)
        @test(v == [1.0, - 1.0])

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

        d = - 5.0 * ones(n)
        @test(BOBYQA.stop_condition_theta_B(θ, n, x, d, s, pd, a, b) == false)

        d = [1.0, - 5.0]
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

        index_list, bool_value = BOBYQA.calculate_theta!(n, x, pd, s, a, b, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, d)
        @test(bool_value = false)
        @test(d == sqrt(2) * ones(n))
        @test(index_list = [])

        n = 2
        x = ones(n)
        d = ones(n)
        s = ones(n)
        pd = ones(n)
        a = ones(n)
        b = 3 * ones(n)

        index_list, bool_value = BOBYQA.calculate_theta!(n, x, pd, s, a, b, - 1.0, - 1.0, - 1.0, - 1.0, - 1.0, - 1.0, - 1.0, d)
        @test(bool_value = true)
        @test(d == (cos(pi / 8.0) + sin(pi / 8.0)) * ones(n))
        @test(index_list = [])

    end

    @testset "stopping_criterion_34" begin

        Δ = 5.0
        n2_pg = 1.0
        dg = 1.0
        dGd = 1.0

        @test(BOBYQA.stopping_criterion_34(Δ, n2_pg, dg, dGd) == false)

        Δ = 1.0
        n2_pg = 0.01
        dg = 1.0
        dGd = 1.0

        @test(BOBYQA.stopping_criterion_34(Δ, n2_pg, dg, dGd) == true)

    end

    @testset "stopping_criterion_34_B" begin
        
        dg = 1.0
        dog = 1.0
        dGd = 1.0
        doGdo = 1.0

        @test(BOBYQA.stopping_criterion_34_B(dg, dog, dGd, doGdo) == false)

        dg = 0.0
        dog = 0.0
        dGd = 1.0
        doGdo = 1.0

        @test(BOBYQA.stopping_criterion_34_B(dg, dog, dGd, doGdo) == true)

    end

    @testset "stopping_criterion_35" begin
        
        pd = [1.0, 0.0]
        pg = [0.0, 1.0]
        n2_pd = dot(pd, pd)
        n2_pg = dot(pg, pg)
        dg = 1.0
        dgd = 1.0

        @test(BOBYQA.stopping_criterion_35(pd, pg, n2_pd, n2_pg, dg, dGd) == false)

        pd = [1.0, 0.0]
        pg = [1.0, 0.0]
        n2_pd = dot(pd, pd)
        n2_pg = dot(pg, pg)
        dg = - 1.0
        dgd = - 1.0

        @test(BOBYQA.stopping_criterion_35(pd, pg, n2_pd, n2_pg, dg, dGd) == true)

    end

end