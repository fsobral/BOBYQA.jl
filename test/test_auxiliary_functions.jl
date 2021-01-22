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
        @test(α == -1.0)
        @test(index_α == 3)
        @test(index_list == [])

        s = - 0.5 * ones(n)
        sg = - 5.0

        α, index_α, index_list = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs)
        @test(α == 1.0)
        @test(index_α == 2)
        @test(lenght(index_list) == n)

        Δ = 0.5

        α, index_α, index_list = BOBYQA.calculate_alpha(n, x, d, s, Δ, a, b, sg, sGd, sGs)
        @test(α == 0.0)
        @test(index_α == 1)
        @test(lenght(index_list) == [])

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

end
