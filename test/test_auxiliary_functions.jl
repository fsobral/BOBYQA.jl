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
    
end
