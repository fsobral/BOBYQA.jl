@testset "Trsbox" begin

    @testset "Exit condition: Initial constraint set " begin

        n = 2    
        g = ones(n)
        G = Matrix{Float64}(I, 3, 3)
        x = ones(n)
        Δ = 5.0
        a = ones(n)
        b = 2 * ones(n)
        d = rand(n)

        msg = BOBYQA.trsbox!(n, g, G, x, Δ, a, b, d)

        @test(isapprox(d, zeros(n), atol = 1.0e-1))
        @test(msg == "All bounds restrictions are active.")
    
    end

    @testset "Exit condition: Initial search direction" begin

        n = 2    
        g = zeros(n)
        G = Matrix{Float64}(I, 3, 3)
        x = 2.0 * ones(n)
        Δ = 5.0
        a = ones(n)
        b = 3.0 * ones(n)
        d = rand(n)

        msg = BOBYQA.trsbox!(n, g, G, x, Δ, a, b, d)

        @test(isapprox(d, zeros(n), atol = 1.0e-1))
        @test(msg == "The initial search direction is null.")


    end

    @testset "Exit condition: α_B criterion" begin
        
        n = 2    
        g = ones(n)
        G = Matrix{Float64}(I, 3, 3)
        x = 2.0 * ones(n)
        Δ = 5.0
        a = ones(n)
        b = 3.0 * ones(n)
        d = rand(n)

        @test(isapprox(d, - ones(n), atol = 1.0e-1))
        @test(msg == "Stopping criterion for α_B has been reached")

    end


end