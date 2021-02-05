@testset "Trsbox" begin

    @testset "Exit condition: Initial constraint set " begin

        n = 2    
        g = ones(n)
        G = Matrix{Float64}(I, 3, 3)
        x = ones(n)
        Δ = 5.0
        a = ones(n)
        b = 2.0 * ones(n)
        d = rand(n)

        msg = BOBYQA.trsbox!(n, g, G, x, Δ, a, b, d)

        @test(isapprox(d, zeros(n), atol = 1.0e-1))
        @test(msg == "All bounds restrictions are active.")
    
    end

    @testset "Exit condition: Initial search direction" begin

        n = 2    
        g = [1.0, 0.0]
        G = Matrix{Float64}(I, 3, 3)
        x = [1.0, 2.0]
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
        G = Matrix{Float64}(I, n, n)
        x = 2.0 * ones(n)
        Δ = 5.0
        a = ones(n)
        b = 3.0 * ones(n)
        d = zeros(n)

        msg = BOBYQA.trsbox!(n, g, G, x, Δ, a, b, d)

        @test(isapprox(d, - ones(n), atol = 1.0e-1))
        @test(msg == "Stopping criterion for α_B has been reached")

    end

    @testset "Exit condition: α_Q criterion" begin
        
        n = 2    
        g = ones(n)
        G = Matrix{Float64}(I, n, n)
        x = 2.0 * ones(n)
        Δ = 5.0
        a = - ones(n)
        b = 3.0 * ones(n)
        d = rand(n)

        msg = BOBYQA.trsbox!(n, g, G, x, Δ, a, b, d)

        @test(isapprox(d, - ones(n), atol = 1.0e-1))
        @test(msg == "Stopping criterion for α_Q has been reached")

    end

    @testset "Exit condition: α_Δ criterion" begin
        
        n = 2    
        g = ones(n)
        G = Matrix{Float64}(I, n, n)
        x = 2.0 * ones(n)
        Δ = 1.0
        a = - ones(n)
        b = 3.0 * ones(n)
        d = rand(n)

        msg = BOBYQA.trsbox!(n, g, G, x, Δ, a, b, d)

        @test(isapprox(d, - ones(n), atol = 1.0e-1))
        @test(msg == "Stopping criterion for α_Δ has been reached")

    end

    @testset "Exit condition: α_Δ with θ_Q criterion" begin
        
        n = 2    
        g = 5.0 * ones(n)
        G = [10.0 0.0; -5.0 1.0]
        x = zeros(n)
        Δ = 0.5
        a = - 10.0 * ones(n)
        b = 10.0 * ones(n)
        d = rand(n)

        msg = BOBYQA.trsbox!(n, g, G, x, Δ, a, b, d)

        sol_d = [- 1.0 / 3.0, - 1000.0 / 1603.0]
        @test(isapprox(d, sol_d, atol = 1.0e-1))
        @test(msg == "Stopping criterion for α_Δ with the choice of θ_Q has been reached")

    end

end