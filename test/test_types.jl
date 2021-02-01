@testset "Hessian type" begin

    @testset "Initialization" begin

        M = rand(5, 5)
        μ = rand(10)
        Y = rand(Float64, 5, 10)

        Q = BOBYQA_Hessian(M, μ, Y)

        @test(Q.M == M)
        @test(Q.μ == μ)
        @test(Q.Y == Y)

        # Dimension error
        @test_throws(DimensionMismatch, BOBYQA_Hessian(M, [1, 2], Y))
        
    end

    @testset "Product overloading" begin

        M = Matrix{Float64}(I, 5, 5)
        μ = [1, 0, 0]
        Y = Matrix{Float64}(I, 5, 3)
        
        Q = BOBYQA_Hessian(M, μ, Y)

        v = [1, 0, 0, 0, 0]
        @test(Q * v == [2, 0, 0, 0, 0])

        y = Vector{Float64}(undef, 5)
        @test(mul!(y, Q, v) == [2, 0, 0, 0, 0])
        @test(y == [2, 0, 0, 0, 0])
        @test(v == [1, 0, 0, 0, 0])
        
        v = [0, 1, 0, 0, 0]
        @test(Q * v == v)
        @test(mul!(y, Q, v) == v)
        @test(y == v)

        # Change multiplier
        Q.μ[3] = 1
        v = [1, 0, 1, 0, 0]

        @test(Q * v == [2, 0, 2, 0, 0])
                
    end
    
end
