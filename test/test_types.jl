@testset "Hessian type" begin

    @testset "Initialization" begin

        M = rand(5, 5)
        μ = rand(10)
        Y = rand(Float64, 5, 10)
        x0 = rand(5)

        Q = BOBYQA_Hessian(M, μ, Y, x0)

        @test(Q.M === M)
        @test(Q.μ === μ)
        @test(Q.Y === Y)
        @test(Q.x0 === x0)

        # Dimension error
        @test_throws(DimensionMismatch, BOBYQA_Hessian(rand(5, 10), μ, Y, x0))
        @test_throws(DimensionMismatch, BOBYQA_Hessian(M, [1, 2], Y, x0))
        @test_throws(DimensionMismatch, BOBYQA_Hessian(M, μ, Y, [0, 0, 0]))

        # Empty type
        Q = BOBYQA_Hessian(5, 2)
        @test(size(Q.M) == (5, 5))
        @test(length(Q.μ) == 2)
        @test(size(Q.Y) == (5, 2))
        @test(length(Q.x0) == 5)
        
    end

    @testset "Product overloading" begin

        M = Matrix{Float64}(I, 5, 5)
        μ = [1, 0, 0]
        Y = Matrix{Float64}(I, 5, 3)
        x0 = zeros(5)
        
        Q = BOBYQA_Hessian(M, μ, Y, x0)

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

        # Change center
        Q.x0 .= ones(5)

        v = [0, 0, 1, 0, 0]

        @test(Q * v == [0, 1, 2, 1, 1])
                
    end

    @testset "Auxiliary" begin

        @testset "Test add_rank1_mul" begin

            n = 5
            m = 2
            
            Q = BOBYQA_Hessian(ones(n, n),
                               ones(m),
                               Matrix{Float64}(I, n, m),
                               zeros(n))

            v = [1, 0, 1, 0, 0]
            y = zeros(n)

            BOBYQA.add_rank1_mul!(y, Q, v)
            @test(y == [1, 0, 0, 0, 0])

            y = ones(n)
            BOBYQA.add_rank1_mul!(y, Q, v)
            @test(y == [2, 1, 1, 1, 1])
            
        end

    end
    
end
