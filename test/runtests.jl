using Test
using NoisyNLPModels
using NLPModels
using ADNLPModels

@testset "NoisyNLPModels Tests" begin
  # Create a base model for testing
  base_model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])

  # Test the constructor
  @testset "RandomNoisyNLPModel constructor" begin
    model = RandomNoisyNLPModel(base_model, 0.1)
    @test model.base_model == base_model
    @test model.noise_level == 0.1
  end

  # Test the obj function
  @testset "obj function" begin
    model = RandomNoisyNLPModel(base_model, 0.1)
    x = model.meta.x0
    @test abs(obj(model, x) - obj(base_model, x)) <= model.noise_level * 1 / 2

    # Test the obj function with a given noise
    noise = 0.2
    @test abs(obj(model, x, noise) - obj(base_model, x) - noise) <= 1e-10
  end

  # Test the grad! function
  @testset "grad! function" begin
    model = RandomNoisyNLPModel(base_model, 0.1)
    x = model.meta.x0
    g = similar(x)
    #testing the gradient
    grad!(model, x, g)
    @test g .- grad(base_model, x) .< model.noise_level * (1 / 2)

    # Test the grad! function with a given noise
    noise = 0.2
    g = similar(x)
    grad!(model, x, g, noise)
    @test g == grad(base_model, x) .+ noise

    # Test the grad function 
    g = grad(model, x)
    @test g .- grad(base_model, x) .< model.noise_level * (1 / 2)

    # Test the grad function with a given noise
    noise = 0.2
    g = grad(model, x, noise)
    @test g == grad(base_model, x) .+ noise
  end
end
