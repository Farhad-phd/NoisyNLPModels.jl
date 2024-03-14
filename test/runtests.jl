using Test
using NoisyNLPModels
using NLPModels
using ADNLPModels

@testset "NoisyNLPModels Tests" begin
  # Create a base model for testing
  base_model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  model = RandomNoisyNLPModel(base_model, 0.1)
  x = model.meta.x0
  g_base = grad(base_model, x)
  obj_base = obj(base_model, x)
  T = eltype(x)
  noise = T(0.2)

  
  @testset "helper-function" begin
    update_noise_level!(model, T(0.4))
    @test model.noise_level == T(0.4)
    update_noise_level!(model, T(0.1))
    @test model.noise_level == T(0.1)
  end
  
  # Test the constructor
  @testset "RandomNoisyNLPModel constructor" begin
    @test model.base_model == base_model
    @test model.noise_level == 0.1
  end

  # Test the obj function
  @testset "obj function" begin
    @test abs(obj(model, x) - obj_base) <= abs(obj_base) * model.noise_level * 1 / 2
    # Test the obj function with a given noise

    @test abs(obj(model, x, noise) - obj_base - noise) <= 1e-10
  end

  # Test the grad! function
  @testset "grad! function" begin
    g = similar(x)
    #testing the gradient
    grad!(model, x, g)

    for i ∈ eachindex(g)
      @test abs(g[i] - g_base[i]) <= abs(g_base[i]) * model.noise_level * 1 / 2
    end

    # Test the grad! function with a given noise

    g = similar(x)
    grad!(model, x, g, noise)

    for i ∈ eachindex(g)
      @test g[i] == g_base[i] .+ noise
    end

    # Test the grad function 
    g = grad(model, x)

    for i ∈ eachindex(g)
      @test abs(g[i] - g_base[i]) <= abs(g_base[i]) * model.noise_level * 1 / 2
    end

    # Test the grad function with a given noise

    g = grad(model, x, noise)

    for i ∈ eachindex(g)
      @test g[i] == g_base[i] .+ noise
    end
  end

end
