module NoisyNLPModels

export RandomNoisyNLPModel
export update_noise_level!
using NLPModels
import NLPModels.obj
import NLPModels.grad!
import NLPModels.grad

# See https://jso.dev/NLPModels.jl/stable/guidelines/#bare-minimum
abstract type AbstractNoisyNLPModel{T, S} <: AbstractNLPModel{T, S} end

mutable struct RandomNoisyNLPModel{T, S, M <: AbstractNLPModel{T, S}} <: AbstractNoisyNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  noise_level::T
  base_model::M
end

# constructor
function RandomNoisyNLPModel(model::AbstractNLPModel{T, S}, noise_level::T) where {T, S}
  M = typeof(model)
  meta = model.meta
  counters = Counters()
  RandomNoisyNLPModel{T, S, M}(meta, counters, max(noise_level, zero(T)), model)
end

RandomNoisyNLPModel(model::AbstractNLPModel{T, S}, noise_level) where {T, S} =
  RandomNoisyNLPModel(model, convert(T, noise_level))

# noisy objective
function obj(model::RandomNoisyNLPModel{T, S, M}, x::S) where {T, S <: (AbstractVector), M}
  base_value = obj(model.base_model, x)
  return base_value + abs(base_value) * model.noise_level * (rand(T) - 1 / 2)
end

# noisy objective with given noise by user (not the one stored in the model)
function obj(
  model::RandomNoisyNLPModel{T, S, M},
  x::S,
  noise::T,
) where {T, S <: (AbstractVector), M}
  return obj(model.base_model, x) + noise
end

# noisy gradient
function grad!(model::RandomNoisyNLPModel{T, S, M}, x::S, g::S) where {T, S <: (AbstractVector), M}
  grad!(model.base_model, x, g)
  for i ∈ eachindex(g)
    g[i] += abs(g[i]) * model.noise_level * (rand(T) - 1 / 2)
  end
  g
end

# noisy gradient with given noise by user (not the one stored in the model)
function grad!(
  model::RandomNoisyNLPModel{T, S, M},
  x::S,
  g::S,
  noise::T,
) where {T, S <: (AbstractVector), M}
  grad!(model.base_model, x, g)
  for i ∈ eachindex(g)
    g[i] += noise
  end
  g
end

function grad(
  model::RandomNoisyNLPModel{T, S, M},
  x::S,
  noise::T,
) where {T, S <: (AbstractVector), M}
  g = grad(model.base_model, x)
  for i ∈ eachindex(g)
    g[i] += noise
  end
  g
end

function update_noise_level!(
  model::RandomNoisyNLPModel{T, S, M},
  noise_level::T,
) where {T, S <: (AbstractVector), M}
  model.noise_level = max(0,noise_level)
end

end
