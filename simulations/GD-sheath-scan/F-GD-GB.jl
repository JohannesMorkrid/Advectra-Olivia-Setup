
## Run all (alt+enter)
using Advectra
using CUDA

domain = Domain(256, 256; Lx=48, Ly=48, MemoryType=CuArray, precision=Float64)
ic = initial_condition(random_crossphased, domain; value=1e-3)

# Linear operator
function Linear(du, u, operators, p, t)
    @unpack laplacian, diff_x = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack ν, μ = p
    dη .= ν * laplacian(η)
    dΩ .= μ * laplacian(Ω)
end

# Non-linear operator, fully non-linear
function NonLinear(du, u, operators, p, t)
    @unpack solve_phi, diff_x, diff_y = operators
    @unpack poisson_bracket, grad_dot_grad, spectral_expm1 = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack ζ, σ, ν, κ = p
    ϕ = solve_phi(η, Ω)

    dη .= poisson_bracket(η, ϕ) - (1 - ζ) * diff_y(ϕ) - ζ * diff_y(η) +
          ν * κ * grad_dot_grad(η, η) - 2ν * κ * diff_x(η) - (σ / κ) * spectral_expm1(-κ * ϕ)
    dΩ .= poisson_bracket(Ω, ϕ) - ζ * diff_y(η) - (σ / κ) * spectral_expm1(-κ * ϕ)

    CUDA.@allowscalar dη[1] = 0
    CUDA.@allowscalar dΩ[1] = 0
end

# Parameters
κ = 1e-2
parameters = (κ=κ, ζ=1e-1, σ=1e-1, ν=1e-2, μ=1e-2)

# Time intervalparameters
tspan = [0.0, 100_000] # 10_000_000.0]

# Diagnostics
diagnostics = @diagnostics [
    progress(; stride=1000),
    probe_all(; positions=[(x, 0) for x in LinRange(-24, 19.2, 10)], stride=100),
    get_log_modes(; stride=50, axis=:diag),
    kinetic_energy_integral(; stride=500),
    potential_energy_integral(; stride=500),
    cfl(; stride=5000, silent=true),
    sample_density(; storage_limit="15 GB"),
    sample_vorticity(; storage_limit="15 GB"),
]

# Collection of specifications defining the problem to be solved
prob = SpectralODEProblem(Linear, NonLinear, ic, domain, tspan; p=parameters, dt=1e-3,
    operators=:all, diagnostics=diagnostics)

# Inverse transform
#inverse_transformation!(u) = @. u[:, :, 1] = exp(u[:, :, 1]) - 1

# Output
output = Output(prob; filename="/cluster/work/projects/nn12110k/GD-sheath-scan/F-GD-GB.h5", 
    simulation_name=:parameters, resume=true, #physical_transform=inverse_transformation!, 
    storage_limit="50 GB")

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

close(output)
