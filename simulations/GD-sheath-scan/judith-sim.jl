## Run all (alt+enter)
using Advectra
using CUDA

domain = Domain(512, 512; Lx=48, Ly=48, MemoryType=CuArray)
ic = initial_condition(random_crossphased, domain; value=1e-3, include_zonal=true, include_streamer=true)

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
    @unpack solve_phi, poisson_bracket, diff_y = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack ζ, σ = p
    ϕ = solve_phi(η, Ω)

    dη .= poisson_bracket(η, ϕ) - (1 - ζ) * diff_y(ϕ) - ζ * diff_y(η) + σ * ϕ
    dΩ .= poisson_bracket(Ω, ϕ) - ζ * diff_y(η) + σ * ϕ

    CUDA.@allowscalar dη[1] = 0
    CUDA.@allowscalar dΩ[1] = 0
    remove_nyquist_modes!(du, domain)
end

# Diagnostics
diagnostics = @diagnostics [
    progress(; stride=5000),
    sample_density(; stride=25),
    sample_vorticity(; stride=25),
    sample_potential(; stride=25)
]

gammas = [0.20705825, 0.20496542, 0.20082876, 0.19619285, 0.18968598, 0.17695927,
    0.16294017, 0.14378273, 0.10855735, 0.07417678]
sigmas = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1]

if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    idx = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    sigmas = sigmas[idx]
    gammas = gammas[idx]
elseif length(ARGS) == 1
    idx = parse(Int, first(ARGS))
    sigmas = sigmas[idx]
    gammas = gammas[idx]
end

for (σ, γ) in zip(sigmas, gammas)
    # Parameters
    parameters = (ζ=1e-1, σ=σ, ν=1e-2, μ=1e-2)

    # Time parameters
    dt = 1e-4 / γ
    tspan = [0.0, 75_000 * dt] # 10_000_000

    # Collection of specifications defining the problem to be solved
    prob = SpectralODEProblem(Linear, NonLinear, ic, domain, tspan; p=parameters, dt=dt,
        operators=:all, diagnostics=diagnostics)

    # Output
    output = Output(prob; filename="/cluster/work/projects/nn12110k/GD-sheath-scan/judith_sim.h5",
        simulation_name=:parameters, resume=true, storage_limit="50 GB")

    println("Running simulation for σ=$σ with γ=$γ:")

    ## Solve and plot
    sol = spectral_solve(prob, MSS3(), output)

    close(output)
end