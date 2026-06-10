## Run all (alt+enter)
using Advectra
using CUDA

# Linear operator
function Linear!(du, u, operators, p, t)
    @unpack laplacian, hyper_laplacian = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack ν, μ, ν_h, μ_h = p
    dη .= ν * laplacian(η) + ν_h*hyper_laplacian(η)
    dΩ .= μ * laplacian(Ω) + μ_h*hyper_laplacian(Ω)
end

# Non-linear operator, fully non-linear
function NonLinear!(du, u, operators, p, t)
    @unpack solve_phi, poisson_bracket, diff_y = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack ζ, σ = p
    ϕ = solve_phi(η, Ω)

    dη .= poisson_bracket(η, ϕ) - (1 - ζ) * diff_y(ϕ) - ζ * diff_y(η) + σ * ϕ
    dΩ .= poisson_bracket(Ω, ϕ) - ζ * diff_y(η) + σ * ϕ
end

gammas = [0.20705825, 0.20496542, 0.20082876, 0.19619285, 0.18968598, 0.17695927,
    0.16294017, 0.14378273, 0.10855735, 0.07417678]
sigmas = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1]

Ns = [128, 256, 512]#, 1024]
Ls = [24, 36, 48]

if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    idx = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    N = Ns[mod1(idx, length(Ns))]
    L = Ls[cld(idx, length(Ns))]*128/N

    sigmas = sigmas[4] # 0.01 fixed
    gammas = gammas[4]
elseif length(ARGS) == 1
    idx = parse(Int, first(ARGS))
    sigmas = sigmas[idx]
    gammas = gammas[idx]
end

domain = Domain(N; L=L, MemoryType=CuArray)
ic = initial_condition(random_crossphased, domain; value=1e-3, include_zonal=true, include_streamer=true)

scale = (256/N)

# Diagnostics
diagnostics = @diagnostics [
    progress(; stride=floor(Int, 5000/Main.scale)),
    
    probe_all(; positions=[(x, 0) for x in LinRange(-Main.L÷2, 0.8*Main.L÷2, 10)], stride=floor(Int, 10/Main.scale)),
    
    kinetic_energy_integral(; stride=floor(Int, 50/Main.scale)),
    potential_energy_integral(; stride=floor(Int, 50/Main.scale)),
    zonal_kinetic_energy_integral(; stride=floor(Int, 50/Main.scale)),
    streamer_kinetic_energy_integral(; stride=floor(Int, 50/Main.scale)),
    enstrophy_energy_integral(; stride=floor(Int, 50/Main.scale)),
    radial_flux(; stride=floor(Int, 50/Main.scale)),

    cfl(; stride=floor(Int, 5000/Main.scale), silent=true),
    
    kinetic_energy_spectrum(stride=floor(Int, 5000/Main.scale) , spectrum=:poloidal),
    kinetic_energy_spectrum(stride=floor(Int, 5000/Main.scale) , spectrum=:wavenumber),
    potential_energy_spectrum(stride=floor(Int, 5000/Main.scale) , spectrum=:poloidal),
    potential_energy_spectrum(stride=floor(Int, 5000/Main.scale) , spectrum=:wavenumber),
    flux_spectrum(stride=floor(Int, 5000/Main.scale) , spectrum=:poloidal),
    flux_spectrum(stride=floor(Int, 5000/Main.scale) , spectrum=:wavenumber),
    
    sample_density(; storage_limit="0.1 GB"),
    sample_vorticity(; storage_limit="0.1 GB"),
    sample_potential(; storage_limit="0.1 GB")
]

for (σ, γ) in zip(sigmas, gammas)
    # Parameters
    parameters = (ζ=1e-1, σ=σ, ν=1e-2, μ=1e-2, ν_h=2e-6, μ_h=2e-6)

    # Time parameters
    dt = scale*2e-4 / γ
    tspan = [0.0, 50_000_000*dt/(4*scale)] # 10_000_000

    # Collection of specifications defining the problem to be solved
    prob = SpectralODEProblem(Linear!, NonLinear!, ic, domain, tspan; p=parameters, dt=dt,
        operators=:all, diagnostics=diagnostics, additional_operators=[OperatorRecipe(:laplacian; order=3, alias=:hyper_laplacian)])

    # Output
    output = Output(prob; filename="/cluster/work/projects/nn12110k/joemork/GD-sheath-scan/GDSI_N-$(N)_L-$(L).h5",
        simulation_name=:parameters, resume=true, storage_limit="100 GB")

    println(stderr, "Running simulation for σ=$σ with γ=$γ (N=$N, L=$L):")

    ## Solve and plot
    sol = spectral_solve(prob, MSS3(), output)

    close(output)
end