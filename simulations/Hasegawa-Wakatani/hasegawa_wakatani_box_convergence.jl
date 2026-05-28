## Run all (alt+enter)
using Advectra
using CUDA

# Linear operator
function Linear!(du, u, operators, parameters, t)
    @unpack hyper_laplacian = operators
    n, Ω = eachslice(u; dims=3)
    dn, dΩ = eachslice(du; dims=3)
    @unpack ν, μ = parameters
    dn .= ν * hyper_laplacian(n)
    dΩ .= μ * hyper_laplacian(Ω)
end

# Non-linear operator, fully non-linear
function NonLinear!(du, u, operators, parameters, t)
    @unpack solve_phi, poisson_bracket, diff_y = operators
    n, Ω = eachslice(u; dims=3)
    dn, dΩ = eachslice(du; dims=3)
    @unpack C = parameters
    ϕ = solve_phi(Ω)

    dn .= poisson_bracket(n, ϕ) - diff_y(ϕ) + C * (ϕ - n)
    dΩ .= poisson_bracket(Ω, ϕ) + C * (ϕ - n)
end

# Diagnostics
diagnostics = @diagnostics [
    progress(; stride=5000),
    # Probes
    probe_all(; positions=[(x, 0) for x in LinRange(-π / 0.15, 16.755160819145562, 10)], stride=10),
    # Energy integrals
    kinetic_energy_integral(; stride=50),
    potential_energy_integral(; stride=50),
    # CFL
    cfl(; stride=5000, silent=true),
    # Spectrums
    kinetic_energy_spectrum(stride=500 , spectrum=:poloidal),
    kinetic_energy_spectrum(stride=500 , spectrum=:wavenumber),
    potential_energy_spectrum(stride=500 , spectrum=:poloidal),
    potential_energy_spectrum(stride=500 , spectrum=:wavenumber),
    flux_spectrum(stride=500 , spectrum=:poloidal),
    flux_spectrum(stride=500 , spectrum=:wavenumber),
    # Fields
    sample_density(; storage_limit="0.1 GB"),
    sample_vorticity(; storage_limit="0.1 GB"),
    sample_potential(; storage_limit="0.1 GB")
]

gammas = [0.2, 0.13880677, 0.14947481, 0.13908339, 0.1068677,  0.06650772, 0.05]
Cs = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
Ls = [1/2, 2].*2π/0.15
suffix = ["half", "double"]
L = Ls[2]

if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    idx = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    Cs = Cs[mod1(idx,7)]
    gammas = gammas[mod1(idx,7)]
    L = LS[cld(idx,7)]
elseif length(ARGS) == 1
    idx = parse(Int, first(ARGS))
    Cs = Cs[idx]
    gammas = gammas[idx]
end

domain = Domain(256, 256; Lx=L, Ly=L, MemoryType=CuArray)
ic = initial_condition(random_crossphased, domain; value=1e-3, include_zonal=true, include_streamer=true)

for (C, γ) in zip(Cs, gammas)
    # Parameters
    parameters = (C=C, ν=1e-4, μ=1e-4)

    # Time parameters
    dt = 2e-4 / γ
    tspan = [0.0, 50_000_000÷4 * dt] # 10_000_000

    # Collection of specifications defining the problem to be solved
    prob = SpectralODEProblem(Linear!, NonLinear!, ic, domain, tspan; p=parameters, dt=dt,
        operators=:all, diagnostics=diagnostics, additional_operators=[OperatorRecipe(:laplacian; order=3, alias=:hyper_laplacian)])

    # Output
    output = Output(prob; filename="/cluster/work/projects/nn12110k/joemork/Hasegawa-Wakatani/HW_C-$(C)_hyper_$(suffix).h5",
        simulation_name=:parameters, resume=true, storage_limit="10 GB")

    println("Running simulation for C=$C with γ=$γ:")

    ## Solve and plot
    sol = spectral_solve(prob, MSS3(), output)

    close(output)
end
