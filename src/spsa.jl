using QuantumToolbox
using CairoMakie
using HDF5

include(pwd()*"/src/low_rank.jl")
include(pwd()*"/src/transmon.jl")

function clamp_modulus!(θ::AbstractVector, max_mod::Real)
    n = div(length(θ), 2)
    for i in 1:n
        re = θ[i]
        im = θ[n + i]
        mod2 = re^2 + im^2
        max_mod2 = max_mod^2
        if mod2 > max_mod2
            scale = sqrt(max_mod2 / mod2)
            θ[i] *= scale
            θ[n + i] *= scale
        end
    end
end

"""
    spsa_optimize(
        objective::Function,
        θ_init::AbstractArray{T};
        max_iter::Int = 1000,
        a::Real = 0.1,
        c::Real = 0.1,
        A::Real = 100.0,
        α::Real = 0.602,
        γ::Real = 0.101,
    ) where {T<:Real}

Performs Simultaneous Perturbation Stochastic Approximation (SPSA) optimization.

Arguments:
- `objective`: Objective function to minimize.
- `θ_init`: Initial parameter vector.

Keyword arguments:
- `max_iter`: Maximum number of iterations (default: 1000).
- `a`: Learning rate coefficient (default: 0.1).
- `c`: Perturbation size coefficient (default: 0.1).
- `A`: Stability constant for learning rate decay (default: 100.0).
- `α`: Decay rate for learning rate (default: 0.602).
- `γ`: Decay rate for perturbation size (default: 0.101).
- `verbose`: If true, prints iteration details (default: true).

Returns:
- `θs`: Vector of parameter vectors at each iteration.
- `fs`: Vector of objective values per iteration.
"""
function spsa_optimize(
    objective::Function,
    θ_init::AbstractArray{T};
    max_iter::Int = 1000,
    a::Real = 0.1,
    c::Real = 0.1,
    A::Real = 100.0,
    α::Real = 0.602,
    γ::Real = 0.101,
    max_val::Real = Inf,
    min_val::Real = -Inf,
    l_tol::Real = 1e-6,
    n_stable_its::Int = 15,
    h5file::Union{Nothing,String} = nothing,
    verbose::Bool = true,
    plot_progress::Bool = false,
    plot_params::NamedTuple = (),
) where {T<:Real}
    θ = copy(θ_init)
    n_params = length(θ)
    fs = Float64[]
    θs = Vector{typeof(θ)}()

    # Initialize HDF5 datasets by overwriting if they exist
    if h5file !== nothing #&& !isfile(h5file)
        h5open(h5file, "w") do file
            create_dataset(file, "θs", datatype(Float64), (iter, n_params))
            create_dataset(file, "fs", datatype(Float64), (iter,))
        end
    end 

    println("done initializing h5 file: $h5file") # debug message

    no_improvement_count = 0
    
    prog_spsa = ProgressBar(max_iter)
    for k in 1:max_iter
        # Generate simultaneous perturbation vector
        Δ = 2 .* rand(Bool, n_params) .- 1
        
        # Compute adaptive step sizes
        c_k = c / k^γ
        a_k = a / (A + k)^α
        
        # Evaluate function at perturbed parameters
        θ_plus = θ .+ c_k .* Δ
        θ_minus = θ .- c_k .* Δ

        task_fp = Threads.@spawn objective(θ_plus)
        task_fm = Threads.@spawn objective(θ_minus)
        f_plus = fetch(task_fp)
        f_minus = fetch(task_fm)
        
        # Compute gradient estimate
        g_hat = (f_plus - f_minus) / (2 * c_k) .* Δ
        
        # Update parameters
        θ .-= a_k .* g_hat
        clamp_modulus!(θ[1:end], max_val) # restrict the modulus of the pulse heights

        # clamp!(θ, min_val, max_val) # restrict the values to be within bounds

        # Store function value and parameters
        loss = (f_plus + f_minus) / 2
        push!(fs, loss)
        push!(θs, copy(θ))

        # --- Append to JLD file after each iteration ---
        if h5file !== nothing
            h5open(h5file, "r+") do file
                # Append the current parameters and function value to the datasets
                file["θs"][k, :] = θs[end]
                file["fs"][k] = loss
            end
        end

        println("done writing to h5 file: $h5file") # debug message

        # add to no_improvement_count if the loss function does not improve by l_tol
        if k > 1 && abs(fs[end] - fs[end-1]) < l_tol
            no_improvement_count += 1
        else
            no_improvement_count = 0
        end

        if no_improvement_count >= n_stable_its
            println("Stopping optimization: The loss function did not improve by $l_tol for the last $n_stable_its iterations.")
            flush(stdout)
            break
        end

        # plot if plot_progress
        if plot_progress
            n_params_half = Int(div(n_params, 2))
            fig = Figure(size = (800, 400))
            ax = Axis(fig[1, 1], title = "SPSA Optimization Progress", xlabel = "Iteration", ylabel = "Objective Value")
            lines!(ax, 1:k, fs, color = :dodgerblue)
            scatter!(ax, k, fs[end], color = :red, markersize = 10)
            ax2 = Axis(fig[1, 2], xlabel = "Time", ylabel = "Pulse", title = "Final Pulse Profile")
            widths = (plot_params.tf - 2 * plot_params.t0) / n_params_half .* ones(n_params_half);
            lines!(ax2, plot_params.tlist, plot_params.func(θs[end][1:n_params_half], widths; t0 = plot_params.t0, k = plot_params.k).(Ref((t0 = plot_params.t0,)), plot_params.tlist), color = :dodgerblue)
            lines!(ax2, plot_params.tlist, plot_params.func(θs[end][n_params_half+1:end], widths; t0 = plot_params.t0, k = plot_params.k).(Ref((t0 = plot_params.t0,)), plot_params.tlist), color = :dodgerblue, linestyle=:dash)
            display(fig)
        end

        # Print iteration details if verbose
        if verbose
            println("\n---***---")
            println("Iteration: $k")
            println("Loss: $(fs[end])")
            println("Parameters: $(θs[end])")
            println("a_k: $(a_k)")
            println("c_k: $(c_k)")
            println("ΔL: $(abs(f_plus - f_minus))")
            println("Gradient ΔL/2c_k: $(abs(g_hat[1]))")
            println("Δθ: $(abs((a_k .* g_hat)[1]))")
            println("No improvement count: $no_improvement_count \n")
        end

        # Update progress bar
        next!(prog_spsa)
    end
    return θs, fs
end

function objective_function(p, mp)
    # unwrap the pulse sequence: p = [ωd, real_heights..., imag_heights...]
    # ωd = p[1] #* mp.ωd_scale
    # heights = p[1:end] # the rest are heights
    n = div(length(p), 2)
    real_heights = p[1:n]
    imag_heights = p[n+1:end]

    # do first with constant widths
    widths = (mp.tf - 2 * mp.t0) / n .* ones(n)
    pulse_sequence = multi_step_function_smooth(real_heights .+ im * imag_heights, widths; t0 = mp.t0, k=mp.k)

    drive_params = (t0 = mp.t0, ωd = mp.ωd);
    tlist = 0:0.1:mp.tf;

    H, c_ops, a, f = get_H_tuples(mp.Nt, mp.N1, mp.N2, mp.Nφ, mp.Ec, mp.Ej, mp.g, mp.ωa, mp.ωf, mp.J, mp.ωd, mp.εd, mp.κ, mp.γ, pulse_sequence);

    # sol_g = solve_lr(H, mp.ψg, mp.M, tlist, c_ops; params = drive_params);
    # sol_e = solve_lr(H, mp.ψe, mp.M, tlist, c_ops; params = drive_params);

    transmon_states = sum([ket2dm(basis(mp.N1, 0) ⊗ basis(mp.N2, 0) ⊗ basis(mp.Nt, k)) for k in 2:mp.Nt-1])

    task_g = Threads.@spawn solve_lr(H, mp.ψg, mp.M, tlist, c_ops; e_ops = [f, transmon_states], params = drive_params, progress_bar = false);
    task_e = Threads.@spawn solve_lr(H, mp.ψe, mp.M, tlist, c_ops; e_ops = [f, transmon_states], params = drive_params, progress_bar = false);

    sol_g = fetch(task_g);
    sol_e = fetch(task_e);

    βg = sol_g.expect[1];
    βe = sol_e.expect[1];

    # calculate the fidelity
    err = assignment_error(βg, βe, tlist, mp.η, mp.κ, mp.γ)[end];
    # snr = signal_to_noise(βg, βe, tlist, mp.η, mp.κ, mp.γ)[end];

    # make the total loss function
    ks_g = sol_g.expect[2];
    ks_e = sol_e.expect[2];

    loss = err + forbidden_states_transmon(ks_g, ks_e, tlist) #+ 5000 * forbidden_states_undriven(2, ρg, ρe, tlist)

    return loss
end;

#=
Obsolete code, kept for reference
=#

"""
    gradient_descent(objective, θ_init; max_iter=100, lr=0.01, δ=1e-6, β1=0.9, β2=0.999, ϵ=1e-8, gtol=1e-6, n_stable_its=10)

    Performs gradient descent optimization with ADAM updates and finite-difference gradient estimation, supporting multi-threaded computation.
    
    # Arguments
    - `objective`: A function that takes a parameter vector and returns a scalar loss value.
    - `θ_init`: Initial parameter vector.
    
    # Keyword Arguments
    - `max_iter`: Maximum number of iterations (default: 100).
    - `lr`: Learning rate for the ADAM optimizer (default: 0.01).
    - `δ`: Finite difference step size for gradient estimation (default: 1e-6).
    - `β1`: Exponential decay rate for the first moment estimates in ADAM (default: 0.9).
    - `β2`: Exponential decay rate for the second moment estimates in ADAM (default: 0.999).
    - `ϵ`: Small constant for numerical stability in ADAM (default: 1e-8).
    - `gtol`: Gradient norm threshold for convergence (default: 1e-6).
    - `n_stable_its`: Number of consecutive iterations with gradient below `gtol` required to stop (default: 10).
    
    # Returns
    - `θs`: Vector of parameter vectors at each iteration.
    - `fs`: Vector of objective function values at each iteration.
    
    # Notes
    - The gradient is estimated using central finite differences and parallelized across available threads.
    - Parameters are clamped to be non-negative after each update.
    - Optimization stops early if the gradient remains below `gtol` for `n_stable_its` consecutive iterations.
"""
function gradient_descent(objective, θ_init; max_iter=100, lr=0.01, δ=1e-6, β1=0.9, β2=0.999, ϵ=1e-8, gtol=1e-6, n_stable_its=10)
    θ = copy(θ_init)
    n = length(θ)
    nt = Threads.nthreads()
    grad = similar(θ)
    fs = Float64[]
    θs = Vector{typeof(θ)}()

    θ_plus = similar(θ, n, nt)
    θ_minus = similar(θ, n, nt)

    m = zeros(length(θ))
    v = zeros(length(θ))

    prog = ProgressBar(max_iter)
    no_improvement_count = 0

    # parallelize the components
    arr = collect(enumerate(Iterators.partition(1:n, cld(n, nt))))

    for k in 1:max_iter
        θ_plus .= θ
        θ_minus .= θ
        Threads.@threads for (tid, idxs) in arr
            for i in idxs
                θ_plus[i, tid] += δ
                θ_minus[i, tid] -= δ
                obj_plus = objective(selectdim(θ_plus, 2, tid)) # create a view
                obj_minus = objective(selectdim(θ_minus, 2, tid)) # create a view
                grad[i] = (obj_plus - obj_minus) / (2δ)
            end
        end

        # ADAM update
        m .= β1 .* m .+ (1 - β1) .* grad
        v .= β2 .* v .+ (1 - β2) .* (grad .^ 2)
        m_hat = m ./ (1 - β1^k)
        v_hat = v ./ (1 - β2^k)
        θ .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ϵ)

        clamp!(θ, -max_height, max_height) # restrict the values to be positive
        push!(fs, objective(θ))
        push!(θs, copy(θ))
        println("\n---***---")
        println("Iteration: $k")
        println("Loss: $(fs[end])")
        println("Parameters: $(θs[end])")
        println("Gradient: $(norm(grad)/n)\n")

        if maximum(abs, grad) < gtol
            no_improvement_count += 1
        else
            no_improvement_count = 0
        end

        if no_improvement_count >= n_stable_its
            println("Stopping optimization: The gradient remained below the threshold $gtol for the last $n_stable_its iterations.")
            flush(stdout)
            break
        end

        next!(prog)
    end
    return θs, fs
end;