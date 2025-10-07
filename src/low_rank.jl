using OrdinaryDiffEq, DiffEqCallbacks
using LinearAlgebra, SparseArrays

struct TimeEvolutionLRSol{
    TT,
    TS,
    TE,
}
    times::TT
    states::TS
    expect::TE
    Ms::Vector{Int}
end

"""
    initialize_m0(N::Int, M::Int, initial_state::AbstractVector; ϵ::Real = 1e-5)
Initializes the low-rank state matrix `m0` with dimensions `N x M`, where the first column is aligned with the provided `initial_state`.
The remaining columns are filled with random vectors orthonormalized against the first column using the Gram-Schmidt process.
- `N`: Dimension of the Hilbert space.
- `M`: Number of low-rank states.
- `initial_state`: The initial state vector to align the first column of `m0`.
- `ϵ`: Small value to scale the random vectors for numerical stability (default is `1e-5`).
Returns the initialized matrix `m0`.
"""
function initialize_m0(N::Int, M::Int, initial_state::AbstractVector; ϵ::Real = 1e-5)
    # Ensure the initial state is normalized
    initial_state *= sqrt((1 - (M - 1) * ϵ^2)) / norm(initial_state)
    
    # Create a random matrix for the remaining columns
    random_matrix = rand(ComplexF64, N, M - 1)
    
    # Combine the initial state with the random matrix
    m0 = hcat(initial_state, random_matrix)
    
    # Apply Gram-Schmidt process to orthonormalize the columns
    for i in 2:M
        for j in 1:i-1
            m0[:, i] -= (m0[:, j]' * m0[:, i]) * m0[:, j]
        end
        m0[:, i] *= ϵ / norm(m0[:, i])
    end
    
    return m0
end

"""
    expval_compute(u, t, integrator)
Computes the expectation values of the operators specified in `integrator.p.e_ops` with respect to the current low-rank state `m`.
- `u`: Current state vector from the integrator (not used in this function).
- `t`: Current time (not used in this function).
- `integrator`: The integrator object containing parameters and state information.
Returns a vector of expectation values for each operator in `integrator.p.e_ops`.
Note: the operators in `integrator.p.e_ops` should be provided as matrices (not as `QuantumObject`).
"""
function expval_compute(u, t, integrator)
    N, M = integrator.p.N, integrator.p.M 
    m = reshape(integrator.u, N, M)
    [tr(m' * op * m) for op in integrator.p.e_ops]
end

#=
Adaptive rank callbacks 
=#

"""
    incM_condition(u, t, integrator)
Condition function to determine whether to increase the rank of the low-rank state matrix `m`.
The rank is increased if the ratio of the smallest to largest eigenvalue of `m' * m` exceeds `err_max`.
- `u`: Current state vector from the integrator.
- `t`: Current time (not used in this function).
- `integrator`: The integrator object containing parameters and state information.
Returns `true` if the rank should be increased, `false` otherwise.
"""
function incM_condition(u, t, integrator)
    # flag to deal with rank being bigger than N
    if integrator.p.M == integrator.p.N
        return false
    end
    # I need this cache because I can't reshape directly the integrator.u -- CHECK if this is still true in the latest version of DifferentialEquations.jl
    copyto!(integrator.p.u_cache, u)
    m = reshape(integrator.p.u_cache, integrator.p.N, integrator.p.M)
    err_max = integrator.p.err_max
    λs = eigvals(Hermitian(m' * m))

    # # ensure that the new column will not trigger a continuous rank increase/decrease loop
    # if integrator.p.ϵ^2 / λs[end] < integrator.p.err_min
    #     throw(ErrorException("The weight of the new column ϵ is too small and will constantly trigger a rank decrease. Consider increasing ϵ or decreasing err_min."))
    # end

    return abs(λs[1]./λs[end]) > err_max # if p_M / p_1 > err_max is true, increase rank
end

"""
    increase_rank!(integrator)
Increases the rank of the low-rank state matrix `m` by adding a new orthonormal column.
The new column is generated as a random vector orthonormalized against the existing columns using the Gram-Schmidt process.
- `integrator`: The integrator object containing parameters and state information.
This function modifies the integrator in place, updating the state vector and parameters to reflect the increased rank.
"""
function increase_rank!(integrator)

    ip = integrator.p
    N, M = ip.N, ip.M

    copyto!(integrator.p.u_cache, integrator.u) 
    m = reshape(integrator.p.u_cache, N, M)

    # Generate a random vector to add as a new column
    new_col = randn(ComplexF64, N)
    # Orthonormalize against all columns of m -- CHECK that this doesn't introduce more allocations 
    for j in 1:M
        new_col -= (m[:, j]' * new_col) * m[:, j]
    end
    new_col .*= ip.ϵ ./ norm(new_col)
    m_new = hcat(m, new_col)

    # normalize!(m_new) # no need to normalize because we do it in dmdt!

    # resize!(integrator, length(m))
    # integrator.u[:] .= reshape(m_new[:, 1:end-1], :)

    resize!(integrator, length(m_new))
    integrator.u[:] .= reshape(m_new, :)

    # update parameters
    integrator.p = merge(integrator.p, 
        (M = ip.M + 1,
        Lm = similar(m_new),
        temp_MM = similar(m_new, M + 1, M + 1),
        temp_m = similar(m_new),
        u_cache = similar(m_new, (M + 1) * N),
    ))
end

"""
    decM_condition(u, t, integrator)
Condition function to determine whether to decrease the rank of the low-rank state matrix `m`.
The rank is decreased if the ratio of the smallest to largest eigenvalue of `m' * m` is less than `err_min`.
- `u`: Current state vector from the integrator.
- `t`: Current time (not used in this function).
- `integrator`: The integrator object containing parameters and state information.
Returns `true` if the rank should be decreased, `false` otherwise.
"""
function decM_condition(u, t, integrator)
    # flag to deal with rank being smaller than 1
    if integrator.p.M == 1
        return false
    end
    # I need this cache because I can't reshape directly the integrator.u
    copyto!(integrator.p.u_cache, u)
    m = reshape(integrator.p.u_cache, integrator.p.N, integrator.p.M)
    err_min = integrator.p.err_min
    λs = eigvals(Hermitian(m' * m))

    return abs(λs[1]./λs[end]) < err_min # if p_M / p_1 < err_min is true, decrease rank
end

"""
    decrease_rank!(integrator)
Decreases the rank of the low-rank state matrix `m` by removing the last column.
- `integrator`: The integrator object containing parameters and state information.
This function modifies the integrator in place, updating the state vector and parameters to reflect the decreased rank.
"""
function decrease_rank!(integrator)

    ip = integrator.p
    N, M = ip.N, ip.M

    copyto!(integrator.p.u_cache, integrator.u) 
    m = reshape(integrator.p.u_cache, N, M)

    m_new = m[:, 1:end-1]

    # normalize!(m_new) # no need to normalize because we do it in dmdt!

    # resize!(integrator, length(m))
    # integrator.u[:] .= reshape(m_new[:, 1:end-1], :)

    resize!(integrator, length(m_new))
    integrator.u[:] .= reshape(m_new, :)

    # update parameters
    integrator.p = merge(integrator.p, 
        (M = ip.M - 1,
        Lm = similar(m_new),
        temp_MM = similar(m_new, M - 1, M - 1),
        temp_m = similar(m_new),
        u_cache = similar(m_new, (M - 1) * N),
    ))
end

#=
Low-rank evolution
=#

"""
    dmdt!(dm, m, p, t)
Computes the time derivative of the low-rank state matrix `m` according to the non-stochastic Schrodinger equation.
- `dm`: Output matrix to store the time derivative of `m`.
- `m`: Current low-rank state matrix.
- `p`: Named tuple containing parameters such as the Hamiltonian `H`, collapse operators `L`, and temporary matrices for computation.
- `t`: Current time (not used in this function).
This function modifies `dm` in place to contain the time derivative of `m`.
Note: the Hamiltonian is time-independent in this version. For time-dependent Hamiltonians, use `dmdt_tuple!`.
"""
function dmdt!(dm, m, p, t)
    # reshape m and dm to matrices
    N, M = p.N, p.M
    @views m = reshape(m, N, M)
    @views dm = reshape(dm, N, M)

    # normalize -- divide by sqrt(tr(m' * m))
    normalize!(m)

    # unwrap parameters
    temp_MM = p.temp_MM
    temp_m = p.temp_m
    L = p.L

    # unitary evolution
    H = p.H
    mul!(dm, H, m, -1im, false)
    
    # dissipative evolution
    Lm = p.Lm
    
    @inbounds for L in p.L
        mul!(Lm, L, m, true, false) # 0 allocations

        copyto!(temp_m, m) # 0 allocations
        fac = qr!(temp_m) # 20512 (14800 w/ ColumNorm()) -- removing column norm helps with allocations on next line
        ldiv!(temp_MM, fac, Lm) # 30416 (111392 w/ ColumNorm()) -- removing column norm helps with allocations
    
        mul!(dm, Lm, adjoint(temp_MM), 0.5, true) # 48 allocations -- we don't temp because the speed tradeoff is favourable
        mul!(dm, adjoint(L), Lm, -0.5, true) # 48 allocations
    end

    dm[:] = vec(dm)
end

"""
    dmdt_tuple!(dm, m, p, t)
Computes the time derivative of the low-rank state matrix `m` according to the non-stochastic Schrodinger equation with a time-dependent Hamiltonian.
- `dm`: Output matrix to store the time derivative of `m`.
- `m`: Current low-rank state matrix.
- `p`: Named tuple containing parameters such as the Hamiltonian `H`, collapse operators `L`, and temporary matrices for computation.
- `t`: Current time.
This function modifies `dm` in place to contain the time derivative of `m`.
Note: the Hamiltonian is time-dependent in this version -- a collection of operators and coefficient functions. For time-independent Hamiltonians, use `dmdt!`.
"""
function dmdt_tuple!(dm, m, p, t)
    # reshape m and dm to matrices
    N, M = p.N, p.M
    @views m = reshape(m, N, M)
    @views dm = reshape(dm, N, M)

    normalize!(m)

    # unwrap parameters
    temp_MM = p.temp_MM
    temp_m = p.temp_m
    L = p.L
    drive_params = p.drive_params

    # unitary evolution
    H = p.H
    mul!(dm, H[1], m, -1im, false)
    @inbounds for Hi in H[2:end]
        mul!(dm, Hi[1], m, -1im * Hi[2](drive_params, t), true) # TODO: speed up by getting rid of * multiplication -- still need to optimize
    end
    
    # dissipative evolution
    Lm = p.Lm
    
    @inbounds for L in p.L
        mul!(Lm, L, m, true, false) # 0 allocations

        copyto!(temp_m, m) # 0 allocations
        fac = qr!(temp_m) # 20512 (14800 w/ ColumNorm()) -- removing column norm helps with allocations on next line
        ldiv!(temp_MM, fac, Lm) # 30416 (111392 w/ ColumNorm()) -- removing column norm helps with allocations
    
        mul!(dm, Lm, adjoint(temp_MM), 0.5, true) # 48 allocations -- we don't temp because the speed tradeoff is favourable
        mul!(dm, adjoint(L), Lm, -0.5, true) # 48 allocations
    end

    # reshape dm back to vector
    dm[:] = vec(dm)
end

# TODO: adapt everything to Qobj and QobjEvo
# TODO: reduce allocations and memory usage
# TODO: cleanup some of the arguments into options
"""
    solve_lr(H::Tuple, ψ0::QuantumObject, M::Int, tlist::AbstractVector, c_ops::AbstractVector;
             e_ops::AbstractVector = [], params::NamedTuple = NamedTuple(), progress_bar::Bool = true)
Solves the time evolution of a quantum system using low-rank methods.
- `H`: Hamiltonian of the system, given as a tuple where the first element is the time-independent part and the rest are time-dependent parts.
- `ψ0`: Initial state of the system, represented as a `QuantumObject`.
- `M`: Number of low-rank states to use in the evolution. If adapt_increase or adapt_decrease is true, this is the initial rank.
- `tlist`: Vector of time points at which to evaluate the solution.
- `c_ops`: List of collapse operators for the system.
- `e_ops`: List of operators for which to compute expectation values (optional).
- `adapt_increase`: Boolean flag to enable or disable adaptive rank increase (default is `true`).
- `adapt_decrease`: Boolean flag to enable or disable adaptive rank decrease (default is `false`).
- `err_max`: Threshold for increasing the rank based on the ratio of eigenvalues (default is `1e-4`).
- `err_min`: Threshold for decreasing the rank based on the ratio of eigenvalues (default is `1e-10`).
- `params`: Named tuple of parameters for the time-dependent Hamiltonian (optional).
- `progress_bar`: Boolean flag to enable or disable the progress bar (default is `true`).
Returns a `TimeEvolutionLRSol` object containing the time points, low-rank states, expectation values, and rank.
"""
function solve_lr(
    H::Union{QuantumObject, AbstractMatrix}, ψ0::QuantumObject, M::Int, tlist::AbstractVector, c_ops::AbstractVector;
    e_ops::AbstractVector = [], params::NamedTuple = NamedTuple(), adapt_increase = true, adapt_decrease = false, 
    err_max = 1e-4, err_min = 1e-10, progress_bar::Bool = true)

    N = length(ψ0)
    ϵ = sqrt(err_max) * 1e-2 # to not trigger an automatic rank increase
    m0 = initialize_m0(N, M, ψ0.data; ϵ = ϵ)
    p = (
        N = N,
        M = M,
        H = get_data(H),
        L = [get_data(L) for L in c_ops],
        drive_params = params, # this is useless if H is not a tuple
        Lm = similar(m0),
        temp_MM = similar(m0, M, M),
        temp_m = similar(m0),
        e_ops = [get_data(op) for op in e_ops],
        u_cache = similar(m0, M * N),
        err_max = err_max,
        err_min = err_min,
        ϵ = ϵ,
    )

    # Progress bar callback
    prog = ProgressBar(length(tlist), enable = progress_bar)
    prog_cb = FunctionCallingCallback((u, t, integrator) -> next!(prog); funcat = tlist)

    # Expectation value callback
    exp_cb = nothing
    saved_values = nothing
    if !isempty(e_ops)
        saved_values = SavedValues(Float64, Vector{ComplexF64})
        exp_cb = SavingCallback(expval_compute, saved_values; saveat=tlist)
    end

    # Rank increase callback
    if adapt_increase
        increase_cb = DiscreteCallback(
            (u, t, integrator) -> incM_condition(u, t, integrator),
            increase_rank!,
            save_positions = (false, false)
        )
    else
        increase_cb = nothing
    end

    # Rank decrease callback
    if adapt_decrease
        decrease_cb = DiscreteCallback(
            (u, t, integrator) -> decM_condition(u, t, integrator),
            decrease_rank!,
            save_positions = (false, false)
        )
    else
        decrease_cb = nothing
    end

    # Save rank callback
    saved_M = SavedValues(Float64, Int)
    M_cb = SavingCallback((u, t, integrator) -> integrator.p.M, saved_M; saveat=tlist)

    # Combine callbacks
    callbackset = CallbackSet(exp_cb, increase_cb, decrease_cb, prog_cb, M_cb)

    prob = ODEProblem{true}(dmdt!, vec(m0), (tlist[1], tlist[end]), p)
    sol = solve(prob, Tsit5();
        saveat=tlist,
        callback=callbackset,
        save_everystep=false,
        save_start=true,
        dense=false
    )

    expect = nothing
    if !isempty(e_ops)
        expect_raw = saved_values.saveval
        expect = Vector([getindex.(expect_raw, i) for i in axes(e_ops)[1]])
    end

    return TimeEvolutionLRSol(tlist, sol.u, expect, saved_M.saveval)
end

function solve_lr(
    H::Tuple, ψ0::QuantumObject, M::Int, tlist::AbstractVector, c_ops::AbstractVector;
    e_ops::AbstractVector = [], params::NamedTuple = NamedTuple(), adapt_increase = true, adapt_decrease = false, 
    err_max::Real = 1e-4, err_min = 1e-8, progress_bar::Bool = true)

    N = length(ψ0)
    ϵ = sqrt(err_max) * 1e-2 # to not trigger an automatic rank increase
    m0 = initialize_m0(N, M, ψ0.data; ϵ = ϵ)
    p = (
        N = N,
        M = M,
        H = (get_data(H[1]), ((get_data(Hi[1]), Hi[2]) for Hi in H[2:end])...),
        L = [get_data(L) for L in c_ops],
        drive_params = params,
        Lm = similar(m0),
        temp_MM = similar(m0, M, M),
        temp_m = similar(m0),
        e_ops = [get_data(op) for op in e_ops],
        u_cache = similar(m0, M * N),
        err_max = err_max,
        err_min = err_min,
        ϵ = ϵ,
    )

    # Progress bar callback
    prog = ProgressBar(length(tlist), enable = progress_bar)
    prog_cb = FunctionCallingCallback((u, t, integrator) -> next!(prog); funcat = tlist)

    # Expectation value callback
    exp_cb = nothing
    saved_values = nothing
    if !isempty(e_ops)
        saved_values = SavedValues(Float64, Vector{ComplexF64})
        exp_cb = SavingCallback(expval_compute, saved_values; saveat=tlist)
    end

    # Rank increase callback
    if adapt_increase
        increase_cb = DiscreteCallback(
            (u, t, integrator) -> incM_condition(u, t, integrator),
            increase_rank!,
            save_positions = (false, false)
        )
    else
        increase_cb = nothing
    end

    # Rank decrease callback
    if adapt_decrease
        decrease_cb = DiscreteCallback(
            (u, t, integrator) -> decM_condition(u, t, integrator),
            decrease_rank!,
            save_positions = (false, false)
        )
    else
        decrease_cb = nothing
    end

    # Save rank callback
    saved_M = SavedValues(Float64, Int)
    M_cb = SavingCallback((u, t, integrator) -> integrator.p.M, saved_M; saveat=tlist)

    # Combine callbacks
    callbackset = CallbackSet(exp_cb, increase_cb, decrease_cb, prog_cb, M_cb)

    prob = ODEProblem{true}(dmdt_tuple!, vec(m0), (tlist[1], tlist[end]), p)
    sol = solve(prob, Tsit5();
        saveat=tlist, #!isempty(e_ops) ? tlist[end] : tlist,
        callback=callbackset,
        save_everystep=false,
        save_start=true,
        dense=false
    )

    expect = nothing
    if !isempty(e_ops)
        expect_raw = saved_values.saveval
        expect = Vector([getindex.(expect_raw, i) for i in axes(e_ops)[1]])
    end

    return TimeEvolutionLRSol(tlist, sol.u, expect, saved_M.saveval)
end

#= 
Post-processing functions
=#

function convert_lr_qobj(sol_lr)
    M = sol_lr.Ms[1]
    N = length(sol_lr.states[1]) ÷ M
    ρ_lr = map(sol_lr.states) do m
        m = reshape(m, N, :);
        ρ = m * m';
        ρ / tr(ρ)
    end
    return Qobj(ρ_lr)
end