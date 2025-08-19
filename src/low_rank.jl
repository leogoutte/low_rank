using OrdinaryDiffEq, DiffEqCallbacks
using LinearAlgebra, SparseArrays
using QuantumToolbox

struct TimeEvolutionLRSol{
    TT,
    TS,
    TE,
}
    times::TT
    states::TS
    expect::TE
end

function initialize_m0(N::Int, M::Int, initial_state::AbstractVector; ϵ::Real = 1e-10)
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

#=
Low-rank evolution
=#

function dmdt!(dm, m, p, t)
    # normalize -- divide by sqrt(tr(m' * m))
    normalize!(m)

    # unwrap parameters
    temp_MM = p.temp_MM
    temp_m = p.temp_m
    L = p.L
    drive_params = p.drive_params

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
end

function dmdt_tuple!(dm, m, p, t)
    # normalize -- divide by sqrt(tr(m' * m))
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
end

# TODO: adapt everything to Qobj and QobjEvo
# TODO: reduce allocations and memory usage
"""
    solve_lr(H::Tuple, ψ0::QuantumObject, M::Int, tlist::AbstractVector, c_ops::AbstractVector;
             e_ops::AbstractVector = [], params::NamedTuple = NamedTuple(), progress_bar::Bool = true, ϵ::Real = 1e-10)
Solves the time evolution of a quantum system using low-rank methods.
- `H`: Hamiltonian of the system, given as a tuple where the first element is the time-independent part and the rest are time-dependent parts.
- `ψ0`: Initial state of the system, represented as a `QuantumObject`.
- `M`: Number of low-rank states to use in the evolution.
- `tlist`: Vector of time points at which to evaluate the solution.
- `c_ops`: List of collapse operators for the system.
- `e_ops`: List of operators for which to compute expectation values (optional).
- `params`: Named tuple of parameters for the time-dependent Hamiltonian (optional).
- `progress_bar`: Boolean flag to enable or disable the progress bar (default is `true`).
- `ϵ`: Small value to ensure numerical stability in the initialization of the low-rank states (default is `1e-10`).
Returns a `TimeEvolutionLRSol` object containing the time points, low-rank states, and expectation values.
"""
function solve_lr(
    H::Union{QuantumObject, AbstractMatrix}, ψ0::QuantumObject, M::Int, tlist::AbstractVector, c_ops::AbstractVector;
    e_ops::AbstractVector = [], params::NamedTuple = NamedTuple(), progress_bar::Bool = true, ϵ::Real = 1e-10,
)

    N = length(ψ0)
    m0 = initialize_m0(N, M, ψ0.data; ϵ = ϵ)
    p = (
        N = N,
        M = M,
        H = get_data(H),
        L = [get_data(L) for L in c_ops],
        drive_params = params,
        Lm = similar(m0),
        temp_MM = similar(m0, M, M),
        temp_m = similar(m0),
    )

    # Progress bar callback
    prog = ProgressBar(length(tlist), enable = progress_bar)
    prog_cb = FunctionCallingCallback((u, t, integrator) -> next!(prog); funcat = tlist)

    # Expectation value callback
    cb = nothing
    saved_values = nothing
    if !isempty(e_ops)
        saved_values = SavedValues(Float64, Vector{ComplexF64})
        cb = SavingCallback(
            (m, t, integrator) -> [tr(m' * op.data * m) for op in e_ops],
            saved_values; saveat=tlist
        )
    end

    # Combine callbacks
    callbackset = isnothing(cb) ? CallbackSet(prog_cb) : CallbackSet(cb, nothing, prog_cb)

    prob = ODEProblem{true}(dmdt!, m0, (tlist[1], tlist[end]), p)
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

    return TimeEvolutionLRSol(tlist, sol.u, expect)
end

function solve_lr(
    H::Tuple, ψ0::QuantumObject, M::Int, tlist::AbstractVector, c_ops::AbstractVector;
    e_ops::AbstractVector = [], params::NamedTuple = NamedTuple(), progress_bar::Bool = true, ϵ::Real = 1e-10,
)
    N = length(ψ0)
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
    )

    # Progress bar callback
    prog = ProgressBar(length(tlist), enable = progress_bar)
    prog_cb = FunctionCallingCallback((u, t, integrator) -> next!(prog); funcat = tlist)

    # Expectation value callback
    cb = nothing
    saved_values = nothing
    if !isempty(e_ops)
        saved_values = SavedValues(Float64, Vector{ComplexF64})
        cb = SavingCallback(
            (m, t, integrator) -> [tr(m' * op.data * m) for op in e_ops],
            saved_values; saveat=tlist
        )
    end

    # Combine callbacks
    callbackset = CallbackSet(cb, prog_cb)

    prob = ODEProblem{true}(dmdt_tuple!, m0, (tlist[1], tlist[end]), p)
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

    return TimeEvolutionLRSol(tlist, sol.u, expect)
end

function convert_lr_qobj(sol::TimeEvolutionLRSol, dims::Tuple)
    ms = sol.states
    ρs = []
    for m in ms
        push!(ρs, Qobj(m * m', dims = dims))
    end
    return ρs
end

#=
Plotting and visualization functions
=#

function plot_wigner_contourf(ρ, xvec, yvec; levels = 10, location=nothing, ax_kwargs = nothing)
    W = wigner(ρ, xvec, yvec)
    max_W = maximum(W)

    if isnothing(location)
        fig = Figure(size = (400, 400))
        location = fig[1, 1]
    end

    ax = Axis(location)
    ctf = contourf!(ax, W', colormap = Reverse(:RdBu), levels = range(-max_W, max_W, length=levels))
    hlines!(ax, length(yvec)/2, color = :black, linewidth = 1, linestyle = :dash, alpha = 0.5)
    vlines!(ax, length(xvec)/2, color = :black, linewidth = 1, linestyle = :dash, alpha = 0.5)
    xlims!(ax, 1, length(xvec))
    ylims!(ax, 1, length(yvec))

    return ax, ctf

end

#=
Obsolete functions
=#

function expect_lr(op, sol, tlist; states = false, ret_times = false)
    if states
        ms = sol # sol is already the states at each point
    else
        ms = [sol(t) for t in tlist]
    end

    expval = []
    for m in ms
        push!(expval, (tr(m' * op * m)))
    end

    if ret_times
        return expval, tlist
    end

    return expval
end

function _pinv_smooth!(
    A::AbstractMatrix{T},
    T1::AbstractMatrix{T},
    T2::AbstractMatrix{T};
    atol::Real = 0.0,
    rtol::Real = (eps(real(float(oneunit(T)))) * min(size(A)...)) * iszero(atol),
) where {T}
    # if isdiag(A)
    #     idxA = diagind(A)
    #     diagA = view(A, idxA)
    #     maxabsA = maximum(abs, diagA)
    #     λ = max(rtol * maxabsA, atol)
    #     return Matrix(Diagonal(pinv.(diagA) .* 1 ./ (1 .+ (λ ./ real(diagA)) .^ 6)))
    # end

    SVD = svd(A)
    λ = max(rtol * maximum(SVD.S), atol)
    SVD.S .= pinv.(SVD.S) .* 1 ./ (1 .+ (λ ./ SVD.S) .^ 6)
    mul!(T2, Diagonal(SVD.S), SVD.U')
    return mul!(T1, SVD.Vt', T2)
end;