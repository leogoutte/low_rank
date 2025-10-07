using OrdinaryDiffEq, LinearAlgebra, SparseArrays
using SpecialFunctions
using QuantumToolbox

#=
Parameters
=#

εd_scale = 2.; # scale of the drive amplitude to match the literature
tp = (Nφ = 300, Ec = 2π * 0.315, Ej = 51 * 2π * 0.315, g = 2π * 0.150, ωa = 2π * 7.2, ωf = 2π * 7.21,
J = 2π * 0.030, εd = εd_scale * 2π * 0.075, ωd = 45.1158, κ = 2π * 0.030, γ = 2π * 8 * 1e-6, η = 0.6);

#=
Basic functions
=# 

""" 
Define the sigmoid function 
"""
function sigmoid(x::Float64, center::Float64, width::Float64)
    return 1 / (1 + exp(-(x - center) / width))
end

""" 
Define the function to generate a smoothed flat pulse 
"""
function smoothed_flat_pulse(t, t0, τ, σ)
    pulse = sigmoid.(t, t0, σ) .* (1 .- sigmoid.(t, t0 + τ, σ))
    return pulse
end;

function multi_step_function_smooth(heights::AbstractVector, widths::AbstractVector; t0::Real=2., k::Real=0.05)
    # Returns a function f(t) that is a smoothed step function using sigmoids
    # The function begins and ends with a zero height, each of width t0
    h = vcat(0.0, heights, 0.0)
    w = vcat(t0, widths, t0)
    edges = cumsum(w)
    function f(p, t)
        val = h[1]
        for i in eachindex(h)[2:end]
            # Smooth transition at each edge using a sigmoid
            val += (h[i] - h[i-1]) * (1 / (1 + exp(-(t - edges[i-1])/k)))
        end
        return val
    end
    return f
end

"""
Functions that will be required for the time-dependence of the rotated frame and drives
"""
signal_sin(p, t) = smoothed_flat_pulse(t, p.t0, p.τ, p.σ) * sin(p.ωd * t);
exp_2p(p, t) = exp(2*im * p.ωd * t);
exp_2m(p, t) = exp(-2*im * p.ωd * t);
signal_exp_2p(p, t) = smoothed_flat_pulse(t, p.t0, p.τ, p.σ) * exp(2*im * p.ωd * t);
signal_exp_2m(p, t) = smoothed_flat_pulse(t, p.t0, p.τ, p.σ) * exp(-2*im * p.ωd * t);
signal(p, t) = smoothed_flat_pulse(t, p.t0, p.τ, p.σ);
flat_signal(p, t) = 1.0;

#=
Generating the Hamiltonian
=#

""" 
make the transmon Hamiltonian in the φ basis
"""
function get_Hφ_trans(Nφ::Int64, Ec::Float64, Ej::Float64)
    φ_list = collect(range(-π, π, Nφ + 1)) # make sure to include the 0 value  -> question: why?
    pop!(φ_list) # remove the last value
    Δφ = φ_list[2] - φ_list[1]

    # matrix of derivative w.r.t φ
    ∂φ = spdiagm(1 => ones(Nφ-1), -1 => -ones(Nφ-1)) ./ 2Δφ 
    ∂φ[1, end] = -1 / 2Δφ
    ∂φ[end, 1] = 1 / 2Δφ
    nφ = -im * ∂φ
    # hermiticity: checked

    # matrix of derivative squared w.r.t φ
    ∂2φ = spdiagm(-1 => ones(Nφ-1), 1 => ones(Nφ-1), 0 => -2ones(Nφ)) ./ Δφ^2 
    ∂2φ[1, end] = 1 / Δφ^2
    ∂2φ[end, 1] = 1 / Δφ^2

    nφ2 = -∂2φ

    # transmon Hamiltonian in the φ basis
    Hφ = 4Ec * nφ2 - Ej * spdiagm(0 => cos.(φ_list))
    return Hφ, nφ
end

"""
diagonalize the Hamiltonian from its expression in the φ basis and truncate it to its lowest Ntrans levels 
"""
function get_diagHam_from_Hφ(Ntrans::Int64, Hφ::AbstractMatrix, nφ::AbstractMatrix, Ej::Float64, shift::Bool=false)  
    λt, vt, Ut = eigsolve(Hφ, eigvals=Ntrans, sigma=-Ej) 
    nt = Ut' * nφ * Ut
    nt = (nt + nt') / 2 |> to_sparse |> dropzeros # enforce Hermicity
    Ht = spdiagm(0 => λt)
    b = spdiagm(1 => sqrt.(1:Ntrans-1))    
    return Ht - λt[1] * I(Ntrans) * shift, b, nt, λt .- λt[1] * shift # shift the Hamiltonian energies by the minimum
end

"""
return the full transmon Hamiltonian
"""
function get_Htrans(Nφ, Nt, Ec, Ej)
    Hφ, nφ = get_Hφ_trans(Nφ, Ec, Ej);
    Ht, bt_small, nt_small, λt = get_diagHam_from_Hφ(Nt, Hφ, nφ, Ej, true);

    # define nt_p
    ntp = triu(nt_small) - triu(nt_small, 2); # applied approximation of only first off-diagonal

    return Ht, ntp, bt_small, λt
end

"""
remove Harmonic ladder and smallest energy -- for transformations into rotated frame
"""
function shift_Htrans(Ht, ωd)
    Nt = shape(Ht)[1]
    return Ht - ωd * spdiagm(0 => 0:(Nt-1))
end

"""
diagonalize the resonator-purcell Hamiltonian 
"""
function get_normal_modes(ωa, ωf, J, N1, N2, Nt)
    Hbdg = diagm(0 => [ωa, ωf, -ωa, -ωf])
    Hbdg += J * kron(sigmaz().data, sigmax().data)
    Hbdg += J * kron(im .* sigmay().data, sigmax().data)

    ωs, U = eigen(Hbdg); # U = (v1, ..., v4)

    # normal mode frequencies
    ω1, ω2 = real(ωs[4]), real(ωs[3]); # corresponding to c1, c2 (resp.)

    # properly normalized transformation matrix to go from (a, f, a', f') to (c1, c2, c2', c1')
    U_fix = zeros(ComplexF64, 4, 4)

    desired_commutators = [-1, -1, 1, 1] # c1', c2', c2, c1

    for i in 1:4
        v_i = U[:, i]
        α, β, γ, δ = v_i
        commutator = abs2(α) + abs2(β) - abs2(γ) - abs2(δ)
        U_fix[:, i] = v_i ./ sqrt(commutator / desired_commutators[i])
    end

    # invert system of equations to go from (c1, c2, c2', c1') to (a, f, a', f')
    U_fix_bis = inv(transpose(U_fix))

    γ1 = U_fix_bis[1, 4]' - U_fix_bis[1, 1]
    γ2 = U_fix_bis[1, 3]' - U_fix_bis[1, 2]
    δ1 = U_fix_bis[2, 4]' - U_fix_bis[2, 1]
    δ2 = U_fix_bis[2, 3]' - U_fix_bis[2, 2]
    γδs = [γ1, γ2, δ1, δ2]

    c1 = destroy(N1) ⊗ qeye(N2) ⊗ qeye(Nt);
    c2 = qeye(N1) ⊗ destroy(N2) ⊗ qeye(Nt);

    # original operators (useful for dissipators)
    a = U_fix_bis[1, 1] * c1' + U_fix_bis[1, 2] * c2' + U_fix_bis[1, 3] * c2 + U_fix_bis[1, 4] * c1
    f = U_fix_bis[2, 1] * c1' + U_fix_bis[2, 2] * c2' + U_fix_bis[2, 3] * c2 + U_fix_bis[2, 4] * c1;

    return [c1, c2], [a, f], [ω1, ω2], γδs
end

"""
make the total Hamiltonian
"""
function get_H_tuples(Nt::Int, N1::Int, N2::Int, Nφ::Int, Ec::Real, Ej::Real, g::Real, ωa::Real, ωf::Real, J::Real, ωd::Real, εd::Real, κ::Real, γ::Real, signal::Function; RWA::Bool=false)
    # resonator-purcell Hamiltonian
    cs, as, ωs, γδs = get_normal_modes(ωa, ωf, J, N1, N2, Nt);
    c1, c2 = cs;
    a, f = as;
    ω1, ω2 = ωs;
    γ1, γ2, δ1, δ2 = γδs;

    # transmon Hamiltonian
    Ht, nt_p, bt_small, _ = get_Htrans(Nφ, Nt, Ec, Ej);
    Htrans_shifted = shift_Htrans(Ht, ωd);
    ntp = qeye(N1) ⊗ qeye(N2) ⊗ Qobj(nt_p);
    Htrans = qeye(N1) ⊗ qeye(N2) ⊗ Qobj(Htrans_shifted);

    # c_ops
    b = qeye(N1) ⊗ qeye(N2) ⊗ Qobj(bt_small);
    c_ops = [sqrt(κ) * f, sqrt(γ)* b];

    # pulses -- split into real and imaginary parts
    real_signal(p, t) = real(signal(p, t));
    imag_signal(p, t) = imag(signal(p, t));
    real_signal_exp_2p(p, t) = real_signal(p, t) * exp(2*im * p.ωd * t);
    real_signal_exp_2m(p, t) = real_signal(p, t) * exp(-2*im * p.ωd * t);
    imag_signal_exp_2p(p, t) = imag_signal(p, t) * exp(2*im * p.ωd * t);
    imag_signal_exp_2m(p, t) = imag_signal(p, t) * exp(-2*im * p.ωd * t);

    # total Hamiltonian
    H0 = Htrans + (ω1 - ωd) * c1' * c1 + (ω2 - ωd) * c2' * c2 + im * g * (ntp * (γ1 * c1' + γ2 * c2') - ntp' * (γ1 * c1 + γ2 * c2))
    Hints = ((im * g * ntp' * (γ1 * c1' + γ2 * c2'), exp_2p), (-im * g * ntp * (γ1 * c1 + γ2 * c2), exp_2m),)
    Hdrives_real = ((-εd/2 * (δ1 * (c1 + c1') + δ2 * (c2 + c2')), real_signal), (εd/2 * (δ1 * c1' + δ2 * c2'), real_signal_exp_2p), (εd/2 * (δ1 * c1 + δ2 * c2), real_signal_exp_2m))
    Hdrives_imag = ((im * εd/2 * (δ1 * (c1' - c1) + δ2 * (c2' - c2)), imag_signal), (im * εd/2 * (δ1 * c1' + δ2 * c2'), imag_signal_exp_2p), (-im * εd/2 * (δ1 * c1 + δ2 * c2), imag_signal_exp_2m))

    if RWA
        return (H0, (-εd/2 * (δ1 * (c1 + c1') + δ2 * (c2 + c2')), real_signal), (im * εd/2 * (δ1 * (c1' - c1) + δ2 * (c2' - c2)), imag_signal)), c_ops, a, f
    end

    return (H0, Hints..., Hdrives_real..., Hdrives_imag...), c_ops, a, f
end

#=
Signal analysis functions
=#
"""
cumulative trapezoidal integration
"""
function cumtrapz(X::AbstractVector, Y::AbstractVector)
  # Check matching vector length
  @assert length(X) == length(Y)
  # Initialize Output
  out = similar(Y)
  out[1] = 0
  # Iterate over arrays
  for i in eachindex(X)[2:end]
    out[i] = out[i-1] + 0.5*(X[i] - X[i-1])*(Y[i] + Y[i-1])
  end
  # Return output
  return out
end

"""
signal-to-noise ratio and fidelity
"""
function signal_to_noise(βg, βe, tlist, η, κ, γ; choice = 1)
    δt = tlist[2] - tlist[1]
    δβ = βg .- βe;

    if choice == 1 # optimal control
        SNR = sqrt.(2 * η * κ * cumtrapz(tlist |> collect, abs2.(δβ)));
        # SNR = sqrt.(δt * cumsum(abs2.(δβ))); # TODO: replace with cumtrapz
    elseif choice == 2 # model-based
        SNR = abs2.(cumsum(abs2.(δβ)));
        SNR ./= cumsum(abs2.(δβ));
        SNR *= 2 * η * κ * δt;
    end
    return SNR
end

function assignment_error(βg, βe, tlist, η, κ, γ; choice = 1)
    if choice == 1 # optimal control
        SNR = signal_to_noise(βg, βe, tlist, η, κ, γ; choice = 1);
        ϵ_sep = 1/2 * erfc.(SNR / 2);
        ϵ_relaxation = tlist .* γ / 2;
        err = ϵ_sep .+ ϵ_relaxation;
    elseif choice == 2 # model-based
        SNR = signal_to_noise(βg, βe, tlist, η, κ, γ; choice = 2);
        ϵ_sep = 1/2 * erfc.(sqrt.(SNR) / 2);
        ϵ_relaxation = tlist .* γ / 2;
        err = ϵ_sep + ϵ_relaxation;
    end
    return err
end;

"""
Forbidden states in the transmon
"""
function forbidden_states_transmon(ks_g, ks_e, tlist)
    tf = tlist[end]
    @assert length(ks_g) == length(tlist)
    @assert length(ks_e) == length(tlist)
    ret = cumtrapz(tlist |> collect, ks_g .+ ks_e)[end]
    return ret / tf |> real
end

function forbidden_states_undriven(ks_g, ks_e, tlist)
    tf = tlist[end]
    @assert length(ks_g) == length(tlist)
    @assert length(ks_e) == length(tlist)
    ret = cumtrapz(tlist |> collect, ks_g .+ ks_e)[end]
    return ret / tf |> real
end