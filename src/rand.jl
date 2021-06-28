"""
Sampling from DPP and k-DPP.

Methods:
--------
    rand(dpp::DeterminantalPointProcess)
    rand(dpp::DeterminantalPointProcess, N::Int)
    randmcmc(dpp::DeterminantalPointProcess, N::Int)
    rand(dpp::DeterminantalPointProcess, N::Int, k::Int)
    randmcmc(dpp::DeterminantalPointProcess, N::Int, k::Int)

References:
-----------
    [1] Kulesza, A., and B. Taskar. Determinantal point processes for machine
        learning. arXiv preprint arXiv:1207.6083, 2012.
    [2] Kang, B. Fast determinantal point process sampling with application to
        clustering. NIPS, 2013.
"""

"""
    _sample_mask!(rng, M, Λ, i)

Sample a mask for an elementary DPP.
"""
function _sample_mask!(
    rng::AbstractRNG, M::SharedMatrix{Bool}, Λ::SharedArray{<:Real}, i::Int
)
    for j in 1:length(Λ)
        M[j, i] = rand(rng) < (Λ[j] / (Λ[j] + 1))
    end
    return M
end

"""
    _sample_k_mask!(rng, M, Λ, E, k, i)

Sample a mask for an elementary k-DPP.
"""
function _sample_k_mask!(
    rng::AbstractRNG,
    M::SharedMatrix{Bool},
    Λ::SharedArray{<:Real},
    E::SharedMatrix{<:Real},
    k::Int,
    i::Int,
)
    j = length(Λ)
    remaining = k

    # iteratively sample a k-mask
    while remaining > 0
        # compute marginal of j given that we choose remaining values from 1:j
        if j == remaining
            marg = 1
        else
            marg = Λ[j] * E[remaining, j] / E[remaining + 1, j + 1]
        end

        # sample marginal
        if rand(rng) <= marg
            M[j, i] = true
            remaining -= 1
        end
        j -= 1
    end
    return M
end

"""
    _sample_from_elementary(rng, V, M, i)

Exact sampling from an elementary DPP. The algorithm based on [1].
"""
function _sample_from_elementary(
    rng::AbstractRNG, V::SharedMatrix{T}, M::SharedMatrix{Bool}, i::Int
) where {T<:Real}
    # select the elementary DPP
    V_mask = M[:, i]

    # edge case: empty sample
    if !any(V_mask)
        return Int[]
    end

    # select the kernel of the elementary DPP
    L = V[:, V_mask]

    Y = Int[]
    mask = ones(Bool, size(L, 2))
    prob = Array{T}(undef, size(L, 1))

    for i in 1:size(L, 2)
        # compute probabilities
        fill!(prob, 0)
        for c in 1:size(L, 2)
            !mask[c] && continue
            for r in 1:size(L, 1)
                prob[r] += L[r, c] .^ 2
            end
        end
        prob ./= sum(prob)

        # sample a point in the original space
        h = findfirst(rand(rng) .<= cumsum(prob))
        push!(Y, h)

        # select and mask-out an element
        j = get_first_nz_idx(L[h, :], mask)
        mask[j] = false

        if any(mask)
            # Subtract scaled Lj from other columns so that their
            # projections on e_s[i] turns into 0. This operation
            # preserves the rank of L_{-j}.
            for c in 1:size(L, 2)
                !mask[c] && continue
                for r in 1:size(L, 1)
                    L[r, c] -= L[r, j] * L[h, c] / L[h, j]
                end
            end

            # Gram-Schmidt orthogonalization
            L[:, mask] = Matrix(qr(L[:, mask]).Q)
        end
    end

    return sort(Y)
end

Random.rand(pp::PointProcess, n::Int) = rand(GLOBAL_RNG, pp, n)
Random.rand(pp::PointProcess) = rand(GLOBAL_RNG, pp)
Random.rand(rng::AbstractRNG, pp::PointProcess) = first(rand(rng, pp, 1))

"""
    rand([rng::AbstractRNG], dpp::DeterminantalPointProcess)::Vector{Int}
    rand([rng::AbstractRNG], dpp::DeterminantalPointProcess, n::Int)::Vector{Vector{Int}}

Exact sampling from a DPP [1].
Returns a vector of indices with respect to the `L` matrix passed to the `DeterminantalPointProcess`.
The length of each vector can vary from 0 to the `size(L,1)`
"""
function Random.rand(
    rng::AbstractRNG, dpp::DeterminantalPointProcess{T}, N::Int
) where {T<:Real}
    Λ = SharedArray{T}(dpp.Lfact.values)
    V = SharedMatrix{T}(dpp.Lfact.vectors)
    M = SharedMatrix{Bool}(zeros(Bool, dpp.size, N))

    # step I: sample masks for elementary DPPs
    pmap(
        (i, seed) -> _sample_mask!(MersenneTwister(seed), M, Λ, i),
        1:N,
        abs.(rand(rng, Int, N)),
    )

    # step II: iteratively sample from a mixture of elementary DPPs
    return pmap(
        (i, seed) -> _sample_from_elementary(MersenneTwister(seed), V, M, i),
        1:N,
        abs.(rand(rng, Int, N)),
    )
end

"""
    rand([rng::AbstractRNG], kdpp::kDeterminantalPointProcess)::Vector{Int}
    rand([rng::AbstractRNG], kdpp::kDeterminantalPointProcess, n::Int)::Vector{Vector{Int}}

Exact sampling from a k-DPP [1].
Returns a vector of indices with respect to the `L` matrix passed to the `kDeterminantalPointProcess`
Each vector is of size `k`.
"""
function Random.rand(rng::AbstractRNG, kdpp::kDPP{T}, N::Int) where {T<:Real}
    dpp = kdpp.dpp
    Λ = SharedArray{T}(dpp.Lfact.values)
    V = SharedMatrix{T}(dpp.Lfact.vectors)
    M = SharedMatrix{Bool}(zeros(Bool, dpp.size, N))

    # compute elementary symmetric polynomials
    E = SharedMatrix{T}(elem_symm_poly(dpp.Lfact.values, kdpp.k))

    # step I: sample masks for elementary DPPs
    pmap(
        (i, seed) -> _sample_k_mask!(MersenneTwister(seed), M, Λ, E, kdpp.k, i),
        1:N,
        abs.(rand(rng, Int, N)),
    )

    # step II: iteratively sample from a mixture of elementary DPPs
    return pmap(
        (i, seed) -> _sample_from_elementary(MersenneTwister(seed), V, M, i),
        1:N,
        abs.(rand(rng, Int, N)),
    )
end

"""
    Perform one MCMC accept-reject transition for DPP.
"""
function _do_mcmc_step!(rng::AbstractRNG, dpp::DeterminantalPointProcess, state::MCMCState)
    # propose an element to swap
    u = rand(rng, 1:(dpp.size))
    insert = !state[1][u]

    # attempt to make a transition
    if insert
        p = _comp_accept_prob(dpp, state, u, insert)
        rand(rng) < p && _update_mcmc_state!(dpp, state, u, insert)
    else  # delete
        new_state = _update_mcmc_state(dpp, state, u, insert)
        p = _comp_accept_prob(dpp, new_state, u, insert)
        if rand(rng) < p
            state[1] .= new_state[1]
            state[2] .= new_state[2]
        end
    end
end

"""
    Perform one MCMC accept-reject transition for k-DPP.
"""
function _do_mcmc_k_step!(rng::AbstractRNG, kdpp::kDPP, state::MCMCState)
    z, L_z_inv = state

    # propose the elements to swap
    u, v = rand(rng, findall(z)), rand(rng, findall(z .== true))

    # copy the state and delete the u element
    new_state = _update_mcmc_state(kdpp.dpp, state, u, false)

    # attempt to make a transition
    p = _comp_accept_prob(kdpp.dpp, new_state, u, v)
    if rand(rng) < p
        # insert the v element into the new state
        _update_mcmc_state!(kdpp.dpp, new_state, v, true)
        state[1] .= new_state[1]
        state[2] .= new_state[2]
    end
end

"""
    Compute accept probability to insert / delete u from the state.
"""
function _comp_accept_prob(
    dpp::DeterminantalPointProcess, state::MCMCState, u::Int, insert::Bool
)
    z, L_z_inv = state

    d_u = dpp.L[u, u]
    if any(z)
        b_u = dpp.L[z, u]
        d_u -= dot(b_u, L_z_inv[z, z] * b_u)
    end

    return insert ? min(1.0, d_u) : min(1.0, 1.0 / d_u)
end

"""
    Compute accept probability to swap u and v.
"""
function _comp_accept_prob(dpp::DeterminantalPointProcess, state::MCMCState, u::Int, v::Int)
    z, L_z_inv = state

    d_u, d_v = dpp.L[u, u], dpp.L[v, v]
    if any(z)
        b_u, b_v = dpp.L[z, u], dpp.L[z, v]
        d_u -= dot(b_u, L_z_inv[z, z] * b_u)
        d_v -= dot(b_v, L_z_inv[z, z] * b_v)
    end

    return min(1.0, d_v / d_u)
end

"""
    Update the state after u is inserted / deleted.
"""
function _update_mcmc_state!(
    dpp::DeterminantalPointProcess, state::MCMCState, u::Int, insert::Bool
)
    z, L_z_inv = state

    if insert
        d_u = dpp.L[u, u]
        if any(z)
            b_u = dpp.L[z, u]
            x_u = L_z_inv[z, z] * b_u
            d_u -= dot(b_u, x_u)

            L_z_inv[z, z] += (x_u * x_u') / d_u
            L_z_inv[z, u] = L_z_inv[u, z] = -x_u / d_u
        end

        L_z_inv[u, u] = 1.0 / d_u
        z[u] = true
    else  # delete
        z[u] = false

        e = L_z_inv[z, u]
        f = L_z_inv[u, u]

        L_z_inv[z, z] -= (e * e') / f
    end
end

"""
    Update the state after u is inserted / deleted.
"""
function _update_mcmc_state(
    dpp::DeterminantalPointProcess, state::MCMCState, u::Int, insert::Bool
)
    new_state = deepcopy(state)
    _update_mcmc_state!(dpp, new_state, u, insert)
    return new_state
end


"""
    MCMC sampling from a DPP [2].

TODO: Add support for running MCMC in parallel, similar as rand.
          Make sure parallelization produces unbiased and consistent samples.
"""
function randmcmc(
    rng::AbstractRNG,
    dpp::DeterminantalPointProcess{T},
    N::Int;
    init_state=nothing,
    return_final_state::Bool=false,
    mix_eps::Real=1e-1,
    mixing_steps::Int=ceil(Int, dpp.size * log(dpp.size / mix_eps)),
    steps_between_samples::Int=mixing_steps,
) where {T}
    # initialize the Markov chain
    state = init_state
    if state === nothing
        L_z_inv = Array{T}(undef, size(dpp.L))
        z = bitrand(rng, dpp.size)  # TODO: improve initialization (?)
        if any(z)
            L_z_inv[z, z] = pinv(dpp.L[z, z])
        end
        state = (z, L_z_inv)
    end

    # sanity check
    @assert state isa MCMCState

    # mix the Markov chain
    for t in 1:mixing_steps
        _do_mcmc_step!(rng, dpp, state)
    end

    Y = []
    for i in 1:N
        push!(Y, findall(state[1]))
        for t in 1:steps_between_samples
            _do_mcmc_step!(rng, dpp, state)
        end
    end

    return return_final_state ? (Y, state) : Y
end

function randmcmc(
    dpp::DeterminantalPointProcess,
    N::Int;
    init_state=nothing,
    return_final_state::Bool=false,
    mix_eps::Real=1e-1,
    mixing_steps::Int=ceil(Int, dpp.size * log(dpp.size / mix_eps)),
    steps_between_samples::Int=mixing_steps,
)
    return randmcmc(
        GLOBAL_RNG,
        dpp,
        N;
        init_state=init_state,
        return_final_state=return_final_state,
        mix_eps=mix_eps,
        mixing_steps=mixing_steps,
        steps_between_samples=steps_between_samples,
    )
end

"""
    randmcmc([rng::AbstractRNG], kdpp::kDPP, N::Int; kwargs...)

MCMC sampling from a k-DPP [2].

## Arguments

- `rng` : Random number generator (by default Random.GLOBAL_RNG is used)
- `kdpp` : k-DeterminantalPointProcess
- `N` : Number of samples

## Keyword Arguments

- `init_state`
- `return_final_state`
- `mix_eps`
- `mixing_steps`
- `steps_between_samples`

TODO: 
- Add support for running MCMC in parallel, similar as rand.
- Make sure parallelization produces unbiased and consistent samples.
"""
function randmcmc(
    rng::AbstractRNG,
    kdpp::kDPP{T},
    N::Int;
    init_state=nothing,
    return_final_state::Bool=false,
    mix_eps::Real=1e-1,
    mixing_steps::Int=ceil(Int, kdpp.k * log(kdpp.k / mix_eps)),
    steps_between_samples::Int=mixing_steps,
) where {T}

    # initialize the Markov chain
    state = init_state
    if state === nothing
        L_z_inv = Array{T}(undef, size(kdpp.dpp.L))
        z = falses(kdpp.dpp.size)  # TODO: improve initialization (?)
        z[1:(kdpp.k)] .= true
        if any(z)
            L_z_inv[z, z] = pinv(kdpp.dpp.L[z, z])
        end
        state = (z, L_z_inv)
    end

    # sanity check
    @assert typeof(state) == MCMCState
    @assert sum(state[1]) == kdpp.k

    # mix the Markov chain
    for t in 1:mixing_steps
        _do_mcmc_k_step!(rng, kdpp, state)
    end

    Y = []
    for i in 1:N
        push!(Y, findall(state[1]))
        for t in 1:steps_between_samples
            _do_mcmc_k_step!(rng, kdpp, state)
        end
    end

    return return_final_state ? (Y, state) : Y
end

function randmcmc(
    kdpp::kDPP,
    N::Int;
    init_state=nothing,
    return_final_state::Bool=false,
    mix_eps::Real=1e-1,
    mixing_steps::Int=ceil(Int, kdpp.k * log(kdpp.k / mix_eps)),
    steps_between_samples::Int=mixing_steps,
)
    return randmcmc(
        GLOBAL_RNG,
        kdpp,
        N;
        init_state=init_state,
        return_final_state=return_final_state,
        mix_eps=mix_eps,
        mixing_steps=mixing_steps,
        steps_between_samples=steps_between_samples,
    )
end
