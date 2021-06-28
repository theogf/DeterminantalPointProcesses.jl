"""Compute the log probability of a sample `z` under the given DPP.
"""
function logpmf(dpp::DeterminantalPointProcess, z::AbstractVector{<:Int})
    L_z_eigvals = eigvals(dpp.L[z, z])
    return sum(log.(L_z_eigvals)) - sum(log.(dpp.Lfact.values .+ 1))
end


"""Compute the log probability of a sample `z` under the given k-DPP.
"""
function logpmf(kdpp::kDeterminantalPointProcess, z::AbstractArray{<:Int})
    L_z_eigvals = eigvals(kdpp.dpp.L[z, z])
    return sum(log.(L_z_eigvals)) .-
           log(elem_symm_poly(kdpp.dpp.Lfact.values, kdpp.k)[end, end])
end

"""Compute the probability of a sample `z` under the given DPP.
"""
function pmf(dpp::DeterminantalPointProcess, z::AbstractArray{<:Int})
    return exp(logpmf(dpp, z))
end

"""Compute the probability of a sample `z` under the given DPP.
"""
function pmf(kdpp::kDeterminantalPointProcess, z::AbstractArray{<:Int})
    return exp(logpmf(kdpp, z))
end
