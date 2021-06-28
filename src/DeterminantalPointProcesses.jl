######################################################################
# DeterminantalPointProcesses.jl
# Determinantal Point Processes in Julia
# http://github.com/theogf/DeterminantalPointProcesses.jl
# MIT Licensed
######################################################################

__precompile__(true)

module DeterminantalPointProcesses

using Distributed
using KernelFunctions
using LinearAlgebra
using Random: Random, rand, bitrand, AbstractRNG, MersenneTwister, GLOBAL_RNG
using Requires
using SharedArrays
import Base: rand

export
    # point process types and aliases
    DeterminantalPointProcess,
    DPP,
    kDeterminantalPointProcess,
    kDPP,

    # mehtods
    logpmf,             # log probability mass function
    pmf,                # probability mass function
    rand,               # generate samples
    randmcmc            # generate samples using MCMC

### source files
function __init__()
    @require KernelFunctions="ec8451be-7e33-11e9-00cf-bbf324bd1392" include("kernelcompat.jl")
end


# types
include("types.jl")

# methods
include("fit.jl")
include("prob.jl")
include("rand.jl")

# utilities
include("utils.jl")

end # module
