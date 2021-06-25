# abstract types
abstract type PointProcess end;

# specific types
struct DeterminantalPointProcess{TL,Tfact} <: PointProcess
    L::TL
    Lfact::Tfact
    size::Int

    function DeterminantalPointProcess(L::Symmetric)
        Lfact = eigen(L)
        return new{typeof(L),typeof(Lfact)}(L, Lfact, length(Lfact.values))
    end

    function DeterminantalPointProcess(Lfact::Eigen)
        L = Symmetric((Lfact.vectors .* Lfact.values') * Lfact.vectors')
        return new{typeof(L),typeof(Lfact)}(L, Lfact, length(Lfact.values))
    end
end

struct kDeterminantalPointProcess{T<:DPP} <: PointProcess
    k::Int
    dpp::T
end

(dpp::DeterminantalPointProcess)(k::Int) = kDPP(k, dpp)
struct KroneckerDeterminantalPointProcess <: PointProcess
    # TODO
end

# aliases
const DPP = DeterminantalPointProcess
const kDPP = kDetermintanlPointProcess
const KDPP = KroneckerDeterminantalPointProcess
const MCMCState = Tuple{BitArray{1},Array{Float64,2}}
