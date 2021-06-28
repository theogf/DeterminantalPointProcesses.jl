# abstract types
abstract type PointProcess end;

"""
    DeterminantalPointProcess(L::AbstractMatrix)
    DeterminantalPointProcess(L::Eigen)

Given a symmetric matrix `L`, creates a `DeterminantalPointProcess` (DPP). One can also pass the eigen object directly

---

    DeterminantalPointProcess(kernel::Kernel, X::AbstractVector)
    DeterminantalPointProcess(kernel::Kernel, X::AbstractMatrix; obsdim=1)

Similar to the basic constructor, will first build the kernel matrix with `kernel` on observations `X`.
If your input is a `AbstractMatrix` you can pass the `obsdim` argument to indicate if rows or columns represent the samples (see docs of [KernelFunctions](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/))
"""
struct DeterminantalPointProcess{T,TL<:AbstractMatrix{T},Tfact} <: PointProcess
    L::TL
    Lfact::Tfact
    size::Int
end

function DeterminantalPointProcess(L::AbstractMatrix)
    issymmetric(L) || error("Given matrix is not symmetric")
    Lfact = eigen(L)
    return DeterminantalPointProcess(L, Lfact, length(Lfact.values))
end

function DeterminantalPointProcess(Lfact::Eigen)
    L = Symmetric((Lfact.vectors .* Lfact.values') * Lfact.vectors')
    return new{eltype(L),typeof(L),typeof(Lfact)}(L, Lfact, length(Lfact.values))
end

"""
    kDeterminantalPointProcess(k::Int, dpp::DeterminantalPointProcess)

Create a k-DPP where the size of the subsets is fixed to k.
You can also create a k-DPP by calling `dpp(k)`
"""
struct kDeterminantalPointProcess{T,Tdpp<:DeterminantalPointProcess{T}} <: PointProcess
    k::Int
    dpp::Tdpp
end

(dpp::DeterminantalPointProcess)(k::Int) = kDPP(k, dpp)
# struct KroneckerDeterminantalPointProcess <: PointProcess
# TODO
# end

# aliases
const DPP = DeterminantalPointProcess
const kDPP = kDeterminantalPointProcess
# const KDPP = KroneckerDeterminantalPointProcess
const MCMCState = Tuple{BitArray{1},Array{Float64,2}}
