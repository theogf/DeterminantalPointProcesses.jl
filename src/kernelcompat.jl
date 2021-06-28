using .KernelFunctions

function DeterminantalPointProcess(kernel::Kernel, X::AbstractVector)
    return DeterminantalPointProcess(kernelmatrix(kernel, X))
end

function DeterminantalPointProcess(kernel::Kernel, X::AbstractMatrix; obsdim=1)
    return DeterminantalPointProcess(kernel, KernelFunctions.vec_of_vecs(X; obsdim=obsdim))
end