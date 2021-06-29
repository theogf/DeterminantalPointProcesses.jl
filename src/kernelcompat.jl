using .KernelFunctions

function DeterminantalPointProcess(kernel::Kernel, X::AbstractVector; parallelize=false)
    return DeterminantalPointProcess(kernelmatrix(kernel, X); parallelize=parallelize)
end

function DeterminantalPointProcess(kernel::Kernel, X::AbstractMatrix; obsdim=1, parallelize=false)
    return DeterminantalPointProcess(kernel, KernelFunctions.vec_of_vecs(X; obsdim=obsdim); parallelize=parallelize)
end