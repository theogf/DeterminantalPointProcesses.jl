using DeterminantalPointProcesses
using Test
using Random, LinearAlgebra, IterTools
using KernelFunctions
import Combinatorics: combinations

rng = MersenneTwister(42)

n = 5
k = 2
A = rand(rng, n, n)
L = Symmetric(A * A')
kernel = SqExponentialKernel()
X = [rand(2) for _ in 1:3]
dpp = DPP(L)

@testset "Ensure correct pdf and logpdf computation" begin
    @testset "Kernel API" begin
        K = kernelmatrix(kernel, X)
        @test DPP(K).L ≈ DPP(kernel, X).L
        @test DPP(K).L ≈ DPP(kernel, reduce(hcat, X); obsdim=2).L
    end
    @testset "For DPP" begin
        # compute the true distribution
        true_pdf = Float64[]
        true_logpdf = Float64[]
        for z in subsets(1:n)
            push!(true_pdf, pdf(dpp, z))
            push!(true_logpdf, logpdf(dpp, z))
        end
        @test sum(true_pdf) ≈ 1.0
        @test all(true_pdf .<= 1.0)
        @test all(true_logpdf .<= 0.0)
    end

    @testset "For k-DPP" begin
        # compute the true distribution
        true_pdf = Float64[]
        true_logpdf = Float64[]
        for z in combinations(1:n, k)
            push!(true_pdf, pdf(dpp(k), z))
            push!(true_logpdf, logpdf(dpp(k), z))
        end
        @test sum(true_pdf) ≈ 1.0
        @test all(true_pdf .<= 1.0)
        @test all(true_logpdf .<= 0.0)
    end
end

@testset "Ensure correct sampling from DPP" begin
    # compute the true distribution
    all_subsets = []
    true_pdf = Float64[]
    for (i, z) in enumerate(subsets(1:n))
        push!(true_pdf, pdf(dpp, z))
        push!(all_subsets, (z, i))
    end
    global subset_to_idx = Dict(all_subsets)

    @testset "Exact sampling" begin
        nb_samples = 1000
        samples = rand(dpp, nb_samples)

        # compute the empirical distribution
        empirical_pdf = zeros(Float64, length(true_pdf))
        for z in samples
            empirical_pdf[subset_to_idx[z]] += 1
        end
        empirical_pdf ./= nb_samples

        # ensure that the empirical pdf is close to the true pdf
        total_variation = maximum(abs.(true_pdf - empirical_pdf))
        @test total_variation ≈ 0.0 atol = 1e-1
    end

    @testset "MCMC sampling" begin
        nb_samples = 1000
        samples, state = randmcmc(dpp, nb_samples; return_final_state=true)

        # ensure that L_z_inv makes sense (i.e., noise did not accumulate)
        z, L_z_inv = state
        @test sum(L_z_inv[z, z] * dpp.L[z, z] - I) ≈ 0 atol=1e-10

        # compute the empirical distribution
        empirical_pdf = zeros(Float64, length(true_pdf))
        for z in samples
            empirical_pdf[subset_to_idx[z]] += 1
        end
        empirical_pdf ./= nb_samples

        # ensure that the empirical pdf is close to the true pdf
        total_variation = maximum(abs.(true_pdf - empirical_pdf))
        @test total_variation ≈ 0.0 atol = 1e-1
    end
end

@testset "Ensure correct sampling from k-DPP" begin
    # compute the true distribution
    all_k_subsets = []
    true_pdf = Float64[]
    for (i, z) in enumerate(combinations(1:n, k))
        push!(true_pdf, pdf(dpp, z))
        push!(all_k_subsets, (z, i))
    end
    true_pdf ./= sum(true_pdf)
    k_subset_to_idx = Dict(all_k_subsets)

    @testset "Exact sampling" begin
        nb_samples = 10000
        samples = rand(dpp(k), nb_samples)

        # ensure that samples are of proper cardinality
        @test all(map(length, samples) .== k)

        # compute the empirical distribution
        empirical_pdf = zeros(Float64, length(true_pdf))
        for z in samples
            empirical_pdf[k_subset_to_idx[z]] += 1
        end
        empirical_pdf ./= nb_samples

        # ensure that the empirical pdf is close to the true pdf
        total_variation = maximum(abs.(true_pdf - empirical_pdf))
        @test total_variation ≈ 0.0 atol = 1e-1
    end

    @testset "MCMC sampling" begin
        nb_samples = 1000
        samples, state = randmcmc(dpp(k), nb_samples; return_final_state=true)

        # ensure that L_z_inv makes sense (i.e., noise did not accumulate)
        z, L_z_inv = state
        @test sum(L_z_inv[z, z] * dpp.L[z, z] - I) ≈ 0 atol = 1e-1

        # ensure that samples are of proper cardinality
        @test all(map(length, samples) .== k)

        # compute the empirical distribution
        empirical_pdf = zeros(Float64, length(true_pdf))
        for z in samples
            empirical_pdf[k_subset_to_idx[z]] += 1
        end
        empirical_pdf ./= nb_samples

        # ensure that the empirical pdf is close to the true pdf
        total_variation = maximum(abs.(true_pdf - empirical_pdf))
        @test_broken total_variation ≈ 0.0
        atol = 1e-1
    end
end
