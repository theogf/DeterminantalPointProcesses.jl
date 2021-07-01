# DeterminantalPointProcesses.jl

[![CI](https://github.com/theogf/DeterminantalPointProcesses.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/theogf/DeterminantalPointProcesses.jl/actions/workflows/CI.yml)
[![Coverage Status](https://coveralls.io/repos/github/theogf/DeterminantalPointProcesses.jl/badge.svg?branch=master)](https://coveralls.io/github/theogf/DeterminantalPointProcesses.jl?branch=master)

!__Disclaimer__! This package is based on the work of [alshedivat/DeterminantalPointProcesses.jl](https://github.com/alshedivat/DeterminantalPointProcesses.jl) and aims at keeping this package alive.

An efficient implementation of Determinantal Point Processes (DPP) in Julia.

### Current features
- Exact sampling [1] from DPP and k-DPP (can be executed in parallel).
- MCMC sampling [2] from DPP and k-DPP (parallelization will be added).
- `pdf` and `logpdf` evaluation functions [1] for DPP and k-DPP.

### Planned features
- Exact sampling using dual representation [1].
- Better integration with MCMC frameworks in Julia (such as [Lora.jl] or [AbstractMCMC.jl]).
- Fitting DPP and k-DPP models to data [3, 4].
- Reduced rank DPP and k-DPP.
- Kronecker Determinantal Point Processes [5].

Any help on these topics would be highly appreciated

### Contributing
Contributions are sought (especially if you are an author of a related paper).
Bug reports are welcome.

## References
[1] Kulesza, A., and B. Taskar. Determinantal point processes for machine learning. [arXiv:1207.6083], 2012.

[2] Kang, B. Fast determinantal point process sampling with application to clustering. NIPS, 2013.

[3] Gillenwater, J., A. Kulesza, E. Fox, and B. Taskar. Expectation-Maximization for learning Determinantal Point Processes. NIPS, 2014.

[4] Mariet, Z., and S. Sra. Fixed-point algorithms for learning determinantal point processes. NIPS, 2015.

[5] Mariet, Z., and S. Sra. Kronecker Determinantal Point Processes. [arXiv:1605.08374], 2016.


[Lora.jl]: https://github.com/JuliaStats/Lora.jl
[AbstractMCMC.jl]: https://github.com/TuringLang/AbstractMCMC.jl
[arXiv:1207.6083]: https://arxiv.org/abs/1207.6083
[arXiv:1605.08374]: https://arxiv.org/abs/1605.08374
