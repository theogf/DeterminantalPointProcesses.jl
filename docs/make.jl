using Documenter, DeterminantalPointProcesses

makedocs(;
    modules=[DeterminantalPointProcesses],
    format=Documenter.Writers.HTMLWriter.HTML(;
        assets=["assets/icon.ico"], analytics="UA-129106538-2"
    ),
    sitename="DeterminantalPointProcesses",
    authors="Theo Galy-Fajou, Maruan Al-Shedivat",
    pages=["Home" => "index.md"],
)

deploydocs(;
    deps=Deps.pip("mkdocs", "python-markdown-math"),
    repo="github.com/theogf/DeterminantalPointProcesses.jl.git",
    target="build",
)
