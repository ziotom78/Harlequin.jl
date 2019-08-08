using Documenter, Harlequin

makedocs(;
    modules=[Harlequin],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/ziotom78/Harlequin.jl/blob/{commit}{path}#L{line}",
    sitename="Harlequin.jl",
    authors="Maurizio Tomasi",
    assets=String[],
)

deploydocs(;
    repo="github.com/ziotom78/Harlequin.jl",
)
