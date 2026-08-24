import Pkg
Pkg.activate(@__DIR__)

cd(@__DIR__) # go into `docs` folder

using Documenter, Literate, DataToFunctions

# convert tutorial/examples to markdown
# Literate.markdown("./src/tutorial.jl", "./src")
# Which markdown files to compile to HTML
# (which is also the sidebar and the table
# of contents for your documentation)

pages = Any[
        "Introduction" => "index.md",
        "Tutorial" => "tutorial.md",
        "API" => "api.md",
        ]
    
# compile to HTML:
makedocs(; sitename="DataToFunctions.jl", pages, modules = [DataToFunctions], checkdocs = :exports)

deploydocs(repo = "github.com/RainerHeintzmann/DataToFunctions.jl.git", devbranch = "develop")