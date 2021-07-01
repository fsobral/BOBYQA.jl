# Turn off precompilation during development
__precompile__(false)

# Main file implementing BOBYQA algorithm in pure Julia

module BOBYQA

import Base: (*)
using LinearAlgebra

# Main module

# Add types
include("bobyqa_types.jl")

include("auxiliary_functions.jl")
include("trsbox.jl")
include("altmov.jl")

end
