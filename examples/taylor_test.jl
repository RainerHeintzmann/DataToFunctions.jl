# using TaylorSeries
# using StaticArrays

# function to_SVec(c::CartesianIndex)
#     return [Tuple(c)...]
# end

# function main()
#     t = set_variables("x", numvars=3, order=4)
#     p = exp.(t)
#     @time q = evaluate.(Ref(p), to_SVec.(CartesianIndices(img)));
#     @time q .= evaluate.(Ref(p), to_SVec.(CartesianIndices(img)));
# end


function get_polynomial(::Val{numvars}, ::Val{0}) where {numvars}
    # @info "Creating polynomials of order 0"
    return (t, c) -> begin 
    # println("c: $(c) $(length(c))");
    c[1]
    end
end

function get_polynomial(::Val{numvars}, ::Val{N}) where {numvars, N}
    # @info "Creating polynomials with $(numvars) variables of order , $(N). Required constants: $((numvars+1)^N)"
    p1 = get_polynomial(Val(numvars), Val(N-1)); # is reused multiple times
    return (t, c) -> begin
        # println("N: $(N), c: $(c) $(length(c))");
        p1(t, c[1:length(c)/(numvars+1)]) + sum(p1(t, c[1+n*length(c)/(numvars+1):(n+1)*length(c)/(numvars+1)]) * t[n] for n in 1:numvars)
    end
end

function main()
    p = p = get_polynomial(Val(2), Val(3))  # 27 indices required
    @time p.(Tuple.(CartesianIndices((100,100))),Ref((1.1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)));
end

