export get_polynomial, get_multi_poly, get_num_poly_vars, get_num_multipoly_vars

function get_polynomial(::Val{numvars}, ::Val{0}) where {numvars}
    # @info "Creating polynomials of order 0"
    return (t, c) -> begin 
        # println("c: $(c) $(length(c))");
        return c[1] 
    end  #, (t,c) -> ntuple(n->c[1], Val(numvars))
end

function get_polynomial(::Val{numvars}, ::Val{N}) where {numvars, N}
    # @info "Creating polynomials with $(numvars) variables of order , $(N). Required constants: $((numvars+1)^N)"
    p1 = get_polynomial(Val(numvars), Val(N-1)); # is reused multiple times
    p2(t, c) = begin
        s = p1(t, c[1+length(c)÷(numvars+1):(2)*length(c)÷(numvars+1)]) * t[1]  # int devision needed for type stability!
        # s = 0
        for n in 2:numvars
            s += p1(t, c[1+n*length(c)÷(numvars+1):(n+1)*length(c)÷(numvars+1)]) * t[n]  # int devision needed for type stability!
        end
        return s
    end
    function p3(t, c)
        # println("N: $(N), c: $(c) $(length(c))");
        p1(t, c[1:length(c)÷(numvars+1)]) + p2(t,c)   # int devision needed for type stability!
    end

    # function p3m(t, c)::NTuple{numvars, Float32}
    #     # println("N: $(N), c: $(c) $(length(c))");
    #     ntuple(n -> p1(t, c[1+(n-1)*((numvars+1)^N):length(c)÷(numvars+1) + (n-1)*((numvars+1)^N)]) + p2(t,c), Val(numvars))   # int devision needed for type stability!
    # end

    return p3 # , p3m
end

function get_multi_poly(::Val{numvars}, ::Val{N}) where {numvars, N}
    # cs_per_comp = ((numvars+1)^N)
    @info "Creating polynomials with $(numvars) variables of order , $(N). Required constants: $(numvars*((numvars+1)^N))"
    p = get_polynomial(Val(numvars), Val(N))
    # return p
    function mpol(t,c)#::NTuple{numvars, T} where T
        return ntuple(n->p(t, split_tuple(c,Val(numvars))[n]), Val(numvars))

        # println("N: $(N), c: $(c) $(length(c))");
        # return Tuple(p(t, c[1+(n-1)*((numvars+1)^N):n*((numvars+1)^N)]) for n=1:numvars)
        # return ntuple(n->p(t, c[1+(n-1)*cs_per_comp:n*cs_per_comp]), Val(numvars))
        # return ntuple(n->p(t, c[1+(n-1)*((numvars+1)^N):n*((numvars+1)^N)]), Val(numvars))
        # return (p(t, c[1+(1-1)*((numvars+1)^N):1*((numvars+1)^N)]), p(t, c[1+(2-1)*((numvars+1)^N):2*((numvars+1)^N)]))
    end
    return mpol # (t, c)->ntuple(n->p(t, c[1+(n-1)*((numvars+1)^N):n*((numvars+1)^N)]), Val(numvars))
    # return (t, c) -> Tuple(p(t,c[1+(n-1)*((numvars+1)^N):n*((numvars+1)^N)]) for n=1:numvars)
end

function get_num_poly_vars(::Val{numvars}, ::Val{N}) where {numvars, N}
    return (numvars+1)^N
end

function get_num_multipoly_vars(::Val{numvars}, ::Val{N}) where {numvars, N}
    return numvars*get_num_poly_vars(Val(numvars), Val(N))
end

function test_poly_allocations()
    get_num_poly_vars(Val(2), Val(3)) # (2+1)^3
    p = get_polynomial(Val(2), Val(3))  # 27 indices required
    @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)));
    # does allocate 9 Mb !

    p = get_polynomial(Val(3), Val(2))  # 9 indices required
    get_num_poly_vars(Val(3), Val(2))
    @time p.(Tuple.(CartesianIndices((100,100,10))),Ref((1.1,2.1,3.1,4,5,6,7,8,9,10,11,12,13,14,15,16)));
    # does allocate 160 Mb !

    p = get_polynomial(Val(2), Val(2))  # 9 indices required
    @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2.1,3.1,4,5,6,7,8,9)));
    # essentially allocation-free

    p = get_polynomial(Val(2), Val(1))  # 3 indices required
    @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2.1,3.1)));
    # essentially allocation-free
    # p((100,100),((1.1,2.2, 3.3)))

    p = get_polynomial(Val(1), Val(0))  # 27 indices required
    @time p.(Tuple.(CartesianIndices((100,100))),Ref((1.1)));
    # essentially allocation-free

end
