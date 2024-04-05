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

    return p3
end

function main()
    (2+1)^3
    p = get_polynomial(Val(2), Val(3))  # 27 indices required
    @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)));
    # does allocate

    (2+1)^2
    p = get_polynomial(Val(2), Val(2))  # 3 indices required
    @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2.1,3.1,4,5,6,7,8,9)));
    # essentially allocation-free

    (2+1)^1
    p = get_polynomial(Val(1), Val(1))  # 3 indices required
    @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2.1,3.1)));
    # essentially allocation-free
    p((100,),((1.1,2.2, 3.3)))

    p = get_polynomial(Val(1), Val(0))  # 27 indices required
    @time p.(Tuple.(CartesianIndices((100,100))),Ref((1.1)));
    # essentially allocation-free
end

