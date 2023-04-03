using Zygote
using .DataToFunctions

data = rand(11,10)
f = get_function(data; super_sampling=2);

loss(p, z) = sum(abs2.(f(p, z) .- data))

# Zygote.forwarddiff needs only one input
loss(p2::Vector{Tuple{Float64, Float64}}) = loss(p2[1], p2[2])


for i in 1:20
    pr = [(0.0, -0.1+i/100.0), (1.0, 1.0)]
    
    # to take derivative of a interpolation process, it is better to use the Zygote.forwarddiff
    loss_grad = Zygote.forwarddiff(loss, pr)

    # it can be seen that increasing the scale variable (y direction) leads to increase in the loss function gradient value
    println(loss_grad)
    
end

array_scale = Array{Tuple{Float64, Float64}, 2}(undef, 200, 2)
list_loss_grad_scale = Array{Float64, 2}(undef, 200, 200)

x_1 = 0.9:0.01:1.1
y_1 = 0.9:0.01:1.1


for i in 1:200
    for j in 1:200
        array_scale[i, :] = [(0.0, 0.0), (0.9 + j/1000.0, 0.9 + i/1000.0)]

        #pr = [(0.0, 0.0), loss_scale[1, :]]
        
        # to take derivative of a interpolation process, it is better to use the Zygote.forwarddiff
        list_loss_grad_scale[i, j] = Zygote.forwarddiff(loss, array_scale[i, :])

        # it can be seen that increasing the scale variable (y direction) leads to increase in the loss function gradient value
        #println(loss_grad)
    end
    
end

surface(1:200, 1:200, list_loss_grad_scale)

