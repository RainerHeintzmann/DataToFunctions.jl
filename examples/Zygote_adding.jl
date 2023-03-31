using Zygote
using DataToFunctions

data = rand(11,10)
f = get_function(data; super_sampling=2);

loss(p, z) = sum(abs2.(f(p, z) .- data))

# Zygote.forwarddiff needs only one input
loss(p2::Vector{Tuple{Float64, Float64}}) = loss(p2[1], p2[2])


for i in 1:10
    pr = [(0.0, 0.0), (1.0, 1.0+i/100.0)]
    
    # to take derivative of a interpolation process, it is better to use the Zygote.forwarddiff
    loss_grad = Zygote.forwarddiff(loss, pr)

    # it can be seen that increasing the scale variable (y direction) leads to increase in the loss function gradient value
    println(loss_grad)
    
end


