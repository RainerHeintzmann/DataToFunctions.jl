using DataToFunctions
using TestImages

function main()

    obj = Float32.(TestImages.shepp_logan(320));

    f = get_function_poly(obj,1)
    c0 = (-10.5, 1.1, 0.1, -20.2, 0.1, 1.1)
    @time warped = f(c0); # 1.7 ms

    fa = get_function_affine(obj); #.+ dtype.(rand(size(sample_data)...))./5.0;
    ca = [1.5, 1.1, 0.6, 1.2, 0.1, 1.1, 2.0]
    @time warpeda = fa(ca); # 1.7 ms

    # non-linear deformation warp
    f2 = get_function_poly(obj, 2); #.+ dtype.(rand(size(sample_data)...))./5.0;
    c02 = (-150.5, 1.1, 0.1, 0.001, 0.001 ,0.001, 0.001, 0.001, 0.001,
           -110.2, 0.1, 1.5, 0.001, 0.0015,-0.001, 0.0014,0.001,-0.001)
    @time warped2 = f2(c02); # 3.3 ms
    @time warped2 .= f2(c02);
    @time f2(c02, warped2);  # 3.0 ms
    # @vt obj warpeda warped warped2

end

