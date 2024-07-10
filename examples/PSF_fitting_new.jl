using PointSpreadFunctions
using Plots

λ_em = 0.5; NA = 1.4; n = 1.52
λ_ex = 0.488 # only needed for some PointSpreadFunctions, such as confocal, ISM or TwoPhoton
pp = PSFParams(λ_em, NA, n; pol=pol_x)

sz = (256, 256, 256)
sampling = (0.020,0.020,0.020)

aberr_sp = Aberrations([Zernike_VerticalAstigmatism],[1.0]); sz=(256,256,256)
pp_sp = PSFParams(λ_em, NA, n; method=MethodPropagateIterative, aberrations= aberr_sp)
p_sp = psf(sz, pp_sp; sampling=sampling);

psf_example = sum(p_sp, dims=3)[:,:,1]

heatmap(p_sp[:, :, 128], aspect_ratio=1)