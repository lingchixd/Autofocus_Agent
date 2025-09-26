import numpy as np
import skimage.io
from skimage import exposure, img_as_ubyte
from microscopeimagequality.degrade import get_airy_psf, ImageDegrader

# read 16-bit TIF，normalize to [0,1]
img_path = r"C:\Users\Lingchi Deng\Desktop\Script\paper_one\test_focus.tiff"
image_uint16 = skimage.io.imread(img_path)
image = image_uint16.astype(np.float32) / 65535.0

# optical parameter
wavelength = 500e-9
numerical_aperture = 0.5
refractive_index = 1.0
pixel_size_meters = 0.65e-6
psf_width_pixels = 51

# z stack
z_values = [ 0.0, 8e-6, 16e-6, 24e-6, 32e-6, 40e-6 ]

degrader = ImageDegrader(random_seed=0,
                         photoelectron_factor=65535,
                         sensor_offset_in_photoelectrons=100)

for z in z_values:
    # psf generation
    psf_width_meters = psf_width_pixels * pixel_size_meters
    psf = get_airy_psf(psf_width_pixels, psf_width_meters, z,
                       wavelength, numerical_aperture, refractive_index)

    # blurred, exposed and noisy
    blurred = degrader.apply_blur_kernel(image, psf)
    exposed = degrader.set_exposure(blurred, exposure_factor=1.0)
    noisy   = degrader.random_noise(exposed)

    # save in PNG (8-bit)
    out_file = f"degraded_z{z*1e6:+.0f}um.png"
    
    print("min/max:", float(noisy.min()), float(noisy.max()))
    vis = exposure.rescale_intensity(noisy, in_range='image')
    vis = exposure.adjust_gamma(vis, gamma=1.1)
    skimage.io.imsave(out_file, img_as_ubyte(vis), check_contrast=False)
