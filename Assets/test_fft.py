import numpy as np
import matplotlib.pyplot as plt


def gauss(rng: np.random.Generator) -> float:
    """
    Mathematica: RandomVariate[NormalDistribution[]]
    => Standard normal N(0,1).
    """
    return float(rng.standard_normal())


def ifft2_numpy_convention(A: np.ndarray) -> np.ndarray:
    """
    Explicit inverse-DFT convention used by NumPy's np.fft.ifft2:

      x[n1, n2] = (1/(N1*N2)) * sum_{k1=0..N1-1} sum_{k2=0..N2-1}
                   A[k1, k2] * exp(+2π i * (k1*n1/N1 + k2*n2/N2))

    This returns complex values; for a Hermitian spectrum the imaginary part should be ~0.
    """
    total_coefficients = len(A)
    crop_start = int(total_coefficients * 0.005)
    
    for x in range(crop_start, total_coefficients):
        for y in range(crop_start, total_coefficients):
            A[x,y] = 0.0
        
    return np.fft.ifft2(A)


def spectral_synthesis_fm2d(N: int, H: float, seed: int) -> np.ndarray:
    """
    Direct translation of Mathematica SpectralSynthesisFM2D with explicit assumptions.

    - N must be even (uses N/2 indices).
    - Builds complex spectrum A (NxN) with Hermitian symmetry.
    - Forces Nyquist axis points to be purely real.
    - Returns Re[InverseFourier[A]] under NumPy's explicit inverse FFT convention above.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if N % 2 != 0:
        raise ValueError("N must be even (the Mathematica code uses N/2).")

    rng = np.random.default_rng(seed)
    A = np.zeros((N, N), dtype=np.complex128)
    half = N // 2

    # --------- First double loop: i,j = 0..N/2 ----------
    # fills the grid as #0
    #                   0#
    for i in range(0, half + 1):
        for j in range(0, half + 1):
            phase = 2.0 * np.pi * float(rng.random())  # RandomReal[] in [0,1)
            if i != 0 or j != 0:
                r2 = float(i * i + j * j)
                rad = (r2 ** (-(H + 1.0) / 2.0)) * gauss(rng)
            else:
                rad = 0.0

            z = rad * np.exp(1j * phase)

            # SetA[i, j, z]
            A[i, j] = z

            # i0 = (i==0)?0:(N-i), same for j0
            i0 = 0 if i == 0 else (N - i)
            j0 = 0 if j == 0 else (N - j)

            # SetA[i0, j0, Conjugate[z]]
            A[i0, j0] = np.conjugate(z)
            
    # --------- Force Nyquist axis entries to be real ----------
    # Mathematica:
    # SetA[N/2,0, Re[GetA[N/2,0]]], etc.
    A[half, 0] = A[half, 0].real + 0j
    A[0, half] = A[0, half].real + 0j
    A[half, half] = A[half, half].real + 0j

    # --------- Second double loop: i,j = 1..N/2-1 ----------
    # fills the grid as #%
    #                   %#
    for i in range(1, half):
        for j in range(1, half):
            phase = 2.0 * np.pi * float(rng.random())
            r2 = float(i * i + j * j)
            rad = (r2 ** (-(H + 1.0) / 2.0)) * gauss(rng)

            z = rad * np.exp(1j * phase)

            # SetA[i, N-j, z]
            A[i, N - j] = z
            # SetA[N-i, j, Conjugate[z]]
            A[N - i, j] = np.conjugate(z)
            
    # --------- Inverse FFT (explicit convention above) ----------
    field_complex = ifft2_numpy_convention(A)
    field_real = field_complex.real  # Mathematica: Re[InverseFourier[A]]

    return field_real


def plot_surface(Z: np.ndarray, stride: int = 4) -> None:
    """
    Simple 3D surface plot similar to ListPlot3D[..., Mesh->None].
    Stride controls decimation for performance.
    """
    N = Z.shape[0]
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing="xy")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        rstride=stride, cstride=stride,
        cmap="terrain",  # close spirit to "GreenBrownTerrain"
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("height")
    ax.set_title("Spectral synthesis terrain (NumPy IFFT convention)")
    plt.show()


if __name__ == "__main__":
    N = 16
    H = 0.86
    seed = 1

    Z = spectral_synthesis_fm2d(N, H, seed)
    print("Stats:", Z.min(), Z.max(), Z.mean(), Z.std())

    # Plot (decimate for speed; try stride=2 for denser mesh)
    plot_surface(Z, stride=4)
