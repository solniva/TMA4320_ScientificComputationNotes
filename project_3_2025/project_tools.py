import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def imshow_plot_u(U, Lx, Ly, cblabel=r'$U$'):
    """
    Visualizes a 2D array `U` as a heatmap using matplotlib's imshow function.
    Parameters:
    -----------
    U : numpy.ndarray
        A 2D array representing the data to be visualized.
    Lx : float
        The length of the domain in the x-direction.
    Ly : float
        The length of the domain in the y-direction.
    cblabel : str, optional
        Label for the colorbar. Default is r'$U$'.
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    Notes:
    ------
    - The colormap used is 'RdBu_r', which is a diverging colormap.
    - The color limits are set to the minimum and maximum values of `U`.
    - The x and y axes are labeled as r'$x$' and r'$y$', respectively.
    - The colorbar is added to the plot with the specified label.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.imshow(U, cmap='RdBu_r', interpolation='bilinear', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label(cblabel)
    img.set_clim(vmin=U.min(), vmax=U.max())
    return fig, ax

def imshow_plot_ut(U, t, Lx, Ly, cblabel=r'$U$'):
    """
    Visualizes a 2D array `U` as an image with a colorbar and adds a time annotation.
    Parameters:
    -----------
    U : ndarray
        A 2D array representing the data to be visualized.
    t : float
        The time value to be displayed as an annotation on the plot.
    Lx : float
        The length of the domain in the x-direction.
    Ly : float
        The length of the domain in the y-direction.
    cblabel : str, optional
        The label for the colorbar. Default is r'$U$'.
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    
    fig, ax = imshow_plot_u(U, Lx, Ly, cblabel)
    tx = ax.text(0, Ly/2*1.1, f"t={t:.3f}",
                 bbox=dict(boxstyle="round", ec='white', fc='white'))
    return fig, ax

def plot_stability_domain(r, xlim=(-3, 3), ylim=(-3, 3), resolution=500):
    """
    Plots the stability domain of a given rational function r(z) where |r(z)| <= 1.
    
    Parameters:
    - r: A function representing the rational function r(z).
    - xlim: Tuple representing the x-axis limits for the plot.
    - ylim: Tuple representing the y-axis limits for the plot.
    - resolution: The number of points along each axis for the grid.
    """
    
    # Create a grid of complex numbers
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Compute |r(z)| on the grid
    R = np.abs(r(Z))
    
    # Plot the stability domain
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, R, levels=[0, 1], colors=['blue'], alpha=0.5)
    plt.contour(X, Y, R, levels=[1], colors=['black'])
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Stability Domain')
    plt.grid(True)
    # plt.show()
    

def create_animation(Uts, Lx, Ly):
    U0, t0 = Uts[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.imshow(U0, cmap='RdBu_r', interpolation='bilinear', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    tx = ax.text(0,Ly/2*1.1,f"t={t0:.3f}",
                bbox=dict(boxstyle="round",ec='white',fc='white'))
    cbar = plt.colorbar(img, ax=ax)
    img.set_clim(vmin=-1, vmax=1)
    
    # Define the animation function 
    def animate(Ut):
        # Unpack the tuple
        U, t = Ut
        # Set image data and text
        img.set_data(U)
        tx.set_text(f"t={t:.3f}")
    
    ani = animation.FuncAnimation(fig, animate, frames=Uts, interval=200, blit=False, repeat=True)
    return ani
    
    