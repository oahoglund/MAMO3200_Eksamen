"""
Program som inneholder alle funksjonene for visualisering
"""

from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

def visualizeP_snapshot(P,X,Y,V, save = False, filename = "snapshot"):
    mlab.figure(size=(800, 700), bgcolor=(1, 1, 1),)
    V_min = V.min()
    V_norm = (V - V_min) / (V.max() - V_min)
    V_scaled = V_norm * P.max()
    
    zmin = P.min()
    zmax = P.max()

    m = mlab.surf(X, Y, P, warp_scale='auto', 
                  vmin=zmin, vmax=zmax,colormap='viridis')
    axes = mlab.axes(
        ranges=(X.min(), X.max(), Y.min(), Y.max(), zmin, zmax),
        xlabel='x [m]', ylabel='y [m]', zlabel='p(x,y)',
        nb_labels=3, line_width=1.0, color=(0,0,0)
    )
    axes.label_text_property.color = (0,0,0)
    axes.title_text_property.color = (0,0,0)
    mlab.outline(line_width=1.0, color=(0,0,0))
    h = mlab.surf(X, Y, V_scaled, color=(1,1,1), warp_scale='auto',opacity=0.3)
    cb = mlab.colorbar(m, title='p(x,y)', orientation='vertical')
    cb.scalar_bar_representation.position = [0.9, 0.1] # type: ignore
    mlab.view(azimuth=45, elevation=60, distance='auto')

    if save:
        filename = os.path.join("bilder", filename)
        mlab.savefig(filename)
        mlab.close()
    else:
        mlab.show()

def visualizeV(V,X,Y):
    mlab.figure(size=(800, 600), bgcolor=(1, 1, 1),)

    m = mlab.surf(X, Y, V, color=(0,0,1), warp_scale='auto')
    axes = mlab.axes(
        ranges=(X.min(), X.max(), Y.min(), Y.max(), V.min(), V.max()),
        xlabel='x [m]', ylabel='y [m]', zlabel='v(x,y)',
        nb_labels=3, line_width=1.0, color=(0,0,0)
    )
    axes.label_text_property.color = (0,0,0)  # axis label text color
    axes.title_text_property.color = (0,0,0)  # title ("x", "y", "z" text) color
    mlab.outline(line_width=1.0, color=(0,0,0))
    mlab.view(azimuth=45, elevation=60, distance='auto')

    mlab.show()

def animateP(P,X,Y,t,V,fps=30):
    """
    Animerer punktsannsyneligheten P[n,i,j], første er for tid, andre x og siste y
    Krever X og Y med samme xy dimensjon som kommer fra meshgrid
    Viser også til hvilke tidspuntker det gjelder. Tidspunktene er en array t
    av lengde n. 
    """
    mlab.figure(size=(800, 700), bgcolor=(1, 1, 1),)
    P0 = P[0]
    v0 = V.max()
    V_min = V.min()
    V_norm = (V - V_min) / (v0 - V_min)
    V_scaled = V_norm * P0.max()
    zmin = P0.min()
    zmax = P0.max()

    m = mlab.surf(X, Y, P0, warp_scale='auto', 
                  vmin=zmin, vmax=zmax,colormap='viridis')
    axes = mlab.axes(
        ranges=(X.min(), X.max(), Y.min(), Y.max(), zmin, zmax),
        xlabel='x [m]', ylabel='y [m]', zlabel='p(x,y)',
        nb_labels=3, line_width=1.0, color=(0,0,0)
    )
    axes.label_text_property.color = (0,0,0)
    axes.title_text_property.color = (0,0,0)
    mlab.outline(line_width=1.0, color=(0,0,0))
    h = mlab.surf(X, Y, V_scaled, color=(1,1,1), warp_scale='auto',opacity=0.3)
    cb = mlab.colorbar(m, title='p(x,y)', orientation='vertical')
    cb.scalar_bar_representation.position = [0.9, 0.1] # type: ignore
    mlab.view(azimuth=45, elevation=60, distance='auto')

    v0_label = mlab.text(0.05, 0.80, f"v0 = {v0:.3e}", width=0.2, color=(0,0,0))
    time_label = mlab.text(0.05, 0.90, f"t = 0.0s", width=0.2, color=(0,0,0))

    @mlab.animate(delay=int(1000/fps))
    def update_animation():
        i = 0
        total_frames = len(P)
        while True:
            n = i % total_frames
            m.mlab_source.scalars = P[n]
            time_label.text = f"t = {t[n]:.4e}s"
            i += 1
            yield
    anim = update_animation() # type: ignore
    mlab.show()

def animateP_save(P,X,Y,t,V,fps=30,
             outdir="frames",
             gif_file="animation.gif",
             gif_width=800):
    """
    Animerer punktsannsyneligheten P[n,i,j], første er for tid, andre x og siste y
    Krever X og Y med samme xy dimensjon som kommer fra meshgrid
    Viser også til hvilke tidspuntker det gjelder. Tidspunktene er en array t
    av lengde n. 
    """
    mlab.figure(size=(800, 700), bgcolor=(1, 1, 1),)
    P0 = P[0]
    v0 = V.max()
    V_min = V.min()
    V_norm = (V - V_min) / (v0 - V_min)
    V_scaled = V_norm * P0.max()
    zmin = P0.min()
    zmax = P0.max()

    m = mlab.surf(X, Y, P0, warp_scale='auto', 
                vmin=zmin, vmax=zmax, colormap='viridis')
    axes = mlab.axes(
        ranges=(X.min(), X.max(), Y.min(), Y.max(), zmin, zmax),
        xlabel='x [m]', ylabel='y [m]', zlabel='p(x,y)',
        nb_labels=3, line_width=1.0, color=(0,0,0)
    )
    axes.label_text_property.color = (0,0,0)  # axis label text color
    axes.title_text_property.color = (0,0,0)  # title ("x", "y", "z" text) color
    mlab.outline(line_width=1.0, color=(0,0,0))
    h = mlab.surf(X, Y, V_scaled, color=(1,1,1), warp_scale='auto',opacity=0.3)
    cb = mlab.colorbar(m, title='p(x,y)', orientation='vertical')
    cb.scalar_bar_representation.position = [0.9, 0.1] # type: ignore
    mlab.view(azimuth=45, elevation=60, distance='auto')

    v0_label = mlab.text(0.05, 0.80, f"v0 = {v0:.3e}", width=0.2, color=(0,0,0))
    time_label = mlab.text(0.05, 0.90, f"t = 0.0s", width=0.2, color=(0,0,0))

    for i in range(len(P)):
        m.mlab_source.scalars = P[i]
        time_label.text = f"t = {t[i]:.4e}s"
        mlab.process_ui_events()

        filename = os.path.join(outdir, f"frame_{i:05d}.png")
        mlab.savefig(filename)

    mlab.close()  # lukk figuren, alt er lagret
    # --- ffmpeg GIF-pipeline: palette + GIF ---
    palette = os.path.join(outdir, "palette.png")

    cmd_palette = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(outdir, "frame_%05d.png"),
        "-vf", f"scale={gif_width}:-1:flags=lanczos,palettegen",
        palette
    ]

    cmd_gif = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(outdir, "frame_%05d.png"),
        "-i", palette,
        "-lavfi", f"scale={gif_width}:-1:flags=lanczos [x]; [x][1:v] paletteuse",
        gif_file
    ]

    subprocess.run(cmd_palette, check=True)
    subprocess.run(cmd_gif, check=True)

    # --- Cleanup: delete frames + palette ---
    for f in os.listdir(outdir):
        path = os.path.join(outdir, f)
        if f.endswith(".png"):
            os.remove(path)

    print(f"GIF lagret: {gif_file}")

def visualizeTotalP(p: np.ndarray, t: np.ndarray, v0, name: str, filename: str, save: bool = False) -> None:
    """
    Tar inn en p array, hvert element i arrayen tilsvarer den totalale sannsyneligheten til bølgefunksjonen til den tiden
    Tar inn enn t array som inneholder tidspunktene til observasjonene
    Plotter dette
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t, p, linewidth=2)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Total sannsynlighet")
    ax.set_title(f"Total sannsynlighet over tid med v_0 = {v0:.1e}\n for {name}")
    ax.grid(True)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    if save:
        filename = os.path.join("bilder", filename)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def visualizeTotalPs(Ps: dict, t: np.ndarray, save: bool = False) -> None:
    """
    Tar inn en p array, hvert element i arrayen tilsvarer den totalale sannsyneligheten til bølgefunksjonen til den tiden
    Tar inn enn t array som inneholder tidspunktene til observasjonene
    Plotter dette
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, p in Ps.items():
        ax.plot(t, p, linewidth=2, label = name)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Total sannsynlighet")
    ax.set_title("Total sannsynlighet over tid")
    ax.grid(True)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend()
    #if save:
    #    fig.savefig(name, dpi=300, bbox_inches="tight")

    plt.show()




if __name__ == "__main__":
    print("Hello")