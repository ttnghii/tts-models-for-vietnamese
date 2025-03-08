import matplotlib.pyplot as plt


def plot_spectrogram(spectrogram) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()
    return fig
