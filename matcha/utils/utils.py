import matplotlib.pyplot as plt

def plot_spectrogram(spectrogram, title="Mel Spectrogram"):
    """
    Affiche un spectrogramme de Mel.
    """
    # Si le spectrogramme a une dimension de batch [1, C, T], on la retire
    if len(spectrogram.shape) == 3:
        spectrogram = spectrogram.squeeze(0)
    
    # Conversion en numpy pour Matplotlib
    data = spectrogram.cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    img = plt.imshow(data, aspect='auto', origin='lower', cmap='Purples')
    
    plt.colorbar(img, label='Log Intensity')
    plt.title(title)
    plt.xlabel('Time (Frames)')
    plt.ylabel('Mel Channels')
    plt.tight_layout()
    plt.show()