from matplotlib import pyplot as plt

def generate_plot(f1, f2, g1, g2, label, metric1="Loss", metric2="Accuracy"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(f1)
    ax[0].plot(f2)
    ax[0].set_title(f'Model {label} - {metric1}')
    ax[0].set_ylabel(metric1)
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['train', 'val'], loc='best')
    
    ax[1].plot(g1)
    ax[1].plot(g2)
    ax[1].set_title(f'Model {label} - {metric2}')
    ax[1].set_ylabel(metric2)
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['train', 'val'], loc='best')
    
    filename = f'plot_{label}.png'
    fig.savefig(filename)
    
    plt.show()

