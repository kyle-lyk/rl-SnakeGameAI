import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('DRL Snake Game Plot')
    display.display(plt.gcf()) # GCF = Get Current Figure

    plt.clf() # Clear Current Figure (Refresh)

    plt.title('Training Snake Game AI...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')

    plt.ylim(ymin=0) # Y-Limits
    plt.legend()

    plt.text(len(scores)-1, scores[-1], str(scores[-1])) # Last Score
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1])) # Last Mean Score
    plt.show(block=False)
    plt.pause(2)

    

