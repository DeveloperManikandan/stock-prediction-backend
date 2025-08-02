import os
import matplotlib.pyplot as plt
from django.conf import settings

def save_plot(plot_img_path):
    # Full file path
    image_path = os.path.join(settings.MEDIA_ROOT, plot_img_path)

    # Create directory if it does not exist
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    # Save the plot image
    plt.savefig(image_path)
    plt.close()

    # Return URL path to the image (for frontend)
    image_url = settings.MEDIA_URL + plot_img_path
    return image_url
