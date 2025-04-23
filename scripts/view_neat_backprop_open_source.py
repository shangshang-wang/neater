import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.visualize_net import view_classifier


if __name__ == "__main__":
    # View the classifier
    view_classifier('assets/outputs/XOR_best.out','backprop_XOR')
    view_classifier('assets/outputs/circle_best.out', 'backprop_circle')
    view_classifier('assets/outputs/spiral_best.out', 'backprop_spiral')
