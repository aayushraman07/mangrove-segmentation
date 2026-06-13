import argparse
import tensorflow as tf
import cv2
import os
from src.config import CONFIG, MODEL_PATH, STATS_PATH
from src.inference import predict_tiled, prepare_drone_frame
from src.utils import load_or_compute_stats

os.makedirs("predictions", exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to TIF, JPG or MP4")
    parser.add_argument("--mode", choices=["satellite", "drone", "video"], default="satellite")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load model + stats
    mean, std = load_or_compute_stats()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if args.mode == "satellite":
        # Add satellite inference logic here (from notebook)
        print("Satellite inference - implement full logic from notebook")
    elif args.mode == "drone":
        img = cv2.imread(args.input)
        data = prepare_drone_frame(img)
        prob_map = predict_tiled(model, data, mean, std)
        # Save overlay...
        print(f"Drone inference complete. Coverage: {100*(prob_map > args.threshold).mean():.1f}%")
    # Add video mode similarly

if __name__ == "__main__":
    main()