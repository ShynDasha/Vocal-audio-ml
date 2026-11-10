"""Show the local pipeline commands without starting expensive work."""
if __name__ == '__main__':
    print("1. python prepare_vocalset.py [--download]")
    print("2. python audio_features.py --max-files 200  (or --all)")
    print("3. python train.py")
