def inspect_file_raw(file_path, num_bytes=100):
    print(f"\n=== Inspecting {file_path} ===")
    
    # Read as binary
    print("First 100 bytes (binary):")
    try:
        with open(file_path, 'rb') as f:
            content = f.read(num_bytes)
            print(content)
    except Exception as e:
        print(f"Error reading binary: {e}")

    # Read as text (in case it's not binary)
    print("\nFirst 100 characters (text):")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(num_bytes)
            print(content)
    except Exception as e:
        print(f"Error reading text: {e}")

if __name__ == "__main__":
    files = [
        "saved_models/kernels.pkl",
        "saved_models/models.pkl",
        "saved_models/scaler.pkl"
    ]
    
    for file_path in files:
        inspect_file_raw(file_path)