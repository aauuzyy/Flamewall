"""
Quick test script to verify RLGym setup
Run this after installing dependencies to check if everything works
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    
    print("Testing RLGym setup...\n")
    
    tests = [
        ("RLBot", "rlbot"),
        ("RLGym", "rlgym"),
        ("Stable-Baselines3", "stable_baselines3"),
        ("PyTorch", "torch"),
        ("NumPy", "numpy"),
        ("TensorBoard", "tensorboard"),
    ]
    
    failed = []
    
    for name, module in tests:
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            failed.append(name)
    
    print("\n" + "="*50)
    
    if not failed:
        print("✓ All dependencies installed successfully!")
        print("\nYou're ready to train! Run:")
        print("  python train_rlgym.py")
        return True
    else:
        print(f"✗ Missing dependencies: {', '.join(failed)}")
        print("\nPlease install with:")
        print("  pip install -r requirements.txt")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print(f"Python version: {sys.version}\n")
    
    if sys.version_info < (3, 7):
        print("⚠ Warning: Python 3.7+ required")
        return False
    elif sys.version_info >= (3, 11):
        print("⚠ Warning: RLGym may not work with Python 3.11+")
        print("  Recommend Python 3.9 or 3.10")
        return False
    else:
        print("✓ Python version compatible\n")
        return True

def check_config_files():
    """Check if RLGym config files exist"""
    import os
    
    print("\nChecking RLGym configuration files...")
    
    files = [
        "train_rlgym.py",
        "rlgym_config/rewards.py",
        "rlgym_config/obs_builder.py",
        "rlgym_config/terminal_conditions.py",
        "src/rlgym_bot.py",
    ]
    
    all_exist = True
    for file in files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✓ All configuration files present")
    else:
        print("\n✗ Some configuration files missing")
    
    return all_exist

if __name__ == "__main__":
    print("="*50)
    print("Flamewall RLGym Setup Verification")
    print("="*50 + "\n")
    
    py_ok = check_python_version()
    imports_ok = test_imports()
    config_ok = check_config_files()
    
    print("\n" + "="*50)
    
    if py_ok and imports_ok and config_ok:
        print("\n🎉 Setup complete! Ready to train!\n")
        print("Next steps:")
        print("1. python train_rlgym.py  (start training)")
        print("2. tensorboard --logdir=logs  (monitor progress)")
    else:
        print("\n⚠ Setup incomplete. Please fix the issues above.\n")
