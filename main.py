"""Root Entry Point for SAL_tech"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    from sal_tech_core.main import main
    main()
