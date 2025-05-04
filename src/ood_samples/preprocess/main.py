import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import get_ood_cases
from case_handler import process_ood_case

def main():
    # Process all OOD cases
    ood_cases = get_ood_cases()
    for case in ood_cases:
        if case != "/home/magata/data/braintumor_data/VIGO_01/original":
            continue
        process_ood_case(case)

if __name__ == "__main__":
    main()
    