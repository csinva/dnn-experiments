import pandas as pd
import numpy as np

def process_results(results):
    # add keys for things which weren't recorded at the teim
    for key in ['H_trace']:
        if key not in results:
            results[key] = 0
    results['H_trace'] = 0
    return results