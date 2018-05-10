import argparse

def is_prcnt(x):
    '''Assess whether or not the value is a float between 0 and 1'''
    error_msg = 'value must be a float in range [0, 1)'
    try:
        x = float(x)
    except:
        raise argparse.ArgumentTypeError(error_msg)
    if x >= 0 and x < 1:
        return x
    raise argparse.ArgumentTypeError(error_msg)
