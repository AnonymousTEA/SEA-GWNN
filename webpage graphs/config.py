import numpy as np

def configure(dataset_str):
    if dataset_str == 'wisconsin':
        params={"lr_rate_mul":0.7,"alpha":0.4,"lambda":0.1,"gamma":0.2,"wd1":0.05,"wd2":0.0001,"dropout":0.3,"layers":4,"hidden":64,"myf":1.2}
        return params
    if dataset_str == 'cornell':
        params={"lr_rate_mul":0.7,"alpha":0.4,"lambda":0.1,"gamma":0.2,"wd1":0.05,"wd2":0.0001,"dropout":0.3,"layers":4,"hidden":64,"myf":1.2}
        return params
    if dataset_str == 'texas':
        params={"lr_rate_mul":0.7,"alpha":0.4,"lambda":0.1,"gamma":0.2,"wd1":0.05,"wd2":0.0001,"dropout":0.3,"layers":4,"hidden":128,"myf":1.2} 
        return params