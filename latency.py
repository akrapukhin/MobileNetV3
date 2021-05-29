import numpy as np
import os
import torch
from timeit import default_timer as timer
from models import *

models = [
    Mobilenet_v3_small(0.25), 
    Mobilenet_v3_small(0.5), 
    Mobilenet_v3_small(1.0),  
    Mobilenet_v3_large(0.25),  
    Mobilenet_v3_large(0.5),   
    Mobilenet_v3_large(1.0) 
]

#GPU inference time measurements
if torch.cuda.is_available():
    print("GPU inference time measurements (means and standard deviations of 300 measurements):")
    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float).cuda()
    
    for model in models:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = []
        model = model.cuda()
        model.eval()

        #GPU warm up
        for _ in range(10):
            _ = model(dummy_input)
    
        #measurements
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)

        timings_mean = np.mean(timings)
        timings_std = np.std(timings)
        print("model:", model.name())
        print('mean: {:.2f} ms'.format(timings_mean))
        print('std:  {:.2f} ms'.format(timings_std))
        print("")
else:
    print("No GPU measurements as CUDA is not available")

#CPU inference time measurements
print("CPU inference time measurements (means and standard deviations of 300 measurements):")
dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float).to('cpu')

for model in models:
    repetitions = 300
    timings = []
    model = model.to('cpu')
    model.eval()
    
    #measurements
    with torch.no_grad():
        for rep in range(repetitions):
            start_time = timer()
            _ = model(dummy_input)
            end_time = timer()
            curr_time = (end_time - start_time) * 1000.0
            timings.append(curr_time)

    timings_mean = np.mean(timings)
    timings_std = np.std(timings)
    print("model:", model.name())
    print('mean: {:.2f} ms'.format(timings_mean))
    print('std:  {:.2f} ms'.format(timings_std))
    print("")
