import torch
import torchvision
import tqdm
import thop


print("Starting Test")

for i in range(1000000):
    if  i%1000 == 0:
        print("step :%d completed" %i)

print("Test Ended")

