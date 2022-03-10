import os
import subprocess
import numpy
import tensorflow

temp = subprocess.run(['nvidia-smi ', '--query-gpu=temperature.gpu', '--format=csv,noheader'], stdout=subprocess.PIPE)

print(tensorflow.config.experimental.list_physical_devices("GPU"))
print(tensorflow.config.experimental.list_physical_devices("CPU"))
print(int(temp.stdout.split()[-1]))


def run(device):
    numpy.random.seed(0)

    # random_base = numpy.random.rand(10000, 10000)
    # cpu_inved = numpy.linalg.inv(random_base)
    # print(random_base.tolist())

    with tensorflow.device(device):
        float_a = tensorflow.constant(1 / 7)

        for i in range(100000):
            if i % 1000 == 0:
                print(i, end='\t')
            float_a = tensorflow.divide(float_a, 3 / 7)
            if i % 1000 == 0:
                print(float_a, end='\t')
            float_a = tensorflow.divide(float_a, 7 / 3)
            if i % 1000 == 0:
                print(float_a)


for i in range(20):
    pass
    # run('CPU:0')

print('  GPU')

for i in range(20):
    run('GPU:0')
