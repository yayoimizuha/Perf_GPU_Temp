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

    random_base = numpy.random.rand(10000, 10000)
    # cpu_inved = numpy.linalg.inv(random_base)
    # print(random_base.tolist())

    with tensorflow.device(device):
        base_tensor = tensorflow.constant(random_base.tolist())

        inved = tensorflow.linalg.inv(base_tensor)

        INVxBASE = tensorflow.matmul(inved, base_tensor)

        Error = tensorflow.subtract(INVxBASE, tensorflow.math.round(INVxBASE))

        print(device, end='\t')
        print(tensorflow.math.reduce_sum(Error))


for i in range(20):
    pass
    run('CPU:0')

print('\n\n\n\n\n\nGPU\n\n\n\n\n\n')


for i in range(20):
    run('GPU:0')
