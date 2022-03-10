import os
import subprocess
import numpy
import tensorflow

temp = subprocess.run(['nvidia-smi ', '--query-gpu=temperature.gpu', '--format=csv,noheader'], stdout=subprocess.PIPE)

print(int(temp.stdout.split()[-1]))

numpy.random.seed(0)


def run():
    random_base = numpy.random.rand(10000, 10000)
    cpu_inved = numpy.linalg.inv(random_base)
    # print(random_base.tolist())

    print(tensorflow.config.experimental.list_physical_devices("GPU"))

    base_tensor = tensorflow.constant(random_base.tolist())

    inved = tensorflow.linalg.inv(base_tensor)

    INVxBASE = tensorflow.matmul(inved, base_tensor)

    Error = tensorflow.subtract(INVxBASE, tensorflow.math.round(INVxBASE)).numpy()


    print(Error.sum())


for i in range(20):
    tensorflow.device('CPU:0')
    run()

print('\n\n\n\n\n\nGPU\n\n\n\n\n\n')


for i in range(20):
    tensorflow.device('/physical_device:GPU:0')
    run()
