import os
import io
import glob
import time
import threading
import picamera.array
import picamera
from PIL import Image
import numpy as np
import zth_car_control
from keras.models import load_model
import tensorflow as tf

"""
模型输出string_angle
string_angle = 90 => 直行
string_angle < 90 => 左转
string_angle > 90 => 右转
"""


# 找到最大的可能性
def get_max_prob_num(predictions_array):
    prediction_edit = np.zeros([1, 5])
    for i in range(0, 5):
        if predictions_array[0][i] == predictions_array.max():
            prediction_edit[0][i] = 1
            return i
    return 2


# 根据神经网络预测的结果来控制小车
def control_car1(string_angle):
    if string_angle < 90:
        print("Left")
        robot.Speed_Wheel_control(2, 2, -2, -2)  # 左旋
        time.sleep(0.25)
    elif string_angle > 90:
        print("Right")
        robot.Speed_Wheel_control(-1, -1, 1, 1)  # 右旋
        time.sleep(0.25)
    elif string_angle == 90:
        print("Forward")
        robot.Speed_Wheel_control(2, 2, 2, 2)  # 所有轮子向前2的速度
    else:
        robot.Speed_Wheel_control(0, 0, 0, 0)  # 所有轮子停止
        print('stop')


def control_car2(string_angle):
    if string_angle < 90:
        print("Left")
        robot.Speed_axis_Yawhold_control(-4, 4)  # 左上
        time.sleep(0.25)
    elif string_angle > 90:
        print("Right")
        robot.Speed_axis_Yawhold_control(4, 4)  # 右上
        time.sleep(0.25)
    elif string_angle == 90:
        print("Forward")
        robot.Speed_axis_Yawhold_control(0, 4)  # 前进
    else:
        robot.Speed_axis_Yawhold_control(0, 0)  # 所有轮子停止
        print('stop')


# 利用神经网络的模型预测图像
# 继承父类threading.Thread
class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    # 要执行的代码写在run函数里面，线程创建后会直接运行run函数
    def run(self):
        global latest_time, model, graph
        while not self.terminated:
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    image = Image.open(self.stream)
                    image_np = np.array(image)
                    camera_data_array = np.expand_dims(image_np, axis=0)
                    current_time = time.time()
                    if current_time > latest_time:
                        if current_time - latest_time > 1:
                            print("")
                        latest_time = current_time
                        with graph.as_default():
                            prediction_array = model.predict(camera_data_array, batch_size=20, verbode=1)
                            # 输出的是概率，比如[0.1,0.1,0.8,0.05,0.04]
                        print(prediction_array)
                        action_num = get_max_prob_num(prediction_array)
                        control_car(action_num)
                finally:
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    with self.owner.lock:
                        self.owner.pool.append(self)


# 多线程处理
class ProcessOutput(object):
    def __init__(self):
        self.done = False
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(4)]
        self.processor = None

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass
            proc.terminated = True
            proc.join()


def main():
    """获取数据，然后预测获得的数据，编辑数据，控制车行驶"""
    global model, graph
    model_loaded = glob.glob('model/*.h5')  # glob.glob()匹配指定的文件
    for single_mod in model_loaded:
        model = load_model(single_mod)
    graph = tf.get_default_graph()

    try:
        with picamera.PiCamera(resolution=(160, 120)) as camera:
            time.sleep(2)
            output = ProcessOutput()
            camera.start_recording(output, format='mjpeg')
            while not output.done:
                camera.wait_recording(1)
            camera.stop_recording()
    finally:
        zth_car_control.clean_GPIO()


if __name__ == '__main__':
    global latest_time
    latest_time = time.time()
    main()
