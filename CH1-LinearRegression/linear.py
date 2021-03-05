import tensorflow as tf
import numpy as np
import pandas as pd

# y = wx+b
def compute_error_for_line_given_points(b, w, points):
    """
    :param b: y=wx+b
    :param w: 斜率
    :param points: 坐标点集
    :return: 函数损失
    """
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w*x + b))**2  # err = [Y-(WX+B)]^2
    return totalError / float(len(points))


def step_gradient(b_cur, w_cur, points, learning_rate):
    """
    :param b_cur:
    :param w_cur:
    :param points:
    :param learningRate:
    :return: 返回经梯度更新后的b和w
    """
    b_grad = 0
    w_grad = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # loss = (WX+B-Y)^2的和
        # b_grad = 2(wx+b-y) loss对b求导
        b_grad += 2 * ((w_cur*x + b_cur) - y) / N
        # w_grad = 2(wx+b-y)*x
        w_grad += 2 * x * ((w_cur*x + b_cur) - y) / N
    # update
    new_b = b_cur - (learning_rate * b_grad)
    new_w = w_cur - (learning_rate * w_grad)
    return new_b, new_w


def grad_des_run(points, start_b, start_w, learning_rate, num_iter):
    """
    :param points:
    :param start_b:
    :param start_w:
    :param learning_rate:
    :param num_iter: 迭代次数
    :return: 经多次迭代得到最终的b和w
    """
    b = start_b
    w = start_w
    for i in range(num_iter):
        b, w = step_gradient(b_cur=b, w_cur=w, points=np.array(points), learning_rate=learning_rate)
    return b, w


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = grad_des_run(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )


if __name__ == '__main__':
    run()