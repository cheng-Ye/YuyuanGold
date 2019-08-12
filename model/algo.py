# 指数平滑算法
def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型结果， float
    '''
    if len(s) < 1:
        return 0
    s_temp = [0 for i in range(len(s)+1)]
    s_temp[0] = s[0]
    for i in range(1, len(s_temp)):
        s_temp[i] = alpha * s[i-1] + (1 - alpha) * s_temp[i-1]
    return s_temp[-1]


def croston(alpha, s):
    '''
    conston算法
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回conston算法结果， float
    '''
    z = []
    p = []
    zero_nums = 0
    for i in range(len(s)):
        if s[i] == 0:
            zero_nums += 1
            if i == 0:
                z.append(0)
            if i == len(s) - 1:
                p.append(max(zero_nums, exponential_smoothing(alpha, p)))
        else:
            p.append(zero_nums)
            zero_nums = 0
            z.append(s[i])
    if len(p) < len(z):
        p.append(exponential_smoothing(alpha, p))
    else:
        z.append(exponential_smoothing(alpha, z))
    if z.count(0) == len(z):
        return 0
    result = exponential_smoothing(
        alpha, z) / (round(exponential_smoothing(alpha, p), 0)+1)
    if result < 0.2*min([item for item in z if item != 0]):
        result = 0
    #result *= (1-alpha/2)
    #print('p,z, result:   ', p,z, result)
    return result