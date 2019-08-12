import numpy as np
from metrics import cal_mape

class Pso_LinearRegression():
    def __init__(self, fit_intercept=True):
        self.X = None
        self.y = None
        self.W = None
        self.number = 0
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.X = np.array(X)
        if self.fit_intercept:
            self.X = np.column_stack((self.X, np.ones(self.X.shape[0])))
        self.y = np.array(y)
        self.number = self.X.shape[1]
        _, gbestpop = self.PSO(self.func, lr1=1, lr2=1, maxgen=300, sizepop=1000, number=self.number, minpop=0.1, maxpop=0.8, minspeed=-0.5, maxspeed=0.5, wmax=0.8, wmin=0.6)
        self.W = np.array(gbestpop)

    def predict(self, X):
        if self.fit_intercept:
            X = np.column_stack((X, np.ones(X.shape[0])))
        pred_y = (self.W * X).sum(axis=1)
        #pred_y = np.ceil(pred_y)
        return pred_y

    def func(self, W):
        pred_y = (W*self.X).sum(axis=1)
        #pred_y = np.ceil(pred_y)
        mape = cal_mape(pred_y, self.y)
        return -mape

    def PSO(self, func, lr1, lr2, maxgen, sizepop, number, minpop, maxpop, minspeed, maxspeed, wmax, wmin):
        '''
        :param d: 测试的天数
        :param func: 调用的函数
        :param lr1:个体学习因子
        :param lr2:社会学习因子
        :param maxgen:最大迭代次数
        :param sizepop:粒子的个数
        :param number:变量个数
        :param minpop:开始粒子位置的最小值
        :param maxpop:开始粒子位置的最大值
        :param minspeed:粒子速度的最小值
        :param maxspeed:粒子速度的最大值
        :param wmax:权重的最大值
        :param wmin:权重的最小值（随着步长的逐渐变化）
        :return:最佳的位置及适应度函数值
        '''

        lr = (lr1, lr2)
        rangepop = (minpop, maxpop)
        rangespeed = (minspeed, maxspeed)

        # 初始化粒子的位置、速度、适应度的值
        def initpopvfit(sizepop, number):
            pop = np.zeros((sizepop, number))
            v = np.zeros((sizepop, number))
            fitness = np.zeros(sizepop)
            for i in range(sizepop):
                for j in range(number):
                    pop[i, j] = (rangepop[1]-rangepop[0]) * np.random.rand()+rangepop[0]
                    v[i, j] = (rangespeed[1]-rangespeed[0]) * np.random.rand()+rangespeed[0]
                #print(i, pop[i])
                fitness[i] = self.func(pop[i])
            #print (pop[1])
            return pop, v, fitness

        def getinitbest(fitness, pop):
            # 群体最优的粒子位置及其适应度值
            gbestpop, gbestfitness = pop[fitness.argmax()].copy(), fitness.max()
            # 个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似
            pbestpop, pbestfitness = pop.copy(), fitness.copy()
            return gbestpop, gbestfitness, pbestpop, pbestfitness

        result = np.zeros(maxgen)
        pop, v, fitness = initpopvfit(sizepop, number)
        gbestpop, gbestfitness, pbestpop, pbestfitness = getinitbest(fitness, pop)
        #wmax = 0.8
        #wmin = 0.6

        for i in range(maxgen):
            fvag = sum(fitness)/sizepop
            fmin = min(fitness)
            # 速度更新
            for j in range(sizepop):
                if fitness[j] < fvag:
                    weight = wmin + (fitness[j]-fmin)*(wmax-wmin)/(fvag-fmin)
                else:
                    weight = wmax

                v[j] = weight*v[j] + lr[0]*np.random.rand()*(pbestpop[j]-pop[j]) + lr[1]*np.random.rand()*(gbestpop-pop[j])
            v[v<rangespeed[0]] = rangespeed[0]
            v[v>rangespeed[1]] = rangespeed[1]

            # 粒子位置更新(为什么根据速度要乘以系数0.5)
            for j in range(sizepop):
                pop[j] += v[j]
            pop[pop<rangepop[0]] = rangepop[0]
            pop[pop>rangepop[1]] = rangepop[1]

            # 适应度更新
            for j in range(sizepop):
                fitness[j] = self.func(pop[j])

            for j in range(sizepop):
                if fitness[j] > pbestfitness[j]:
                    pbestfitness[j] = fitness[j]
                    pbestpop[j] = pop[j].copy()

            if pbestfitness.max() > gbestfitness:
                gbestfitness = pbestfitness.max()
                gbestpop = pop[pbestfitness.argmax()].copy()

            result[i] = gbestfitness
            # print(gbestpop)
            # print(gbestfitness)
        return gbestfitness, gbestpop


if __name__ == "__main__":
    model = Pso_LinearRegression(fit_intercept=True)
    model.fit([[1, 2], [3, 4]], [1, 2])
    print(model.W)
