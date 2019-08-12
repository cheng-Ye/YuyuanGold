import numpy as np
from pso_linear import Pso_LinearRegression
from algo import exponential_smoothing
from data_conf import data_mg_pkl


def pso_linear(x2_t):
    train_x, train_y, test_x = x2_t['train_x'], x2_t['train_y'], x2_t['test_x']
    model = Pso_LinearRegression(fit_intercept=False)
    model.fit(train_x, train_y)
    test_y = model.predict(test_x)
    return test_y

class Control_model():
    def __init__(self, sku, x1_t, x2_t, y_true_t, inv_t):
        self.alpha1, self.alpha2 = 0.1, 0.01
        self.coef = None
        self.k_t, self.w1_t, self.w2_t = None, 0, 0
        self.sku = sku
        self.x1_t, self.x2_t = x1_t, x2_t
        self.inv_t = inv_t
        self.y_pred_t, self.y_true_t = y_true_t, y_true_t

    def init_coef(self):
        return {self.sku: (self.w1_t, self.w2_t, self.y_pred_t)}

    def update_coef(self):
        self.coef.update({self.sku: (self.w1_t, self.w2_t, self.y_pred_t)})
        return self.coef

    def get_sku_coef(self):
        self.coef = data_mg_pkl(self.init_coef, 'Control_coef.pkl')
        if self.coef.get(self.sku):
            self.w1_t, self.w2_t, self.y_pred_t = self.coef[self.sku]

    def adap_coef(self):
        self.get_sku_coef()
        self.w1_t = self.w1_t + self.alpha1 * (-self.inv_t)
        self.w2_t = self.w2_t + self.alpha2 * (self.y_true_t-self.y_pred_t)
        self.k_t = np.tanh(self.w1_t * min(0, self.inv_t) +
                           self.w2_t * abs(self.y_true_t-self.y_pred_t))

    def refer_model(self, x1_t, x2_t):
        beta = 0.5
        y_pred_t1_next = beta * exponential_smoothing(0.8, x1_t) + (1-beta) * pso_linear(x2_t)
        return y_pred_t1_next

    def run_model(self):
        self.adap_coef()
        predict = self.refer_model(self.x1_t, self.x2_t)
        self.y_pred_t = int(predict * (self.k_t+1))
        data_mg_pkl(self.update_coef, 'Control_coef.pkl', force_dump=True)
        return self.y_pred_t, int(predict)
