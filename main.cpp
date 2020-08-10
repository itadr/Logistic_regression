#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

class logistic_regression {
 private:
  int params_num;
  std::vector<double> params, error;

 public:
  logistic_regression() {}
  void train_one_data(const std::vector<double>& x, const double y,
                      std::vector<double>& params_grad, const int params_num);
  void fit(const std::vector<std::vector<double>>& x_train,
           const std::vector<double>& y_train, const int batch_size,
           const double learning_rate, const int max_iteration,
           const double stop_eps, const double penalty_coeff);
  // linear_prediction returns w0+w1*x1+w2*x2+\ldots wn*xn
  double linear_prediction(const std::vector<double>& x);
  // predict_proba returns the probability of predicted y is 0
  // which is estimated by 1/(1+exp(w0+w1*x1+w2*x2+\ldots +wn*xn))
  double predict_proba(const std::vector<double>& x);
  double log_loss(const std::vector<double>& x, const double y);
};
void logistic_regression::train_one_data(const std::vector<double>& x,
                                         const double y,
                                         std::vector<double>& params_grad,
                                         const int params_num) {
  const double exp_xw = exp(linear_prediction(x));
  const double train_coeff = y - exp_xw / (1 + exp_xw);
  for (int k = 0; k < params_num; k++) {
    params_grad[k] += train_coeff * x[k];
  }
}
void logistic_regression::fit(const std::vector<std::vector<double>>& x_train,
                              const std::vector<double>& y_train,
                              const int batch_size, const double learning_rate,
                              const int max_iteration, const double stop_eps,
                              const double penalty_coeff) {
  // params incluedes constant term of logistic regression
  params_num = x_train[0].size() + 1;
  // data_num is the number of train data
  const int data_num = x_train.size();
  // initialize all parameters to 0
  params = std::vector<double>(params_num);
  // params_grad is used to calculate gradient of los function in each iteration
  std::vector<double> params_grad(params_num, 0);
  // data_order is used to shuffle train data for SGD
  std::vector<int> data_order(data_num);
  std::random_device rnd;
  std::mt19937 engine(rnd());
  // current_order is the number of data used in data_order
  int current_order = data_num, next_order;

  for (int i = 0; i < max_iteration; i++) {
    // initialize params grad to 0
    fill(params_grad.begin(), params_grad.end(), 0);
    // decide which data is used in this iteration
    next_order = current_order + batch_size;
    // if shuffling data is necessary
    if (next_order >= data_num) {
      for (int j = current_order; j < data_num; j++) {
        train_one_data(x_train[j], y_train[j], params_grad, params_num);
      }
      // shuffle data
      std::shuffle(data_order.begin(), data_order.end(), engine);
      next_order -= data_num;
      for (int j = 0; j < next_order; j++) {
        train_one_data(x_train[j], y_train[j], params_grad, params_num);
      }
    }
    // if shuffling data is not necessary
    else {
      for (int j = current_order; j < next_order; j++) {
        train_one_data(x_train[j], y_train[j], params_grad, params_num);
      }
    }
    for (int k = 0; k < params_num; k++) {
      params_grad[k] =
          params_grad[k] / (double)batch_size - penalty_coeff * params[k];
    }
    // update parameters of logistic regression
    for (int k = 0; k < params_num; k++) {
      params[k] = params[k] + learning_rate * params_grad[k];
    }
    current_order = next_order;
  }
}
double logistic_regression::linear_prediction(const std::vector<double>& x) {
  // constant term
  double res = params[0];
  for (int i = 0; i < x.size(); i++) {
    res += params[i + 1] * x[i];
  }
  return res;
}
double logistic_regression::predict_proba(const std::vector<double>& x) {
  return 1 / (1 + exp(linear_prediction(x)));
}

double logistic_regression::log_loss(const std::vector<double>& x,
                                     const double y) {}

int main() { return 0; }
