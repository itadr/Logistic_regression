#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include<random>

class logistic_regression {
 private:
  int params_num;
  std::vector<double> params, error;

 public:
  logistic_regression() {}
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
  params = std::vector<double>(params_num, 0);
  // params_grad is used to calculate gradient of los function in each iteration
  std::vector<double> params_grad(params_num, 0);
  // data_order is used to shuffle train data for SGD
  std::vector<int> data_order(data_num);
  // current_order is the number of data used in data_order
  int current_order=0;
  for(int i=0;i<max_iteration;i++){

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
