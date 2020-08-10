#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class logistic_regression {
 private:
  int params_num;
  std::vector<double> params, error, epoch;

 public:
  logistic_regression() {}
  void train_one_data(const std::vector<double>& x, const double y,
                      std::vector<double>& params_grad, const int params_num);
  void fit(const std::vector<std::vector<double>>& x_train,
           const std::vector<double>& y_train, const int batch_size,
           const double learning_rate, const int max_iteration,
           const double penalty_coeff);
  // linear_prediction returns w0+w1*x1+w2*x2+\ldots wn*xn
  double linear_prediction(const std::vector<double>& x);
  // predict_proba returns the probability of predicted y is 0
  // which is estimated by 1/(1+exp(w0+w1*x1+w2*x2+\ldots +wn*xn))
  double predict_proba(const std::vector<double>& x);
  double log_loss(const std::vector<double>& x, const double y);
  double mean_log_loss(const std::vector<std::vector<double>>& x,
                       const std::vector<double>& y);
  void write_csv(const std::string& filename);
};
void logistic_regression::train_one_data(const std::vector<double>& x,
                                         const double y,
                                         std::vector<double>& params_grad,
                                         const int params_num) {
  const double exp_xw = exp(linear_prediction(x));
  const double train_coeff = y - exp_xw / (1 + exp_xw);
  // intercept
  params_grad[0] += train_coeff;
  // coefficient
  for (int k = 1; k < params_num; k++) {
    params_grad[k] += train_coeff * x[k - 1];
  }
}
void logistic_regression::fit(const std::vector<std::vector<double>>& x_train,
                              const std::vector<double>& y_train,
                              const int batch_size, const double learning_rate,
                              const int max_iteration,
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
  std::random_device rnd;
  std::mt19937 engine(rnd());
  // current_order is the number of data used in data_order
  int current_order = data_num, next_order;
  // initialize epoch and error record
  epoch.clear();
  error.clear();
  // record initial epoch and error
  epoch.push_back(0);
  error.push_back(mean_log_loss(x_train, y_train));
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
    // calculate gradient of log likelihood with L2 regularization
    for (int k = 0; k < params_num; k++) {
      params_grad[k] =
          params_grad[k] / (double)batch_size - penalty_coeff * params[k];
    }
    // update parameters of logistic regression
    for (int k = 0; k < params_num; k++) {
      params[k] = params[k] + learning_rate * params_grad[k];
    }
    current_order = next_order;
    // record epoch and error
    epoch.push_back(epoch.back() + (double)batch_size / (data_num));
    const double logloss = mean_log_loss(x_train, y_train);
    error.push_back(logloss);
    std::cout << i << " iteration "
              << " error=" << logloss << std::endl;
  }
  std::cout << "parameter: " << std::endl;
  for (int i = 0; i < params_num; i++) {
    std::cout << i << ": " << params[i] << std::endl;
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
                                     const double y) {
  if (y == 0) {
    return -log(predict_proba(x));
  } else {
    return -log(1.0 - predict_proba(x));
  }
}
double logistic_regression::mean_log_loss(
    const std::vector<std::vector<double>>& x, const std::vector<double>& y) {
  double res = 0;
  for (int i = 0; i < y.size(); i++) {
    res += log_loss(x[i], y[i]);
  }
  return res / (double)y.size();
}

void logistic_regression::write_csv(const std::string& filename) {
  std::ofstream ofs(filename + ".csv");
  if (!ofs) {
    std::cerr << "file open failed" << std::endl;
    exit(true);
  }
  ofs << "iteration,epoch,loss" << std::endl;
  for (int i = 0; i < epoch.size(); i++) {
    ofs << i << "," << epoch[i] << "," << error[i] << std::endl;
  }
  ofs.close();
}

int main() {
  std::random_device rnd;
  std::mt19937 engine(rnd());
  std::uniform_real_distribution<> dist(-1, 1);

  const int data_num = 10000;
  const int params_num = 4;
  // data
  std::vector<std::vector<double>> x_train(data_num, std::vector<double>(4));
  std::vector<double> y_train(data_num);
  // generate 10000 data of x_train and y_train
  for (int i = 0; i < 10000; i++) {
    double y_linear = 0.1;
    for (int j = 0; j < 4; j++) {
      x_train[i][j] = dist(engine);
      y_linear += (double)(j + 1) * x_train[i][j];
    }
    const double y_proba = 1 / (1 + exp(y_linear));
    y_train[i] = y_proba > 0.5 ? 0 : 1;
  }
  logistic_regression lr;
  lr.fit(x_train, y_train, 1, 0.001, 50000, 0.1);
  lr.write_csv("batch_1_");
  lr.fit(x_train, y_train, 10, 0.001, 50000, 0.1);
  lr.write_csv("batch_10_");
  lr.fit(x_train, y_train, 100, 0.001, 50000, 0.1);
  lr.write_csv("batch_100_");
  lr.fit(x_train, y_train, 1000, 0.001, 50000, 0.1);
  lr.write_csv("batch_1000_");
  lr.fit(x_train, y_train, 10000, 0.001, 50000, 0.1);
  lr.write_csv("batch_10000_");
  return 0;
}
