#include <iostream>
#include <vector>

class logistic_regression {
 private:
  int params_num;
  std::vector<double> params;
 public:
  logistic_regression() {}
  void fit(const std::vector<std::vector<double>>& x_train,
           const std::vector<double>& y_train, const double learning_rate,
           const int max_iteration, const double stop_eps,
           const double penalty_coeff);
  double predict(const std::vector<double>& x_test);
};

void logistic_regression::fit(const std::vector<std::vector<double>>& x_train,
                              const std::vector<double>& y_train,
                              const double learning_rate,
                              const int max_iteration, const double stop_eps,
                              const double penalty_coeff) {
  params = std::vector<double>(x_train[0].size() + 1, 0);
}

int main() { return 0; }
