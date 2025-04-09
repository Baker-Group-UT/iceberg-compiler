#ifndef __QECCode_HPP__
#define __QECCode_HPP__
#include "Pauli.hpp"
#include <set>
#include <string>
#include <vector>

class QECCode {
private:
  int n; // number of qubits
  int k; // number of logical qubits
  int d; // distance of the code

  std::vector<PauliGate> stabilizers; // stabilizer generators

public:
  QECCode(int n, int k, int d, const std::vector<PauliGate> &stabilizers)
      : n(n), k(k), d(d), stabilizers(n - k, PauliGate(n)) {
    if (stabilizers.size() != n - k) {
      throw std::invalid_argument("Invalid number of stabilizers");
    }
    for (size_t i = 0; i < n - k; i++) {
      if (stabilizers[i].getN() != n) {
        throw std::invalid_argument("Invalid stabilizer size");
      }
      this->stabilizers[i] = stabilizers[i];
    }

    // check if all stabilizers commute with each other
    for (size_t i = 0; i < n - k; i++) {
      for (size_t j = i + 1; j < n - k; j++) {
        if (!(stabilizers[i] + stabilizers[j]))
          throw std::invalid_argument("Stabilizers " + std::to_string(i) +
                                      " and " + std::to_string(j) +
                                      " do not commute");
      }
    }
  }

  int getN() const { return n; }
  int getK() const { return k; }
  int getD() const { return d; }
  const std::vector<PauliGate> &getStabilizers() const { return stabilizers; }

  /*
   * Compute the normalizer of the stabilizer group
   * @return the normalizer of the stabilizer group
   */
  std::vector<PauliGate> computeNormalizer() const {
    std::vector<PauliGate> normalizer;
    normalizer.push_back(PauliGate(n, Pauli::I));

    std::set<PauliGate> normalizer_group;
    for (unsigned long long i = 0; i < (1ULL << (n - k)); i++) {
      PauliGate group_gate(n);
      for (int j = 0; j < n - k; j++)
        if (i & (1ULL << j))
          group_gate *= stabilizers[j];
      normalizer_group.insert(group_gate);
    }

    for (unsigned long long i = 0; i < (1ULL << (2 * n)); i++) {
      PauliGate gate(n);
      for (int j = 0; j < n; j++) {
        if (i & (1ULL << (2 * j)))
          gate[j] *= Pauli::X;
        if (i & (1ULL << (2 * j + 1)))
          gate[j] *= Pauli::Z;
      }
      if (normalizer_group.count(gate))
        continue;
      bool commutes = true;
      for (const auto &stabilizer : stabilizers) {
        if (!(gate + stabilizer)) {
          commutes = false;
          break;
        }
      }
      if (!commutes)
        continue;

      for (auto it = normalizer_group.begin(); it != normalizer_group.end();
           it++)
        normalizer_group.insert(gate * (*it));
      normalizer.push_back(gate);
    }
    return normalizer;
  }
};

#endif // !__QECCode_HPP__
