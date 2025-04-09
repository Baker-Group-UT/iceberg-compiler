#include "QECCode.hpp"
#include <iostream>

void output_normalizers(QECCode &code) {
  std::vector<PauliGate> normalizer = code.computeNormalizer();
  std::cout << "Normalizers for [" << code.getN() << ", " << code.getK() << ", "
            << code.getD() << "] code (" << normalizer.size() << " gates):\n";
  for (const auto &gate : normalizer) {
    std::cout << "  " << gate << '\n';
  }
}

int main() {
  QECCode Q422(4, 2, 2, {PauliGate(4, Pauli::X), PauliGate(4, Pauli::Z)});
  output_normalizers(Q422);

  QECCode Q613(6, 1, 3,
               {std::vector<Pauli>{Pauli::X, Pauli::I, Pauli::Y, Pauli::Z,
                                   Pauli::X, Pauli::I},
                std::vector<Pauli>{Pauli::Z, Pauli::I, Pauli::X, Pauli::Y,
                                   Pauli::Z, Pauli::I},
                std::vector<Pauli>{Pauli::I, Pauli::X, Pauli::X, Pauli::X,
                                   Pauli::X, Pauli::I},
                std::vector<Pauli>{Pauli::I, Pauli::Z, Pauli::Z, Pauli::Z,
                                   Pauli::Z, Pauli::I},
                std::vector<Pauli>{Pauli::I, Pauli::I, Pauli::I, Pauli::I,
                                   Pauli::I, Pauli::X}});
  output_normalizers(Q613);

  QECCode Q622(6, 2, 2,
               {std::vector<Pauli>{Pauli::X, Pauli::X, Pauli::X, Pauli::X,
                                   Pauli::I, Pauli::I},
                std::vector<Pauli>{Pauli::Z, Pauli::Z, Pauli::Z, Pauli::Z,
                                   Pauli::I, Pauli::I},
                std::vector<Pauli>{Pauli::I, Pauli::I, Pauli::I, Pauli::I,
                                   Pauli::X, Pauli::I},
                std::vector<Pauli>{Pauli::I, Pauli::I, Pauli::I, Pauli::I,
                                   Pauli::I, Pauli::X}});
  output_normalizers(Q622);

  QECCode Q642(6, 4, 2, {PauliGate(6, Pauli::X), PauliGate(6, Pauli::Z)});
  output_normalizers(Q642);

  QECCode Q713(7, 1, 3,
               {std::vector<Pauli>{Pauli::X, Pauli::I, Pauli::I, Pauli::X,
                                   Pauli::I, Pauli::X, Pauli::X},
                std::vector<Pauli>{Pauli::Z, Pauli::I, Pauli::I, Pauli::Z,
                                   Pauli::I, Pauli::Z, Pauli::Z},
                std::vector<Pauli>{Pauli::I, Pauli::X, Pauli::I, Pauli::X,
                                   Pauli::X, Pauli::X, Pauli::I},
                std::vector<Pauli>{Pauli::I, Pauli::Z, Pauli::I, Pauli::Z,
                                   Pauli::Z, Pauli::Z, Pauli::I},
                std::vector<Pauli>{Pauli::I, Pauli::I, Pauli::X, Pauli::I,
                                   Pauli::X, Pauli::X, Pauli::X},
                std::vector<Pauli>{Pauli::I, Pauli::I, Pauli::Z, Pauli::I,
                                   Pauli::Z, Pauli::Z, Pauli::Z}});
  output_normalizers(Q713);

  return 0;
}
