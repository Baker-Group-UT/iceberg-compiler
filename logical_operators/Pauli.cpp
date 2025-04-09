#include "Pauli.hpp"

std::ostream &operator<<(std::ostream &os, const Pauli p) {
  char output;
  if (p == Pauli::I)
    output = 'I';
  else if (p == Pauli::Z)
    output = 'Z';
  else if (p == Pauli::X)
    output = 'X';
  else if (p == Pauli::Y)
    output = 'Y';
  os << output;
  return os;
}

Pauli &operator*=(Pauli &self, const Pauli &other) {
  if (self == other)
    self = Pauli::I;
  else {
    self = static_cast<Pauli>(
        (static_cast<int>(self) + static_cast<int>(other)) % 4);
    if (self == Pauli::I)
      self = Pauli::X;
  }
  return self;
}

Pauli operator*(Pauli lhs, const Pauli &rhs) {
  lhs *= rhs;
  return lhs;
}

PauliGate::PauliGate(std::vector<Pauli> paulis)
    : n(paulis.size()), paulis(paulis) {
  if (paulis.size() == 0) {
    throw std::invalid_argument("PauliGate must have at least one qubit");
  }
}

PauliGate::PauliGate(int n) : PauliGate(n, Pauli::I) {}

PauliGate::PauliGate(int n, Pauli pauli)
    : PauliGate(std::vector<Pauli>(n, pauli)) {}

int PauliGate::getN() const { return n; }

PauliGate &PauliGate::operator*=(const PauliGate &other) {
  if (n != other.n) {
    throw std::invalid_argument(
        "PauliGates must have the same number of qubits");
  }
  for (int i = 0; i < n; i++) {
    if (paulis[i] != Pauli::I && other.paulis[i] != Pauli::I &&
        static_cast<int>(paulis[i]) ==
            (static_cast<int>(other.paulis[i]) + 1) % 4)
      globalPhase ^= 1;
    paulis[i] *= other.paulis[i];
  }
  return *this;
}

PauliGate operator*(PauliGate lhs, const PauliGate &rhs) {
  lhs *= rhs;
  return lhs;
}

bool PauliGate::operator+(const PauliGate &other) const {
  if (n != other.n) {
    throw std::invalid_argument(
        "PauliGates must have the same number of qubits");
  }
  bool commute = true;
  for (int i = 0; i < n; i++)
    if (paulis[i] != Pauli::I && other.paulis[i] != Pauli::I &&
        paulis[i] != other.paulis[i])
      commute ^= 1;
  return commute;
}

bool PauliGate::operator<(const PauliGate &other) const {
  if (n != other.n) {
    throw std::invalid_argument(
        "PauliGates must have the same number of qubits");
  }
  for (int i = 0; i < n; i++) {
    if (paulis[i] < other.paulis[i])
      return true;
    else if (paulis[i] > other.paulis[i])
      return false;
  }
  return false;
}

Pauli &PauliGate::operator[](int i) {
  if (i < 0 || i >= n) {
    throw std::out_of_range("Index out of range");
  }
  return paulis[i];
}

const Pauli &PauliGate::operator[](int i) const {
  if (i < 0 || i >= n) {
    throw std::out_of_range("Index out of range");
  }
  return paulis[i];
}

std::ostream &operator<<(std::ostream &os, const PauliGate &pauliGate) {
  os << '[';
  for (size_t i = 0; i < pauliGate.getN(); i++)
    os << pauliGate[i] << (i != pauliGate.getN() - 1 ? ", " : "]");
  return os;
}
