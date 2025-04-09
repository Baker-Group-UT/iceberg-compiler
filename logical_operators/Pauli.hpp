#ifndef __Pauli_HPP__
#define __Pauli_HPP__

#include <ostream>
#include <vector>

enum class Pauli { I = 0, Z, X, Y };
std::ostream &operator<<(std::ostream &os, const Pauli p);
Pauli &operator*=(Pauli &self, const Pauli &other);
Pauli operator*(Pauli left, const Pauli &right);

class PauliGate {
private:
  int n;                     // number of qubits
  std::vector<Pauli> paulis; // Pauli operators
  bool globalPhase = false;  // global phase flag

public:
  /*
   * Constructor for PauliGate
   * @param n number of qubits
   * @param paulis vector of Pauli operators
   * @throws std::invalid_argument if n is less than 1
   */
  PauliGate(std::vector<Pauli> paulis);
  PauliGate(int n);
  PauliGate(int n, Pauli pauli);

  /*
   * Get the number of qubits
   * @return the number of qubits
   */
  int getN() const;

  /*
   * Multiply this PauliGate with another PauliGate and assign the result
   * @param other the other PauliGate
   * @throws std::invalid_argument if the number of qubits is not the same
   * @return this the updated PauliGate
   */
  PauliGate &operator*=(const PauliGate &other);

  /*
   * Multiply lhs PauliGate with rhs PauliGate
   * @param lhs the first PauliGate
   * @param rhs the second PauliGate
   * @throws std::invalid_argument if the number of qubits is not the same
   * @return the result of the multiplication
   */
  friend PauliGate operator*(PauliGate lhs, const PauliGate &rhs);

  /*
   * Check if two PauliGates commute
   * @param other the other PauliGate
   * @throws std::invalid_argument if the number of qubits is not the same
   * @return true if the two PauliGates commute, false otherwise
   */
  bool operator+(const PauliGate &other) const;

  /*
   * Compare two PauliGates
   * @param other the other PauliGate
   * @return true if this < other, false otherwise
   */
  bool operator<(const PauliGate &other) const;

  /*
   * Get the i-th Pauli operator
   * @param i index of the Pauli operator
   * @throws std::out_of_range if i is out of range
   * @return the i-th Pauli operator
   */
  Pauli &operator[](int i);

  /*
   * Get the i-th Pauli operator
   * @param i index of the Pauli operator
   * @throws std::out_of_range if i is out of range
   * @return the i-th Pauli operator
   */
  const Pauli &operator[](int i) const;
};

std::ostream &operator<<(std::ostream &os, const PauliGate &pauliGate);

#endif // !__Pauli_HPP__
