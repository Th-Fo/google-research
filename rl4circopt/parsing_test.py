# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for parsing."""


from absl.testing import parameterized

import numpy as np
from scipy import stats

from rl4circopt import circuit
from rl4circopt import parsing


class ParseGateTest(parameterized.TestCase):

  @parameterized.parameters([
      (circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))), circuit.RotZGate),
      (circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]]), circuit.PhasedXGate),
      (circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0])), circuit.ControlledZGate),
      (circuit.RotZGate(0.42), circuit.MatrixGate),
      (circuit.PhasedXGate(0.815, 0.4711), circuit.MatrixGate),
      (circuit.ControlledZGate(), circuit.MatrixGate)
  ])
  def test_positive(self, gate_in, gate_type):
    # call the function to be tested
    gate_out = parsing.parse_gate(gate_in, gate_type)

    # check type of gate_out
    self.assertIsInstance(gate_out, gate_type)

    # check properties of gate_out
    self.assertEqual(gate_out.get_num_qubits(), gate_in.get_num_qubits())
    np.testing.assert_allclose(
        gate_out.get_pauli_transform(),
        gate_in.get_pauli_transform(),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters([
      (circuit.RotZGate(0.42), circuit.RotZGate),
      (circuit.PhasedXGate(0.815, 0.4711), circuit.PhasedXGate),
      (circuit.ControlledZGate(), circuit.ControlledZGate),
      (circuit.MatrixGate(stats.unitary_group.rvs(2)), circuit.MatrixGate),
      (circuit.MatrixGate(stats.unitary_group.rvs(4)), circuit.MatrixGate)
  ])
  def test_identical(self, gate_in, gate_type):
    # call the function to be tested
    gate_out = parsing.parse_gate(gate_in, gate_type)

    # check gate_out
    self.assertIs(gate_out, gate_in)

  @parameterized.parameters([
      (circuit.RotZGate(0.42), circuit.PhasedXGate),
      (circuit.RotZGate(0.42), circuit.ControlledZGate),
      (circuit.PhasedXGate(0.815, 0.4711), circuit.RotZGate),
      (circuit.PhasedXGate(0.815, 0.4711), circuit.ControlledZGate),
      (circuit.ControlledZGate(), circuit.RotZGate),
      (circuit.ControlledZGate(), circuit.PhasedXGate)
  ])
  def test_negative(self, gate, gate_type):
    with self.assertRaises(circuit.GateNotParsableError):
      # call the function to be tested
      parsing.parse_gate(gate, gate_type)

  def test_type_error_gate_not_a_gate(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'gate is not a Gate (found type: float)'):
      parsing.parse_gate(47.11, circuit.MatrixGate)

  def test_type_error_gatetype_no_type(self):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'gate_type is not a type (found: instance of int)'):
      parsing.parse_gate(gate, 42)

  def test_type_error_gatetype_no_gate(self):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'gate_type is not a Gate type (found: str)'):
      parsing.parse_gate(gate, str)

  def test_no_parser_registered_error(self):
    class DummyGate(circuit.Gate):
      pass

    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        parsing.NoParserRegisteredError,
        'no parser registered for gate type DummyGate'):
      parsing.parse_gate(gate, DummyGate)

  def test_runtime_error_parser_raises_unexpected_error_without_message(self):
    class DummyGate(circuit.Gate):
      pass

    def parser(gate):
      raise ValueError

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, parser)
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate unexpectedly raised ValueError'
        ' (without message)'):
      gate_out = parsing.parse_gate(gate, DummyGate)

  def test_runtime_error_parser_raises_unexpected_error_with_message(self):
    class DummyGate(circuit.Gate):
      pass

    def parser(gate):
      raise ValueError('my error message')

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, parser)
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate unexpectedly raised ValueError'
        ' "my error message"'):
      gate_out = parsing.parse_gate(gate, DummyGate)

  def test_runtime_error_parser_returns_none(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, lambda gate: None)
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate did not return anything'):
      gate_out = parsing.parse_gate(gate, DummyGate)

  def test_runtime_error_parser_returns_invalid_type(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, lambda gate: gate)
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate returned an instance'
        ' of type MatrixGate'):
      gate_out = parsing.parse_gate(gate, DummyGate)


class RegisterParserTest(parameterized.TestCase):

  def test_sucessful(self):
    # preparation work: define a simple Gate
    class IdentityGate(circuit.Gate):
      def __init__(self):
        super().__init__(num_qubits=1)
      def get_operator(self):
        return np.eye(2, dtype=complex)

    # preparation work: define a Parser
    class Parser:
      def __init__(self):
        self.num_called = 0
      def __call__(self, gate):
        self.num_called += 1
        if isinstance(gate, IdentityGate):
          return gate
        else:
          raise circuit.GateNotParsableError

    # preparation work: construct the parser and a gate
    parser = Parser()
    gate_in = IdentityGate()

    # call the function to be tested
    parsing.register_parser(IdentityGate, parser)

    # verify that the parser has actually been registered by calling
    # parsing.parse_gate
    gate_out = parsing.parse_gate(gate_in, IdentityGate)

    # check that parsing.parse_gate has returned the expected result
    self.assertIs(gate_out, gate_in)

    # check that the parser has been called exactly once
    self.assertEqual(parser.num_called, 1)

  def test_type_error_gatetype_no_type(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'gate_type is not a type (found: instance of int)'):
      parsing.register_parser(42, lambda gate: gate)

  def test_type_error_gatetype_no_gate(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'gate_type is not a Gate type (found: str)'):
      parsing.register_parser(str, lambda gate: gate)

  def test_type_error_parser_not_callable(self):
    class DummyGate(circuit.Gate):
      pass

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'parser is not callable (found type: float)'):
      parsing.register_parser(DummyGate, 47.11)


if __name__ == '__main__':
  absltest.main()
