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


class CheckOperationsTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          [circuit.Operation(circuit.RotZGate(0.42), [42])],
          [circuit.RotZGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
                  [42]
              )
          ],
          [circuit.RotZGate]
      ),
      (
          [circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42])],
          [circuit.PhasedXGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]]),
                  [42]
              )
          ],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.Operation(circuit.ControlledZGate(), [47, 11])],
          [circuit.ControlledZGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0])),
                  [47, 11]
              )
          ],
          [circuit.ControlledZGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
                  [42]
              ),
              circuit.Operation(
                  circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]]),
                  [37]
              ),
              circuit.Operation(
                  circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0])),
                  [47, 11]
              )
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37]),
              circuit.Operation(circuit.ControlledZGate(), [47, 11])
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37]),
              circuit.Operation(circuit.ControlledZGate(), [47, 11])
          ],
          [circuit.MatrixGate, circuit.MatrixGate, circuit.MatrixGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate(stats.unitary_group.rvs(2)),
                  [42]
              ),
              circuit.Operation(
                  circuit.MatrixGate(stats.unitary_group.rvs(4)),
                  [47, 11]
              )
          ],
          [circuit.MatrixGate, circuit.MatrixGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
                  [42]
              ),
              circuit.Operation(
                  circuit.PhasedXGate(0.815, 0.4711),
                  [37]
              ),
          ],
          [circuit.RotZGate, circuit.PhasedXGate]
      )
  )
  def test_positive(self, operations, gate_types):
    # call the function to be tested
    is_parsable = parsing.check_operations(operations, *gate_types)

    # check type and value of is_parsable
    self.assertIs(type(is_parsable), bool)
    self.assertTrue(is_parsable)

  @parameterized.parameters(
      (
          [circuit.Operation(circuit.RotZGate(0.42), [42])],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.Operation(circuit.RotZGate(0.42), [42])],
          [circuit.ControlledZGate]
      ),
      (
          [circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42])],
          [circuit.RotZGate]
      ),
      (
          [circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42])],
          [circuit.ControlledZGate]
      ),
      (
          [circuit.Operation(circuit.ControlledZGate(), [47, 11])],
          [circuit.RotZGate]
      ),
      (
          [circuit.Operation(circuit.ControlledZGate(), [47, 11])],
          [circuit.PhasedXGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.PhasedXGate, circuit.PhasedXGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.PhasedXGate, circuit.MatrixGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.RotZGate, circuit.RotZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.MatrixGate, circuit.RotZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.PhasedXGate, circuit.RotZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37]),
              circuit.Operation(circuit.ControlledZGate(), [47, 11])
          ],
          [circuit.PhasedXGate, circuit.RotZGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37]),
              circuit.Operation(circuit.ControlledZGate(), [47, 11])
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.RotZGate]
      )
  )
  def test_negative(self, operations, gate_types):
    # call the function to be tested
    is_parsable = parsing.check_operations(operations, *gate_types)

    # check type and value of is_parsable
    self.assertIs(type(is_parsable), bool)
    self.assertFalse(is_parsable)

  def test_type_error_operations_no_sequence(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'operations is not a sequence (found type: bool)'):
      parsing.check_operations(False, circuit.MatrixGate, circuit.MatrixGate)

  def test_type_error_operation_not_an_operation(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all operations are instances of Operation'
        ' [found: instance(s) of float]'):
      parsing.check_operations(
          [operation, 47.11],
          circuit.MatrixGate, circuit.MatrixGate
      )

  def test_type_error_gatetype_no_type(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gate_types are types [found: instance(s) of int]'):
      parsing.check_operations(
          [operation, operation],
          circuit.MatrixGate, 42
      )

  def test_type_error_gatetype_no_gate(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gate_types are subtypes of Gate [found types: str]'):
      parsing.check_operations(
          [operation, operation],
          circuit.MatrixGate, str
      )

  def test_value_error_too_many_operations(self):
    operations = [
        circuit.Operation(
            circuit.MatrixGate(stats.unitary_group.rvs(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(stats.unitary_group.rvs(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'inconsistent length of operations and gate_types (2 vs 1)'):
      parsing.check_operations(operations, circuit.MatrixGate)

  def test_value_error_too_many_types(self):
    operations = [
        circuit.Operation(
            circuit.MatrixGate(stats.unitary_group.rvs(2)),
            [42]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'inconsistent length of operations and gate_types (1 vs 2)'):
      parsing.check_operations(
          operations,
          circuit.MatrixGate, circuit.MatrixGate
      )

  def test_no_parser_registered_error(self):
    class DummyGate(circuit.Gate):
      pass

    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        parsing.NoParserRegisteredError,
        'no parser registered for gate type DummyGate'):
      parsing.check_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_raises_unexpected_error_without_message(self):
    class DummyGate(circuit.Gate):
      pass

    def parser(gate):
      raise ValueError

    # preparation work: register the (erroneous) parser and construct operations
    parsing.register_parser(DummyGate, parser)
    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate unexpectedly raised ValueError'
        ' (without message)'):
      gate_out = parsing.check_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_raises_unexpected_error_with_message(self):
    class DummyGate(circuit.Gate):
      pass

    def parser(gate):
      raise ValueError('my error message')

    # preparation work: register the (erroneous) parser and construct operations
    parsing.register_parser(DummyGate, parser)
    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate unexpectedly raised ValueError'
        ' "my error message"'):
      gate_out = parsing.check_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_returns_none(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct operations
    parsing.register_parser(DummyGate, lambda gate: None)
    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate did not return anything'):
      gate_out = parsing.check_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_returns_invalid_type(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct operations
    parsing.register_parser(DummyGate, lambda gate: gate)
    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate returned an instance'
        ' of type MatrixGate'):
      gate_out = parsing.check_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )


class ParseOperationsTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          [
              circuit.Operation(
                  circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
                  [42]
              )
          ],
          [circuit.RotZGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]]),
                  [42]
              )
          ],
          [circuit.PhasedXGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0])),
                  [47, 11]
              )
          ],
          [circuit.ControlledZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37]),
              circuit.Operation(circuit.ControlledZGate(), [47, 11])
          ],
          [circuit.MatrixGate, circuit.MatrixGate, circuit.MatrixGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
                  [42]
              ),
              circuit.Operation(
                  circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]]),
                  [37]
              ),
              circuit.Operation(
                  circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0])),
                  [47, 11]
              )
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.ControlledZGate]
      )
  )
  def test_positive(self, operations_in, gate_types):
    # call the function to be tested
    operations_out = parsing.parse_operations(operations_in, *gate_types)

    # check type of operations_out
    self.assertIs(type(operations_out), list)

    # check length of operations_out
    self.assertLen(operations_out, len(operations_in))

    # check types of operations_out elements
    self.assertTrue(all(
        type(operation) is circuit.Operation
        for operation in operations_out
    ))

    # extract gates of operations_out
    gates_out = [operation.get_gate() for operation in operations_out]

    # check types of gates_out elements
    self.assertTrue(all(
        isinstance(gate_out, gate_type)
        for gate_out, gate_type in zip(gates_out, gate_types)
    ))

    # check properties of operations_out/gates_out elements
    self.assertTrue(all(
        operation_out.get_num_qubits() == operation_in.get_num_qubits()
        for operation_out, operation_in in zip(operations_out, operations_in)
    ))
    self.assertTrue(all(
        operation_out.get_qubits() == operation_in.get_qubits()
        for operation_out, operation_in in zip(operations_out, operations_in)
    ))
    self.assertTrue(all(
        np.allclose(
            gate_out.get_pauli_transform(),
            operations_in.get_gate().get_pauli_transform()
        )
        for gate_out, operations_in in zip(gates_out, operations_in)
    ))

  @parameterized.parameters(
      (
          [circuit.Operation(circuit.RotZGate(0.42), [42])],
          [circuit.RotZGate]
      ),
      (
          [circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42])],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.Operation(circuit.ControlledZGate(), [47, 11])],
          [circuit.ControlledZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37]),
              circuit.Operation(circuit.ControlledZGate(), [47, 11])
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.Operation(
                  circuit.MatrixGate(stats.unitary_group.rvs(2)),
                  [42]
              ),
              circuit.Operation(
                  circuit.MatrixGate(stats.unitary_group.rvs(4)),
                  [47, 11]
              )
          ],
          [circuit.MatrixGate, circuit.MatrixGate]
      )
  )
  def test_identical(self, operations_in, gate_types):
    # call the function to be tested
    operations_out = parsing.parse_operations(operations_in, *gate_types)

    # check type of operations_out
    self.assertIs(type(operations_out), list)

    # check length of operations_out
    self.assertLen(operations_out, len(operations_in))

    # check elements of operations_out
    self.assertTrue(all(
        operation_out is operation_out
        for operation_out, operation_out in zip(operations_out, operations_in)
    ))

  def test_mixed(self):
    # preparation work: construct two operations
    operation_a = circuit.Operation(
        circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
        [42]
    )
    operation_b = circuit.Operation(
        circuit.PhasedXGate(0.815, 0.4711),
        [37]
    )

    # call the function to be tested
    operations_out = parsing.parse_operations(
        [operation_a, operation_b],
        circuit.RotZGate, circuit.PhasedXGate
    )

    # check type of operations_out
    self.assertIs(type(operations_out), list)

    # check length of gate_out
    self.assertLen(operations_out, 2)

    # check type of operations_out[0]
    self.assertIs(type(operations_out[0]), circuit.Operation)

    # extract gate of operations_out[0]
    gate_out_a = operations_out[0].get_gate()

    # check properties of operations_out[0]
    self.assertEqual(
        operations_out[0].get_num_qubits(),
        operation_a.get_num_qubits()
    )
    self.assertEqual(operations_out[0].get_qubits(), operation_a.get_qubits())
    self.assertIsInstance(gate_out_a, circuit.RotZGate)
    np.testing.assert_allclose(
        gate_out_a.get_pauli_transform(),
        operation_a.get_gate().get_pauli_transform(),
        rtol=1e-5, atol=1e-8
    )

    # check operations_out[1]
    self.assertIs(operations_out[1], operation_b)

  @parameterized.parameters(
      (
          [circuit.Operation(circuit.RotZGate(0.42), [42])],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.Operation(circuit.RotZGate(0.42), [42])],
          [circuit.ControlledZGate]
      ),
      (
          [circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42])],
          [circuit.RotZGate]
      ),
      (
          [circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42])],
          [circuit.ControlledZGate]
      ),
      (
          [circuit.Operation(circuit.ControlledZGate(), [47, 11])],
          [circuit.RotZGate]
      ),
      (
          [circuit.Operation(circuit.ControlledZGate(), [47, 11])],
          [circuit.PhasedXGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.PhasedXGate, circuit.PhasedXGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.PhasedXGate, circuit.MatrixGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.RotZGate, circuit.RotZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.MatrixGate, circuit.RotZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37])
          ],
          [circuit.PhasedXGate, circuit.RotZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37]),
              circuit.Operation(circuit.ControlledZGate(), [47, 11])
          ],
          [circuit.PhasedXGate, circuit.RotZGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.Operation(circuit.RotZGate(0.42), [42]),
              circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [37]),
              circuit.Operation(circuit.ControlledZGate(), [47, 11])
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.RotZGate]
      )
  )
  def test_negative(self, operations, gate_types):
    with self.assertRaises(circuit.GateNotParsableError):
      # call the function to be tested
      parsing.parse_operations(operations, *gate_types)

  def test_type_error_operations_no_sequence(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'operations is not a sequence (found type: bool)'):
      parsing.parse_operations(False, circuit.MatrixGate, circuit.MatrixGate)

  def test_type_error_operation_not_an_operation(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all operations are instances of Operation'
        ' [found: instance(s) of float]'):
      parsing.parse_operations(
          [operation, 47.11],
          circuit.MatrixGate, circuit.MatrixGate
      )

  def test_type_error_gatetype_no_type(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gate_types are types [found: instance(s) of int]'):
      parsing.parse_operations(
          [operation, operation],
          circuit.MatrixGate, 42
      )

  def test_type_error_gatetype_no_gate(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gate_types are subtypes of Gate [found types: str]'):
      parsing.parse_operations(
          [operation, operation],
          circuit.MatrixGate, str
      )

  def test_value_error_too_many_operations(self):
    operations = [
        circuit.Operation(
            circuit.MatrixGate(stats.unitary_group.rvs(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(stats.unitary_group.rvs(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'inconsistent length of operations and gate_types (2 vs 1)'):
      parsing.parse_operations(operations, circuit.MatrixGate)

  def test_value_error_too_many_types(self):
    operations = [
        circuit.Operation(
            circuit.MatrixGate(stats.unitary_group.rvs(2)),
            [42]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'inconsistent length of operations and gate_types (1 vs 2)'):
      parsing.parse_operations(
          operations,
          circuit.MatrixGate, circuit.MatrixGate
      )

  def test_no_parser_registered_error(self):
    class DummyGate(circuit.Gate):
      pass

    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        parsing.NoParserRegisteredError,
        'no parser registered for gate type DummyGate'):
      parsing.parse_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_raises_unexpected_error_without_message(self):
    class DummyGate(circuit.Gate):
      pass

    def parser(gate):
      raise ValueError

    # preparation work: register the (erroneous) parser and construct operations
    parsing.register_parser(DummyGate, parser)
    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate unexpectedly raised ValueError'
        ' (without message)'):
      gate_out = parsing.parse_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_raises_unexpected_error_with_message(self):
    class DummyGate(circuit.Gate):
      pass

    def parser(gate):
      raise ValueError('my error message')

    # preparation work: register the (erroneous) parser and construct operations
    parsing.register_parser(DummyGate, parser)
    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate unexpectedly raised ValueError'
        ' "my error message"'):
      gate_out = parsing.parse_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_returns_none(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct operations
    parsing.register_parser(DummyGate, lambda gate: None)
    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate did not return anything'):
      gate_out = parsing.parse_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_returns_invalid_type(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct operations
    parsing.register_parser(DummyGate, lambda gate: gate)
    operations = [
        circuit.Operation(
            circuit.MatrixGate(np.eye(2)),
            [42]
        ),
        circuit.Operation(
            circuit.MatrixGate(np.eye(4)),
            [47, 11]
        )
    ]

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate returned an instance'
        ' of type MatrixGate'):
      gate_out = parsing.parse_operations(
          operations,
          circuit.MatrixGate, DummyGate
      )


class ParseOperationTest(parameterized.TestCase):

  @parameterized.parameters([
      (
          circuit.Operation(
              circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
              [42]
          ),
          circuit.RotZGate
      ),
      (
          circuit.Operation(circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]]), [42]),
          circuit.PhasedXGate
      ),
      (
          circuit.Operation(circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0])),
                            [47, 11]),
          circuit.ControlledZGate
      ),
      (
          circuit.Operation(circuit.RotZGate(0.42), [42]),
          circuit.MatrixGate
      ),
      (
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42]),
          circuit.MatrixGate
      ),
      (
          circuit.Operation(circuit.ControlledZGate(), [47, 11]),
          circuit.MatrixGate
      )
  ])
  def test_positive(self, operation_in, gate_type):
    # call the function to be tested
    operation_out = parsing.parse_operation(operation_in, gate_type)

    # check type of operation_out
    self.assertIsInstance(operation_out, circuit.Operation)

    # extract gates of operation_out
    gate_out = operation_out.get_gate()

    # check type of gate_out
    self.assertIsInstance(gate_out, gate_type)

    # check properties of operation_out and gate_out
    self.assertEqual(
        operation_out.get_num_qubits(),
        operation_in.get_num_qubits()
    )
    self.assertEqual(
        operation_out.get_qubits(),
        operation_in.get_qubits()
    )
    np.testing.assert_allclose(
        gate_out.get_pauli_transform(),
        operation_in.get_gate().get_pauli_transform(),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters([
      (
          circuit.Operation(circuit.RotZGate(0.42), [42]),
          circuit.RotZGate
      ),
      (
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42]),
          circuit.PhasedXGate
      ),
      (
          circuit.Operation(circuit.ControlledZGate(), [47, 11]),
          circuit.ControlledZGate
      ),
      (
          circuit.Operation(circuit.MatrixGate(stats.unitary_group.rvs(2)),
                            [42]),
          circuit.MatrixGate
      ),
      (
          circuit.Operation(circuit.MatrixGate(stats.unitary_group.rvs(4)),
                            [47, 11]),
          circuit.MatrixGate
      )
  ])
  def test_identical(self, operation_in, gate_type):
    # call the function to be tested
    operation_out = parsing.parse_operation(operation_in, gate_type)

    # check gate_out
    self.assertIs(operation_out, operation_in)

  @parameterized.parameters([
      (
          circuit.Operation(circuit.RotZGate(0.42), [42]),
          circuit.PhasedXGate
      ),
      (
          circuit.Operation(circuit.RotZGate(0.42), [42]),
          circuit.ControlledZGate
      ),
      (
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42]),
          circuit.RotZGate
      ),
      (
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [42]),
          circuit.ControlledZGate
      ),
      (
          circuit.Operation(circuit.ControlledZGate(), [47, 11]),
          circuit.RotZGate
      ),
      (
          circuit.Operation(circuit.ControlledZGate(), [47, 11]),
          circuit.PhasedXGate
      )
  ])
  def test_negative(self, operation, gate_type):
    with self.assertRaises(circuit.GateNotParsableError):
      # call the function to be tested
      parsing.parse_operation(operation, gate_type)

  def test_type_error_operation_not_an_operation(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'operation is not an Operation (found type: float)'):
      parsing.parse_operation(47.11, circuit.MatrixGate)

  def test_type_error_gatetype_no_type(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'gate_type is not a type (found: instance of int)'):
      parsing.parse_operation(operation, 42)

  def test_type_error_gatetype_no_gate(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'gate_type is not a Gate type (found: str)'):
      parsing.parse_operation(operation, str)

  def test_no_parser_registered_error(self):
    class DummyGate(circuit.Gate):
      pass

    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        parsing.NoParserRegisteredError,
        'no parser registered for gate type DummyGate'):
      parsing.parse_operation(operation, DummyGate)

  def test_runtime_error_parser_raises_unexpected_error_without_message(self):
    class DummyGate(circuit.Gate):
      pass

    def parser(gate):
      raise ValueError

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, parser)
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate unexpectedly raised ValueError'
        ' (without message)'):
      gate_out = parsing.parse_operation(operation, DummyGate)

  def test_runtime_error_parser_raises_unexpected_error_with_message(self):
    class DummyGate(circuit.Gate):
      pass

    def parser(gate):
      raise ValueError('my error message')

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, parser)
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate unexpectedly raised ValueError'
        ' "my error message"'):
      gate_out = parsing.parse_operation(operation, DummyGate)

  def test_runtime_error_parser_returns_none(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, lambda gate: None)
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate did not return anything'):
      gate_out = parsing.parse_operation(operation, DummyGate)

  def test_runtime_error_parser_returns_invalid_type(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, lambda gate: gate)
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate returned an instance'
        ' of type MatrixGate'):
      gate_out = parsing.parse_operation(operation, DummyGate)


class CheckGatesTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          [circuit.RotZGate(0.42)],
          [circuit.RotZGate]
      ),
      (
          [circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j])))],
          [circuit.RotZGate]
      ),
      (
          [circuit.PhasedXGate(0.815, 0.4711)],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]])],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.ControlledZGate()],
          [circuit.ControlledZGate]
      ),
      (
          [circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0]))],
          [circuit.ControlledZGate]
      ),
      (
          [
              circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
              circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]]),
              circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0]))
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.ControlledZGate()
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.ControlledZGate()
          ],
          [circuit.MatrixGate, circuit.MatrixGate, circuit.MatrixGate]
      ),
      (
          [
              circuit.MatrixGate(stats.unitary_group.rvs(2)),
              circuit.MatrixGate(stats.unitary_group.rvs(4))
          ],
          [circuit.MatrixGate, circuit.MatrixGate]
      ),
      (
          [
              circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.RotZGate, circuit.PhasedXGate]
      )
  )
  def test_positive(self, gates, gate_types):
    # call the function to be tested
    is_parsable = parsing.check_gates(gates, *gate_types)

    # check type and value of is_parsable
    self.assertIs(type(is_parsable), bool)
    self.assertTrue(is_parsable)

  @parameterized.parameters(
      (
          [circuit.RotZGate(0.42)],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.RotZGate(0.42)],
          [circuit.ControlledZGate]
      ),
      (
          [circuit.PhasedXGate(0.815, 0.4711)],
          [circuit.RotZGate]
      ),
      (
          [circuit.PhasedXGate(0.815, 0.4711)],
          [circuit.ControlledZGate]
      ),
      (
          [circuit.ControlledZGate()],
          [circuit.RotZGate]
      ),
      (
          [circuit.ControlledZGate()],
          [circuit.PhasedXGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.PhasedXGate, circuit.PhasedXGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.PhasedXGate, circuit.MatrixGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.RotZGate, circuit.RotZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.MatrixGate, circuit.RotZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.PhasedXGate, circuit.RotZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.ControlledZGate()
          ],
          [circuit.PhasedXGate, circuit.RotZGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.ControlledZGate()
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.RotZGate]
      )
  )
  def test_negative(self, gates, gate_types):
    # call the function to be tested
    is_parsable = parsing.check_gates(gates, *gate_types)

    # check type and value of is_parsable
    self.assertIs(type(is_parsable), bool)
    self.assertFalse(is_parsable)

  def test_type_error_gates_no_sequence(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'gates is not a sequence (found type: bool)'):
      parsing.check_gates(False, circuit.MatrixGate, circuit.MatrixGate)

  def test_type_error_gate_not_a_gate(self):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gates are instances of Gate [found: instance(s) of float]'):
      parsing.check_gates(
          [gate, 47.11],
          circuit.MatrixGate, circuit.MatrixGate
      )

  def test_type_error_gatetype_no_type(self):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gate_types are types [found: instance(s) of int]'):
      parsing.check_gates(
          [gate, gate],
          circuit.MatrixGate, 42
      )

  def test_type_error_gatetype_no_gate(self):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gate_types are subtypes of Gate [found types: str]'):
      parsing.check_gates(
          [gate, gate],
          circuit.MatrixGate, str
      )

  def test_value_error_too_many_gates(self):
    gates = [
        circuit.MatrixGate(stats.unitary_group.rvs(2)),
        circuit.MatrixGate(stats.unitary_group.rvs(4))
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'inconsistent length of gates and gate_types (2 vs 1)'):
      parsing.check_gates(gates, circuit.MatrixGate)

  def test_value_error_too_many_types(self):
    gates = [circuit.MatrixGate(stats.unitary_group.rvs(2))]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'inconsistent length of gates and gate_types (1 vs 2)'):
      parsing.check_gates(gates, circuit.MatrixGate, circuit.MatrixGate)

  def test_no_parser_registered_error(self):
    class DummyGate(circuit.Gate):
      pass

    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        parsing.NoParserRegisteredError,
        'no parser registered for gate type DummyGate'):
      parsing.check_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )

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
      gate_out = parsing.check_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )

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
      gate_out = parsing.check_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_returns_none(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, lambda gate: None)
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate did not return anything'):
      gate_out = parsing.check_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )

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
      gate_out = parsing.check_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )


class ParseGatesTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          [circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j])))],
          [circuit.RotZGate]
      ),
      (
          [circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]])],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0]))],
          [circuit.ControlledZGate]
      ),
      (
          [
              circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j]))),
              circuit.MatrixGate([[0.0, 1.0], [1.0, 0.0]]),
              circuit.MatrixGate(np.diag([1.0, 1.0, 1.0, -1.0]))
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.ControlledZGate()
          ],
          [circuit.MatrixGate, circuit.MatrixGate, circuit.MatrixGate]
      )
  )
  def test_positive(self, gates_in, gate_types):
    # call the function to be tested
    gates_out = parsing.parse_gates(gates_in, *gate_types)

    # check type of gates_out
    self.assertIs(type(gates_out), list)

    # check length of gates_out
    self.assertLen(gates_out, len(gates_in))

    # check types of gates_out elements
    self.assertTrue(all(
        isinstance(gate, gate_type)
        for gate, gate_type in zip(gates_out, gate_types)
    ))

    # check properties of gates_out elements
    self.assertTrue(all(
        gate_out.get_num_qubits() == gate_in.get_num_qubits()
        for gate_out, gate_in in zip(gates_out, gates_in)
    ))
    self.assertTrue(all(
        np.allclose(
            gate_out.get_pauli_transform(),
            gate_in.get_pauli_transform()
        )
        for gate_out, gate_in in zip(gates_out, gates_in)
    ))

  @parameterized.parameters(
      (
          [circuit.RotZGate(0.42)],
          [circuit.RotZGate]
      ),
      (
          [circuit.PhasedXGate(0.815, 0.4711)],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.ControlledZGate()],
          [circuit.ControlledZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.ControlledZGate()
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.MatrixGate(stats.unitary_group.rvs(2)),
              circuit.MatrixGate(stats.unitary_group.rvs(4))
          ],
          [circuit.MatrixGate, circuit.MatrixGate]
      )
  )
  def test_identical(self, gates_in, gate_types):
    # call the function to be tested
    gates_out = parsing.parse_gates(gates_in, *gate_types)

    # check type of gates_out
    self.assertIs(type(gates_out), list)

    # check length of gates_out
    self.assertLen(gates_out, len(gates_in))

    # check elements of gates_out
    self.assertTrue(all(
        gate_out is gate_in
        for gate_out, gate_in in zip(gates_out, gates_in)
    ))

  def test_mixed(self):
    # preparation work: construct two gates
    gate_a = circuit.MatrixGate(np.diag(np.exp([0.42j, 0.137j])))
    gate_b = circuit.PhasedXGate(0.815, 0.4711)

    # call the function to be tested
    gates_out = parsing.parse_gates(
        [gate_a, gate_b],
        circuit.RotZGate, circuit.PhasedXGate
    )

    # check type of gates_out
    self.assertIs(type(gates_out), list)

    # check length of gate_out
    self.assertLen(gates_out, 2)

    # check type of gates_out[0]
    self.assertIsInstance(gates_out[0], circuit.RotZGate)

    # check properties of gates_out[0]
    self.assertEqual(gates_out[0].get_num_qubits(), gate_a.get_num_qubits())
    np.testing.assert_allclose(
        gates_out[0].get_pauli_transform(),
        gate_a.get_pauli_transform(),
        rtol=1e-5, atol=1e-8
    )

    # check gates_out[1]
    self.assertIs(gates_out[1], gate_b)

  @parameterized.parameters(
      (
          [circuit.RotZGate(0.42)],
          [circuit.PhasedXGate]
      ),
      (
          [circuit.RotZGate(0.42)],
          [circuit.ControlledZGate]
      ),
      (
          [circuit.PhasedXGate(0.815, 0.4711)],
          [circuit.RotZGate]
      ),
      (
          [circuit.PhasedXGate(0.815, 0.4711)],
          [circuit.ControlledZGate]
      ),
      (
          [circuit.ControlledZGate()],
          [circuit.RotZGate]
      ),
      (
          [circuit.ControlledZGate()],
          [circuit.PhasedXGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.PhasedXGate, circuit.PhasedXGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.PhasedXGate, circuit.MatrixGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.RotZGate, circuit.RotZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.MatrixGate, circuit.RotZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711)
          ],
          [circuit.PhasedXGate, circuit.RotZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.ControlledZGate()
          ],
          [circuit.PhasedXGate, circuit.RotZGate, circuit.ControlledZGate]
      ),
      (
          [
              circuit.RotZGate(0.42),
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.ControlledZGate()
          ],
          [circuit.RotZGate, circuit.PhasedXGate, circuit.RotZGate]
      )
  )
  def test_negative(self, gates, gate_types):
    with self.assertRaises(circuit.GateNotParsableError):
      # call the function to be tested
      parsing.parse_gates(gates, *gate_types)

  def test_type_error_gates_no_sequence(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'gates is not a sequence (found type: bool)'):
      parsing.parse_gates(False, circuit.MatrixGate, circuit.MatrixGate)

  def test_type_error_gate_not_a_gate(self):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gates are instances of Gate [found: instance(s) of float]'):
      parsing.parse_gates(
          [gate, 47.11],
          circuit.MatrixGate, circuit.MatrixGate
      )

  def test_type_error_gatetype_no_type(self):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gate_types are types [found: instance(s) of int]'):
      parsing.parse_gates(
          [gate, gate],
          circuit.MatrixGate, 42
      )

  def test_type_error_gatetype_no_gate(self):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'not all gate_types are subtypes of Gate [found types: str]'):
      parsing.parse_gates(
          [gate, gate],
          circuit.MatrixGate, str
      )

  def test_value_error_too_many_gates(self):
    gates = [
        circuit.MatrixGate(stats.unitary_group.rvs(2)),
        circuit.MatrixGate(stats.unitary_group.rvs(4))
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'inconsistent length of gates and gate_types (2 vs 1)'):
      parsing.parse_gates(gates, circuit.MatrixGate)

  def test_value_error_too_many_types(self):
    gates = [circuit.MatrixGate(stats.unitary_group.rvs(2))]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'inconsistent length of gates and gate_types (1 vs 2)'):
      parsing.parse_gates(gates, circuit.MatrixGate, circuit.MatrixGate)

  def test_no_parser_registered_error(self):
    class DummyGate(circuit.Gate):
      pass

    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        parsing.NoParserRegisteredError,
        'no parser registered for gate type DummyGate'):
      parsing.parse_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )

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
      gate_out = parsing.parse_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )

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
      gate_out = parsing.parse_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )

  def test_runtime_error_parser_returns_none(self):
    class DummyGate(circuit.Gate):
      pass

    # preparation work: register the (erroneous) parser and construct a gate
    parsing.register_parser(DummyGate, lambda gate: None)
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        'internal error: parser for DummyGate did not return anything'):
      gate_out = parsing.parse_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )

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
      gate_out = parsing.parse_gates(
          [gate, gate],
          circuit.MatrixGate, DummyGate
      )


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
