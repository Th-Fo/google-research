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
"""Tool for parsing Gate types.

This module currently contains two functions, parse_gates(...) and
parse_operations(...), which implement gate parsing for sequences of Gate and
Operation instances, respectively.
"""

from typing import Callable

from rl4circopt import circuit

_parsers = {}


# TODO(tfoesel): improve design for this module and add unit tests


def check_gates(gates, *gate_types):
  """Checks whether the gates match the expected types."""
  return parse_gates(gates, *gate_types) is not None


def check_operations(operations, *gate_types):
  """Checks whether the gates of the operations match the expected types."""
  return parse_operations(operations, *gate_types) is not None


def parse_gates(gates, *gate_types):
  """Parses gates into expected gate types."""

  if len(gates) != len(gate_types):
    raise ValueError('inconsistent length of gates and gate_types (%d vs %d)'
                     %(len(gates), len(gate_types)))

  parsed_gates = []

  try:
    return [
        parse_gate(gate, gate_type)
        for gate, gate_type in zip(gates, gate_types)
    ]
  except circuit.GateNotParsableError:
    return None


def parse_operations(operations, *gate_types):
  """Parse operations into expected gate types."""

  if len(operations) != len(gate_types):
    raise ValueError('inconsistent length of operations and gate_types'
                     ' (%d vs %d)'%(len(operations), len(gate_types)))

  for operation in operations:
    if not isinstance(operation, circuit.Operation):
      raise TypeError('%s is not an Operation'%type(operation).__name__)

  parsed_gates = parse_gates(
      [operation.get_gate() for operation in operations],
      *gate_types
  )

  if parsed_gates is None:
    return None
  else:
    parsed_operations = []

    for operation, parsed_gate in zip(operations, parsed_gates):
      if operation.get_gate() is not parsed_gate:
        operation = circuit.Operation(parsed_gate, operation.get_qubits())
      parsed_operations.append(operation)

    return parsed_operations

def parse_gate(gate: circuit.Gate, gate_type: type) -> circuit.Gate:
  """Tries to convert a gate into the expected gate type.

  Returns an instance of the specified gate_type which is equivalent to the
  specified gate, or raises a GateNotParsableError to indicate that this is not
  possible.

  Args:
      gate: the gate to be parsed.
      gate_type: a subtype of Gate into which the input gate should be
          converted.

  Returns:
      an instance of the specified gate_type that matches the specified input
      gate up to maximally a global phase. If the input gate is already an
      instance of gate_type, then the output can be identical to it.

  Raises:
      TypeError: if gate is not a Gate instance, or if gate_type is not a type
          instance that represents a subtype of Gate.
      NoParserRegisteredError: if no parser is registered for the specified
          gate_type.
      GateNotParsableError: if no instance of the specified gate_type exists
          that matches the specified input gate up to maximally a global phase.
      RuntimeError: if the parser registered for gate_type does not behave as
          expected, i.e. if it raises an exception different from
          GateNotParsableError (given an input that is promised to be a Gate
          instance), or if its return value is not an instance of the specified
          gate_type.
  """

  # check the input arguments
  if not isinstance(gate, circuit.Gate):
    raise TypeError('gate is not a Gate (found type: %s)'%type(gate).__name__)
  if not isinstance(gate_type, type):
    raise TypeError('gate_type is not a type (found: instance of %s)'
                    %type(gate_type).__name__)
  if not issubclass(gate_type, circuit.Gate):
    raise TypeError('gate_type is not a Gate type (found: %s)'
                    %gate_type.__name__)

  try:
    parser = _parsers[gate_type]
  except KeyError:
    raise NoParserRegisteredError('no parser registered for gate type %s'
                                  %gate_type.__name__)

  try:
    gate_out = parser(gate)
  except circuit.GateNotParsableError as err:
    raise err
  except Exception as err:
    error_message = getattr(err, 'message', str(err))
    raise RuntimeError(
        'internal error: parser for %s unexpectedly raised %s %s'%(
            gate_type.__name__,
            type(err).__name__,
            '\"%s\"'%error_message if error_message else '(without message)'
        )
    )

  if gate_out is None:
    raise RuntimeError('internal error: parser for %s did not return anything'
                       %gate_type.__name__)
  if not isinstance(gate_out, gate_type):
    raise RuntimeError(
        'internal error: parser for %s returned an instance of type %s'
        %(gate_type.__name__, type(gate_out).__name__)
    )

  return gate_out


def register_parser(gate_type: type,
                    parser: Callable[[circuit.Gate], circuit.Gate]):
  """Register a new parser for a Gate type.

  Args:
    gate_type: the subtype of Gate for which the parser should be responsible.
    parser: a callable with signature `gate_out = parser(gate_in)`, where
      gate_in is promised to be an instance of Gate, and `gate_out` must be an
      instance of the specified gate_type. If gate_in cannot be represented by
      an instance of gate_type, the parser is supposed to raise a
      GateNotParsableError.

  Raises:
    TypeError: if gate_type is not a type instance that represents a subtype of
      Gate, or if parser is not callable.
  """

  # check the input arguments
  if not isinstance(gate_type, type):
    raise TypeError('gate_type is not a type (found: instance of %s)'
                    %type(gate_type).__name__)
  if not issubclass(gate_type, circuit.Gate):
    raise TypeError('gate_type is not a Gate type (found: %s)'
                    %gate_type.__name__)
  if not callable(parser):
    raise TypeError('parser is not callable (found type: %s)'
                    %type(parser).__name__)

  # print a warning if a parser has already been registered for this gate type
  if gate_type in _parsers:
    print(
        'overriding parser for gate type %s'%gate_type.__name__,
        file=sys.stderr
    )

  # actually register the parser
  _parsers[gate_type] = parser


class NoParserRegisteredError(Exception):
  """Indicates that no parser has been registered for a certain gate type."""
  pass


# register parsers for standard gates

register_parser(circuit.MatrixGate, circuit.MatrixGate.parse)
register_parser(circuit.PhasedXGate, circuit.PhasedXGate.parse)
register_parser(circuit.RotZGate, circuit.RotZGate.parse)
register_parser(circuit.ControlledZGate, circuit.ControlledZGate.parse)
register_parser(circuit.FermionicSimulationGate,
                circuit.FermionicSimulationGate.parse)
