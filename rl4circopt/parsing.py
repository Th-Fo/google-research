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
"""Tools for parsing gates, either standalone or as part of an operation.

This module provides the following 6 functions:

* check_operations(...)   checks whether the gates of an operation sequence
                          could be parsed into the specified gate types
* parse_operations(...)   parses the gates in an operation sequence into the
                          specified gate types (if possible)
* parse_operation(...)    parses the gate of one operation into a specified gate
                          type (if possible)
* check_gates(...)        checks whether a gate sequence could be parsed into
                          the specified gate types
* parse_gates(...)        parses a gate sequence into the specified gate types
                          (if possible)
* parse_gate(...)         parses one gate into a specified gate type (if
                          possible)

These functions internally call specific parsers responsible for the respective
target gate types. Such a parser has the signature

    gate_out = parser(gate_in)

where gate_out must be an instance of the target gate type.

The management of these parsers is also done by this module. For the standard
gates defined in circuit.py, parsers are automatically available. More parsers
can be added using the register_parser(...) function.
"""

from typing import Callable, List, Sequence, Tuple

from rl4circopt import circuit

import sys

_parsers = {}


def check_operations(operations: Sequence[circuit.Operation],
                     *gate_types: Tuple[type]
                    ) -> bool:
  """Checks if a sequence of operations could be converted into the expected gate types.

  Returns a bool indicating whether, for every index n, it is possible to
  convert the gate of operations[n] into gate_types[n].

  Args:
      operations: the sequence of operations to be checked.
      gate_types: a tuple of Gate subtypes, of the same length as operations.
          For every index n, `gate_types[n]` specifies the Gate type into which
          the gate of `operations[n]` should be converted.

  Returns:
      True iff for every index n, there is an instance of gate_types[n] which
      matches the gate of operations[n] up to maximally a global phase.

  Raises:
      TypeError: if operations is not a sequence of Operation instances, or if
          gate_types does not only contain type instances that represent
          subtypes of Gate.
      ValueError: if the lengths of operations and gate_types do not match.
      NoParserRegisteredError: if no parser is registered for one of the
          specified gate_types.
      RuntimeError: if the parser registered for one of the gate_types does not
          behave as expected, i.e. if it raises an exception different from
          GateNotParsableError (given an input that is promised to be a Gate
          instance), or if its return value is not an instance of the specified
          Gate type.
  """

  try:
    # besides a GateNotParsableError, possibly also raises a TypeError,
    # ValueError, NoParserRegisteredError or RuntimeError
    parse_operations(operations, *gate_types)
  except circuit.GateNotParsableError:
    return False
  else:
    return True


def parse_operations(operations: Sequence[circuit.Operation],
                     *gate_types: Tuple[type]
                    ) -> List[circuit.Operation]:
  """Tries to convert a sequence of operations into the expected gate types.

  Returns a list of operations where the n-th output operation is equivalent to
  the n-th input operation, and its gate is simultaneously an instance of
  gate_types[n], or raises a GateNotParsableError to indicate that this is not
  possible.

  Args:
      operations: the sequence of operations to be parsed.
      gate_types: a tuple of Gate subtypes, of the same length as operations.
          For every index n, `gate_types[n]` specifies the Gate type into which
          the gate of `operations[n]` should be converted.

  Returns:
      a sequence of operations whose gates are an instance of their
      corresponding gate_types, and which match their corresponding input
      operations up to maximally a global phase. If the gates of some input
      operations are already an instance of their target Gate type, these
      operations may become part of the output without any conversion.

  Raises:
      TypeError: if operations is not a sequence of Operation instances, or if
          gate_types does not only contain type instances that represent
          subtypes of Gate.
      ValueError: if the lengths of operations and gate_types do not match.
      NoParserRegisteredError: if no parser is registered for one of the
          specified gate_types.
      GateNotParsableError: if for the gate of (at least) one of the input
          operations, no instance of the corresponding specified Gate type
          exists that matches this gate up to maximally a global phase.
      RuntimeError: if the parser registered for one of the gate_types does not
          behave as expected, i.e. if it raises an exception different from
          GateNotParsableError (given an input that is promised to be a Gate
          instance), or if its return value is not an instance of the specified
          Gate type.
  """
  try:
    operations = list(operations)
  except TypeError:
    raise TypeError('operations is not a sequence (found type: %s)'
                    %type(operations).__name__)

  if not all(isinstance(operation, circuit.Operation)
             for operation in operations):
    illegal_types = set(
        type(operation)
        for operation in operations
        if not isinstance(operation, circuit.Operation)
    )
    raise TypeError(
        'not all operations are instances of Operation [found: instance(s)'
        ' of %s]'%', '.join(sorted(t.__name__ for t in illegal_types))
    )

  _check_gate_types(gate_types)  # possibly raises a TypeError

  if len(operations) != len(gate_types):
    raise ValueError(
        'inconsistent length of operations and gate_types (%d vs %d)'
        %(len(operations), len(gate_types))
    )

  return [
      parse_operation(operation, gate_type)
      for operation, gate_type in zip(operations, gate_types)
  ]


def parse_operation(operation: circuit.Operation,
                    gate_type: type
                   ) -> circuit.Operation:
  """Tries to convert an operation into the expected gate type.

  Returns an operation which is equivalent to the input operation, and whose
  gate is simultaneously an instance of the specified gate_type, or raises a
  GateNotParsableError to indicate that this is not possible.

  Args:
      operation: the operation to be parsed.
      gate_type: a subtype of Gate into which the gate of the input operation
          should be converted.

  Returns:
      an operation whose gate is an instance of the specified gate_type, and
      which matches the specified input operation up to maximally a global
      phase. If the gate of the input operation is already an instance of
      gate_type, then the output can be identical to it.

  Raises:
      TypeError: if operation is not an Operation instance, or if gate_type is
          not a type instance that represents a subtype of Gate.
      NoParserRegisteredError: if no parser is registered for the specified
          gate_type.
      GateNotParsableError: if no instance of the specified gate_type exists
          that matches the gate of the specified input operation up to maximally
          a global phase.
      RuntimeError: if the parser registered for gate_type does not behave as
          expected, i.e. if it raises an exception different from
          GateNotParsableError (given an input that is promised to be a Gate
          instance), or if its return value is not an instance of the specified
          gate_type.
  """

  # check input argument operation (gate_type will be checked in parse_gate)
  if not isinstance(operation, circuit.Operation):
    raise TypeError('operation is not an Operation (found type: %s)'
                    %type(operation).__name__)

  # extract the gate of the input operation
  gate_in = operation.get_gate()

  # try to parse this gate to the specified gate_type (possibly raises a
  # TypeError, GateNotParsableError or RuntimeError)
  gate_out = parse_gate(gate_in, gate_type)

  # return the parsed operation
  return operation if gate_out is gate_in else operation.replace_gate(gate_out)


def check_gates(gates: Sequence[circuit.Gate],
                *gate_types: Tuple[type]
               ) -> bool:
  """Checks if a sequence of gates could be converted into the expected gate types.

  Returns a bool indicating whether, for every index n, it is possible to
  convert gates[n] into gate_types[n].

  Args:
      gates: the sequence of gates to be checked.
      gate_types: a tuple of Gate subtypes, of the same length as gates. For
          every index n, `gate_types[n]` specifies the Gate type into which
          `gates[n]` should be converted.

  Returns:
      True iff for every index n, there is an instance of gate_types[n] which
      matches gates[n] up to maximally a global phase.

  Raises:
      TypeError: if gates is not a sequence of Gate instances, or if gate_types
          does not only contain type instances that represent subtypes of Gate.
      ValueError: if the lengths of gates and gate_types do not match.
      NoParserRegisteredError: if no parser is registered for one of the
          specfied gate_types.
      RuntimeError: if the parser registered for one of the gate_types does not
          behave as expected, i.e. if it raises an exception different from
          GateNotParsableError (given an input that is promised to be a Gate
          instance), or if its return value is not an instance of the specified
          Gate type.
  """

  try:
    # besides a GateNotParsableError, possibly also raises a TypeError,
    # ValueError, NoParserRegisteredError or RuntimeError
    parse_gates(gates, *gate_types)
  except circuit.GateNotParsableError:
    return False
  else:
    return True


def parse_gates(gates: Sequence[circuit.Gate],
                *gate_types: Tuple[type]
               ) -> List[circuit.Gate]:
  """Tries to convert a sequence of gates into the expected gate types.

  Returns a list of gates where the n-th output gate is equivalent to the n-th
  input gate and is simultaneously an instance of the gate_types[n], or raises a
  GateNotParsableError to indicate that this is not possible.

  Args:
      gates: the sequence of gates to be parsed.
      gate_types: a tuple of Gate subtypes, of the same length as gates. For
          every index n, `gate_types[n]` specifies the Gate type into which
          `gates[n]` should be converted.

  Returns:
      a sequence of gates which are instances of their corresponding gate_types,
      and which match their corresponding input gate up to maximally a global
      phase. If some input gates are already an instance of their target Gate
      type, they may become part of the output without any conversion.

  Raises:
      TypeError: if gates is not a sequence of Gate instances, or if gate_types
          does not only contain type instances that represent subtypes of Gate.
      ValueError: if the lengths of gates and gate_types do not match.
      NoParserRegisteredError: if no parser is registered for one of the
          specified gate_types.
      GateNotParsableError: if for (at least) one of the input gates, no
          instance of the corresponding specified Gate type exists that matches
          this gate up to maximally a global phase.
      RuntimeError: if the parser registered for one of the gate_types does not
          behave as expected, i.e. if it raises an exception different from
          GateNotParsableError (given an input that is promised to be a Gate
          instance), or if its return value is not an instance of the specified
          Gate type.
  """
  try:
    gates = list(gates)
  except TypeError:
    raise TypeError('gates is not a sequence (found type: %s)'
                    %type(gates).__name__)

  if not all(isinstance(gate, circuit.Gate) for gate in gates):
    illegal_types = set(
        type(gate)
        for gate in gates
        if not isinstance(gate, circuit.Gate)
    )
    raise TypeError(
        'not all gates are instances of Gate [found: instance(s) of %s]'
        %', '.join(sorted(t.__name__ for t in illegal_types))
    )

  _check_gate_types(gate_types)  # possibly raises a TypeError

  if len(gates) != len(gate_types):
    raise ValueError('inconsistent length of gates and gate_types (%d vs %d)'
                     %(len(gates), len(gate_types)))

  return [
      parse_gate(gate, gate_type)
      for gate, gate_type in zip(gates, gate_types)
  ]


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


def _check_gate_types(gate_types):
  if not all(isinstance(gate_type, type) for gate_type in gate_types):
    illegal_types = set(
        type(gate_type)
        for gate_type in gate_types
        if not isinstance(gate_type, type)
    )
    raise TypeError(
        'not all gate_types are types [found: instance(s) of %s]'
        %', '.join(sorted(t.__name__ for t in illegal_types))
    )
  if not all(issubclass(gate_type, circuit.Gate) for gate_type in gate_types):
    illegal_types = set(
        gate_type
        for gate_type in gate_types
        if not issubclass(gate_type, circuit.Gate)
    )
    raise TypeError(
        'not all gate_types are subtypes of Gate [found types: %s]'
        %', '.join(sorted(t.__name__ for t in illegal_types))
    )


# register parsers for standard gates

register_parser(circuit.MatrixGate, circuit.MatrixGate.parse)
register_parser(circuit.PhasedXGate, circuit.PhasedXGate.parse)
register_parser(circuit.RotZGate, circuit.RotZGate.parse)
register_parser(circuit.ControlledNotGate, circuit.ControlledNotGate.parse)
register_parser(circuit.ControlledZGate, circuit.ControlledZGate.parse)
register_parser(circuit.FermionicSimulationGate,
                circuit.FermionicSimulationGate.parse)
