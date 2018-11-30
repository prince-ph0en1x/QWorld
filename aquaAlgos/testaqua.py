"""
This is a simple tutorial example to show how to build an algorithm to extend Qiskit AQUA library.
Algorithms are designed to be dynamically discovered within Qiskit AQUA.
For this the entire parent directory 'QWorld_IBM' should be moved under the 'qiskit_aqua' directory.
The current demonstration notebook shows how to explicitly register the algorithm and works without re-locating this code.
The former automatic discovery does however allow the algorithm to be found and seen in the UI browser, and selected from the GUI when choosing an algorithm.
"""
import logging
import numpy as np
from qiskit import QuantumRegister
from qiskit.tools.qi.qi import state_fidelity

from qiskit_aqua import QuantumAlgorithm
from qiskit_aqua import AlgorithmError
from qiskit_aqua import get_initial_state_instance


logger = logging.getLogger(__name__)

class testAqua(QuantumAlgorithm):
    """The Tutorial Sample testAqua algorithm."""
    PROP_EXPANSION_ORDER = 'expansion_order'

    """
    A configuration dictionary defines the algorithm to QISKIt AQUA. It can contain
    the following though this sample does not have them all.
    name: Is the registered name and will be used as the case-sensitive key to load an instance
    description: As it implies a brief description of algorithm
    classical: True if purely a classical algorithm that does not need a quantum backend
    input_schema: A json schema detailing the configuration variables of this entity.
                  Each variable as a type, and can be given default, minimum etc. This conforms
                  to JSON Schema which can be consulted for for detail. The existing algorithms
                  and other pluggable entities may also be helpful to refer to.
    problems: A list of problems the algorithm can solve
    depends: A list of dependent object types
    defaults: A list of configurations for the dependent objects. May just list names if the
              dependent's defaults are acceptable
    """
    EVOLUTIONFIDELITY_CONFIGURATION = {
        'name': 'testAqua',
        'description': 'Sample Demo testAqua Algorithm for Quantum Systems',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'evolution_fidelity_schema',
            'type': 'object',
            'properties': {
                PROP_EXPANSION_ORDER: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
            },
            'additionalProperties': False
        },
        'problems': []
    }

    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): algorithm configuration
        """
        super().__init__(configuration or self.EVOLUTIONFIDELITY_CONFIGURATION.copy())

    """
    init_params is called via run_algorithm. The params contain all the configuration settings 
    of the objects. algo_input contains data computed from above for the algorithm. A simple
    algorithm may have all its data in configuration params such that algo_input is None  
    """
    def init_params(self, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: EnergyInput instance
        """
        if algo_input is None:
            raise AlgorithmError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        evolution_fidelity_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        expansion_order = evolution_fidelity_params.get(testAqua.PROP_EXPANSION_ORDER)

        # Set up initial state, we need to add computed num qubits to params
        initial_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        initial_state_params['num_qubits'] = operator.num_qubits
        initial_state = get_initial_state_instance(initial_state_params['name'])
        initial_state.init_params(initial_state_params)

        self.init_args(operator, initial_state, expansion_order)

    """
    If directly use these objects programmatically then init_args is more convenient to call
    than init_params. init_params itself uses this to do the actual object initialization. 
    """
    def init_args(self, operator, initial_state, expansion_order):
        self._operator = operator
        self._initial_state = initial_state
        self._expansion_order = expansion_order
        self._ret = {}

    """
    Once the algorithm has been initialized then run is called to carry out the computation
    and the result is returned as a dictionary. 
    """
    def run(self):
        evo_time = 1
        # get the groundtruth via simple matrix * vector
        state_out_exact = self._operator.evolve(self._initial_state.construct_circuit('vector'), evo_time, 'matrix', 0)

        qr = QuantumRegister(self._operator.num_qubits, name='q')
        circuit = self._initial_state.construct_circuit('circuit', qr)
        circuit += self._operator.evolve(
            None, evo_time, 'circuit', 1,
            quantum_registers=qr,
            expansion_mode='suzuki',
            expansion_order=self._expansion_order
        )

        result = self.execute(circuit)
        state_out_dynamics = np.asarray(result.get_statevector(circuit))

        self._ret['score'] = state_fidelity(state_out_exact, state_out_dynamics)

        return self._ret
