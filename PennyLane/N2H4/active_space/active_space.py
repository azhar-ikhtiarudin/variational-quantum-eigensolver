import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import matplotlib.pyplot as plt

A_to_au_conversion = 1.8897259885789

def calc_adapt_vqe_N2H4(threshold, active_electrons=4, active_orbitals=4):
    symbols = ["N", "H", "H", "N", "H", "H"]
    # ground state coordinate: 
    geometry = np.array([0.0, 0.0, 0.0 , 0.0, 0.0, 1.015264, 0.978541, 0.0, -0.270591, -0.627449, 1.276052, -0.477492 , -0.897827, 1.825923, 0.332013 , 0.080714, 1.825923, -0.953842])*A_to_au_conversion

    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        geometry,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        # method="pyscf"
    )

    active_electrons = active_electrons

    singles, doubles = qchem.excitations(active_electrons, qubits)

    print(f"Total number of excitations = {len(singles) + len(doubles)}")
    singles_excitations = [qml.SingleExcitation(0.0, x) for x in singles]
    doubles_excitations = [qml.DoubleExcitation(0.0, x) for x in doubles]
    operator_pool = doubles_excitations + singles_excitations   

    hf_state = qchem.hf_state(active_electrons, qubits)
    dev = qml.device("default.qubit", wires=qubits)
    @qml.qnode(dev)
    def circuit():
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]]
        return qml.expval(H)
    energy_array = []
    
    opt = qml.optimize.AdaptiveOptimizer()
    for i in range(len(operator_pool)):
        circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool)
        energy_array.append(energy)
        if i % 3 == 0:
            print("n = {:},  E = {:.8f} H, Largest Gradient = {:.3f}".format(i, energy, gradient))
            # print(qml.draw(circuit, decimals=None)())
            print()
        if gradient < threshold*10^(-threshold):
            break
    return energy_array, circuit

config = [[4,6]]
threshold = 3

for i in range(len(config)):
    print("Configuration: ", threshold, config[i][0], config[i][1])
    E, circuit = calc_adapt_vqe_N2H4(threshold, active_electrons=config[i][0], active_orbitals=config[i][1])
    
    file_path = f"data/N2H4_{threshold}_{config[i][0]}_{config[i][1]}.txt"
    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write each element of the array to a new line
        for element in E:
            file.write(str(element) + "\n")
