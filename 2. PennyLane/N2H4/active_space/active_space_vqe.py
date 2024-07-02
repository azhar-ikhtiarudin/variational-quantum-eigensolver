import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import matplotlib.pyplot as plt

A_to_au_conversion = 1.8897259885789

def calc_vqe_N2H4(threshold, active_electrons, active_orbitals):
    symbols = ["N", "H", "H", "N", "H", "H"]
    geometry = np.array([0.0, 0.0, 0.0 , 0.0, 0.0, 1.015264, 0.978541, 0.0, -0.270591, -0.627449, 1.276052, -0.477492 , -0.897827, 1.825923, 0.332013 , 0.080714, 1.825923, -0.953842])*A_to_au_conversion
    
    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        geometry,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        # method="pyscf"
    )
    print("qubit:", qubits)
    dev = qml.device("lightning.qubit", wires=qubits)
    electrons = active_electrons
    hf = qml.qchem.hf_state(electrons, qubits)

    def circuit(param, wires):
        qml.BasisState(hf, wires=wires)
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

    @qml.qnode(dev, interface="autograd")
    def cost_fn(param):
        circuit(param, wires=range(qubits))
        return qml.expval(H)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)

    # store the values of the cost function
    energy = [cost_fn(theta)]
    # store the values of the circuit parameter
    angle = [theta]
    max_iterations = 100
    conv_tol = threshold

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy.append(cost_fn(theta))
        angle.append(theta)

        conv = np.abs(energy[-1] - prev_energy)
        # print("conv: ", conv)
        print("conv: ", conv)
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break

    return energy, angle, n


config = [[2,2]]
threshold = 1e-6

for i in range(len(config)):
    print("Configuration: ", threshold, config[i][0], config[i][1])
    E, angle, n = calc_vqe_N2H4(threshold, active_electrons=config[i][0], active_orbitals=config[i][1])
    
    file_path = f"data/N2H4_{threshold}_{config[i][0]}_{config[i][1]}.txt"
    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write each element of the array to a new line
        for element in E:
            file.write(str(element) + "\n")
