import numpy as np
import scipy
import matplotlib.pyplot as plt

import numpy as np
import scipy
import matplotlib.pyplot as plt


class Huckel:
    def __init__(self, filename, charge, r_cutoff=1.5, alpha=-11.4, beta=-0.8):
        self.filename = filename
        self.charge = charge
        self.r_cutoff = r_cutoff
        self.alpha = alpha
        self.beta = beta

    def run(self):
        print(f"Running Hückel method on file {self.filename}")

        self.atoms = self.read_geometry()
        print(f"Found {len(self.atoms)} carbon atoms")

        print("Building Hückel Hamiltonian")
        self.H = self.build_huckel_hamiltonian()

        print("Solving eigenvalue problem")
        self.solve_eigenvalues(self.H)

        self.compute_occupation()

        self.compute_energy()

        self.population_analysis()

    def read_geometry(self):
        """This function reads the geometry of a molecule from an XYZ file
        and stores the atom symbol and coordinates of the carbon atoms in a list of tuples.
        """
        atoms = []
        with open(self.filename, "r") as f:
            lines = f.readlines()  # read all lines into a list
            for line in lines[2:]:  # skip the first two lines
                parts = line.split()  # split the line into parts
                symbol = parts[0]
                if symbol == "C":  # only keep carbon atoms
                    xyz = np.array([float(x) for x in parts[1:4]])
                    atoms.append((symbol, xyz))
        return atoms

    def build_huckel_hamiltonian(self):
        """This function builds the Hückel Hamiltonian matrix given a list of atoms."""
        N = len(
            self.atoms
        )  # find the number of atoms from the length of the atoms list
        H = np.zeros((N, N))
        np.fill_diagonal(H, self.alpha)
        for i in range(N):
            # only loop over half of the pairs, excluding the diagonal
            for j in range(i + 1, N):
                r_i = self.atoms[i][1]
                r_j = self.atoms[j][1]
                distance = np.linalg.norm(r_i - r_j)
                if distance < self.r_cutoff:
                    H[i, j] = H[j, i] = self.beta
        return H

    def solve_eigenvalues(self, H):
        """This function solves the eigenvalue problem H C = ε C."""
        self.ε, self.C = np.linalg.eigh(H)
        print(f"Eigenvalues: {self.ε} eV")

    def compute_occupation(self):
        """This function finds the occupation of the molecular orbitals."""

        self.nel = len(self.ε) - self.charge
        print(f"Number of electrons: {self.nel}")

        # find the occupied states
        ndocc = self.nel // 2
        nsocc = self.nel % 2
        print(f"Number of doubly occupation states: {ndocc}")
        print(f"Number of singly occupied states: {nsocc}")
        self.occupation = np.zeros_like(self.ε)
        self.occupation[:ndocc] = 2
        if nsocc:
            self.occupation[ndocc] = 1

    def compute_energy(self):
        """This function computes the energy of the molecule."""
        self.energy = np.dot(self.ε, self.occupation)
        print(f"Total energy: {self.energy:.3f} eV")

    def population_analysis(self):
        """This function performs a population analysis of the molecule."""
        # find the spin up and spin down occupation numbers. Singly occupied states are counted as spin up.
        spin_up_occupation = [1 if x == 1 or x == 2 else 0 for x in self.occupation]
        spin_down_occupation = [1 if x == 2 else 0 for x in self.occupation]

        # compute the density matrices
        self.density_up = self.C @ np.diag(spin_up_occupation) @ self.C.T
        self.density_down = self.C @ np.diag(spin_down_occupation) @ self.C.T

        # compute the Mulliken charges and spin density
        self.mulliken_charges = 1 - np.diag(self.density_up + self.density_down)
        self.spin_density = np.diag(self.density_up - self.density_down)

        print(f"\nMulliken charges")
        print("Atom   Charge     Spin")
        for i in range(len(self.atoms)):
            print(
                f"{i + 1:3} {self.mulliken_charges[i]:+9.3f} {self.spin_density[i]:+9.3f}"
            )

        # compute the bond order matrix
        self.B = 2 * (self.density_up**2 + self.density_down**2)
        print("\nπ Bond order")
        print("Atom pair  Bond order")
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                if i != j:
                    print(f"{i+ 1:3} {j + 1:3}       {self.B[i, j]:.3f}")

    def plot(self, coefficients, label=None, size=4):
        """
        A function to plot molecular orbitals with given coordinates and coefficients.
        The probability of finding an electron is proportional to the area of the circles.
        """
        coordinates = [
            (atom[1][0], atom[1][1]) for atom in self.atoms
        ]  # Extract coordinates

        # Ensure coefficients is 2D (matrix-like) so we can plot single or multiple orbitals
        if np.ndim(coefficients) == 1:
            coefficients = coefficients[:, np.newaxis]

        num_plots = coefficients.shape[
            1
        ]  # Number of columns determines the number of plots

        # Compute the grid size to organize plots (e.g., 3x2 for 6 plots)
        n_cols = 2  # 1 #3 #int(np.ceil(np.sqrt(num_plots)))  # Columns in the grid
        n_rows = int(np.ceil(num_plots / n_cols))  # Rows in the grid

        # Create a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(size * n_cols, size * n_rows))
        axes = np.ravel(axes)  # Flatten the axes array for easy iteration

        # Plot each column of coefficients in its own subplot
        for i in range(num_plots):
            ax = axes[i]
            ax.set_aspect("equal", "box")

            # Set plot limits with padding
            x_vals, y_vals = zip(*coordinates)
            ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
            ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)

            # Plot molecular orbitals and atoms
            for (x, y), c in zip(coordinates, coefficients[:, i]):
                radius = np.abs(c) * 0.9
                color = "orange" if c > 0 else "blue"
                mo = plt.Circle((x, y), radius, color=color, alpha=0.5)
                ax.add_patch(mo)
                atom = plt.Circle((x, y), 0.05, color="black")
                ax.add_patch(atom)

            # Set smaller font size for axes labels
            ax.tick_params(axis="both", which="major", labelsize=8)
            if label:
                if num_plots > 1:
                    ax.set_title(f"{label} - {i+1}", fontsize=8)
                else:
                    ax.set_title(f"{label}", fontsize=8)

        # Hide any unused subplots if num_plots < n_rows * n_cols
        for j in range(num_plots, n_rows * n_cols):
            fig.delaxes(axes[j])

        # Adjust layout to avoid overlaps
        plt.tight_layout()
        plt.show()
