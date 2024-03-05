import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

sns.set_theme(style="ticks")
sns.set_context('paper', font_scale=1.5)


# Method for transmission matrix
def create_D_MAT(r, t):
    mat = np.array([[1, r], [r, 1]])
    return 1 / t * mat


# Method for propagation matrix
def create_P_MAT(k, L):
    mat = np.array([[np.exp(-1j * L * k), 0], [0, np.exp(L * 1j * k)]])
    return mat


def tmm(stack, df):
    # Stores the number of layers (including the air either side of the stack)
    n_layers = len(stack)
    # Arrays to store results
    w__ = []
    r__ = []
    k__ = []

    # Iterate through each wavelength
    for wavelength in df['wavelength']:
        # Arrays to store transmission and propagation matrices
        D_mats = []
        P_mats = []

        # Iterate through each layer except for the last (air)
        for i in range(n_layers - 1):
            # Identify the two materials either side of a boundary
            material1 = stack[i][0]
            material2 = stack[i + 1][0]

            # Fetch the refractive indexes for both materials
            n1 = df[material1][df['wavelength'] == wavelength].values[0]
            n2 = df[material2][df['wavelength'] == wavelength].values[0]

            # Calculate the reflection and transmission coefficients from Fresnel equations
            r = ((n1 - n2) / (n1 + n2))
            t = (2 * n1) / (n1 + n2)

            # If not the first layer (air), calculate propagation matrix
            if i >= 1:
                # Wave number in free space
                k0 = (2 * np.pi) / wavelength
                # Wave number in medium (layer)
                k = n1 * k0
                # Calculate P matrix
                P_mats.append(create_P_MAT(k, layers[i][1]))

            # Create D matrix
            D_mats.append(create_D_MAT(r, t))

        # Set the first value of the transfer matrix to the first D matrix
        M = D_mats[0]
        # Iterate through the P matrices
        for i in range(len(P_mats)):
            # Multiple transfer matrix by P matrix then D matrix
            M = M @ P_mats[i]
            M = M @ D_mats[i + 1]

        # Calculate total reflectance and transmittance of stack
        K = np.abs(1 / M[0][0]) ** 2
        R = (np.abs(M[1][0] / M[0][0])) ** 2

        # Add results to arrays
        w__.append(wavelength)
        r__.append(R)
        k__.append(K)

    # Return results as a dataframe
    results = pd.DataFrame({'wl': w__, 'R': r__, 'T': k__})
    return results

def plot_results(data, stack ):
    # Create a new figure and axis
    fig, ax1 = plt.subplots()

    # Plot the first graph on the left axis
    sns.lineplot(data, x='wl', y='R', ax=ax1, label='Reflectance')
    ax1.set_ylabel('Reflectance')  # Set the label color for the left axis
    plt.legend([], [], frameon=False)
    # Create a twin axis sharing the x-axis with the first axis
    ax2 = ax1.twinx()

    # Plot the second graph on the right axis
    sns.lineplot(data, x='wl', y='T', ax=ax2, label='Transmittance', color='red')
    ax2.set_ylabel('Transmittance')  # Set the label color for the right axis

    plt.legend([], [], frameon=False)
    # Set the scale for the right axis

    # Set labels and title
    handles = []
    for i in range(1, len(stack) - 1):
        label = '{}: {} nm'.format(stack[i][0], stack[i][1])
        handles.append(mlines.Line2D([], [], label=label, linestyle='None'))
    labels = [line.get_label() for line in handles]
    ax1.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.15, 0.5))

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_title('Reflectance and Transmittance Curves')

    line1 = mlines.Line2D([], [], label='Reflectance')
    line2 = mlines.Line2D([], [], color='red', label='Transmittance')
    handles = [line1, line2]
    labels = [line.get_label() for line in handles]

    ax2.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.subplots_adjust(right=0.5, up=0.75)
    plt.show()

materials_db = pd.read_csv('refractive-indexB.csv', index_col=0)
materials = materials_db.columns[1:]
materials_db[materials] = materials_db[materials].astype(complex)

layers = [['air', 1000], ['quartz', 5], ['glass', 25], ['indium tin oxide', 15], ['copper', 3], ['air', 100]]

results = tmm(layers, materials_db)
print(results)
plot_results(results, layers)