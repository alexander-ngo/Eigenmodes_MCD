import subprocess
import brainspace.mesh as mesh
import nibabel as nib
import numpy as np
import pandas as pd
from brainspace.gradient.alignment import procrustes_alignment
from lapy import Solver, io


# Helper functions
def eigenmodes(tria, n_modes):
    """Calculate the eigenvalues and eigenmodes of a surface.

    Parameters
    ----------
    tria (vtk object):
        Surface triangular mesh
    n_modes (int):
        Number of eigenmodes to be calculated

    Returns
    ------
    evals (array): Eigenvalues
    emodes (array): Eigenmodes
    """

    # laplace-beltrami operator
    fem = Solver(tria)
    evals, emodes = fem.eigs(k=n_modes)

    return evals, emodes


def indices(surface_old, surface_new):
    """Extract indices of vertices of the two surfaces that match.

    Parameters
    ----------
    surface_old (vtk object):
        Surface triangular mesh
    surface_new (vtk ibject):
        Surface triangular mesh

    Returns
    ------
    indices (array): indices of vertices
    """

    # match vertices between old/new surface
    indices = np.zeros([np.shape(surface_new.Points)[0], 1])
    for i in range(np.shape(surface_new.Points)[0]):
        indices[i] = np.where(
            np.all(np.equal(surface_new.Points[i, :], surface_old.Points), axis=1)
        )[0][0]
    indices = indices.astype(int)

    return indices


def surface_eigenmodes(surface_fname, medial_fname, n_modes=200):
    """Calculate the eigenmodes of a cortical surface with application of a mask (e.g., to remove the medial wall).

    Parameters
    ----------
    surface_fname (str):
        Filename of input fsnative surface
    medial_fname (str):
        Filename of mask to be applied on the surface (e.g., cortex without medial wall, values = 1 for mask and 0 elsewhere)
    n_modes (int):
        Number of eigenmodes to be calculated

    Returns
    ------
    evals (array): Eigenvalues
    emodes (array): Eigenmodes
    """

    # load fsnative surface
    surface = mesh.mesh_io.read_surface(surface_fname)

    # mask medial wall
    medial = nib.load(medial_fname).darrays[0].data
    new_surface = mesh.mesh_operations.mask_points(surface, medial)

    # calculate eigenvalues and eigenmodes
    tria = io.import_vtk(surface)
    tria.v = new_surface.Points
    tria.t = np.reshape(new_surface.Polygons, [new_surface.n_cells, 4])[:, 1:4]
    evals, emodes = eigenmodes(tria, n_modes)

    # reshape emodes to match vertices of original surface
    idx = indices(surface, new_surface)
    emodes_reshaped = np.zeros([surface.n_points, np.shape(emodes)[1]])
    for mode in range(np.shape(emodes)[1]):
        emodes_reshaped[idx, mode] = np.expand_dims(emodes[:, mode], axis=1)

    return evals, emodes_reshaped


def resample(data, current_fname, target_fname):
    """Resample data, given two spherical surfaces that are in register.

    Parameters
    ----------
    data (array):
        Data in the <current> surface to be resampled
    current_fname (str):
        Filename of <current> sphere surface mesh
    target_fname (str):
        Filename of <target> sphere surface mesh

    Returns
    ------
    resampled_data (array): Resampled data in <target> surface
    """
    # save data as a temporary file
    tmp_fname = "../data/tmp/current_data.func.gii"
    nib.save(data, tmp_fname)

    # resample using workbench command
    out_fname = "../data/tmp/resampled_data.func.gii"
    cmd = [
        "wb_command",
        "-metric-resample",
        tmp_fname,
        current_fname,
        target_fname,
        "BARYCENTRIC",
        out_fname,
    ]
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if not process.returncode:
        ValueError(f"Resampling failed:\n{process.stderr}")

    return nib.load(out_fname).darrays[0].data


# Main function
def main():
    print()
    print("---------------")
    print("Loading dataset")
    print("---------------")

    print()
    print("Demographics")
    print("---------------")
    demographics = pd.read_csv("../data/raw/demographics.csv")
    age = demographics["age"]
    sex = demographics["sex"]
    lateralization = demographics["lateralization"]
    diagnosis = demographics["diagnosis"]

    print()
    print("Eigenmodes")
    print("---------------")
    n_vertices = 64984
    processed = True

    eigenvalues = np.zeros([len(demographics["sub"]), n_vertices, 200])
    eigenmodes = np.zeros([len(demographics["sub"]), n_vertices, 200])

    # generate reference eigenmodes (fsLR32k template surface)
    refence_eigenvalues, reference_eigenmodes = surface_eigenmodes(
        "../data/raw/fsLR-32k.L.sphere.reg.surf.gii",
        "../data/raw/fsLR-32k.L.medialwall.label.gii",
    )

    for i, sub, ses, group in enumerate(
        zip(demographics["sub"], demographics["ses"], demographics["group"])
    ):
        for j, hemi in enumerate(["L", "R"]):
            if processed:
                evals, emodes = surface_eigenmodes(
                    f"../data/raw/controls/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_space-nativepro_surf-fsnative_label-midthickness.surf.gii",
                    f"../datas/raw/controls/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_space-nativepro_surf-fsnative_medialwall.label.gii",
                )
                emodes_resampled = resample(
                    emodes,
                    f"../data/raw/controls/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_space-fsnative_label-sphere.surf.gii",
                    "../data/surfaces/fsLR-32k.L.sphere.reg.surf.gii",
                )

                # align eigenmodes
                emodes_aligned = procrustes_alignment(
                    emodes_resampled, reference_eigenmodes
                )

                # save eigenvalues and eigenmodes as files
                evals_fname = f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_label-eigenvalues.txt"
                emodes_fname = f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_space-fsnative_label-eigenmodes.func.gii"
                emodes_resampled_fname = f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_space-fsLR-32k_label-eigenmodes.func.gii"
                np.savetxt(evals_fname, evals)
                nib.save(emodes, emodes_fname)
                nib.save(emodes_resampled, emodes_resampled_fname)

                eigenvalues[i, n_vertices // 2 * (j) : n_vertices // 2 * (j + 1), :] = (
                    evals
                )
                eigenmodes[i, n_vertices // 2 * (j) : n_vertices // 2 * (j + 1), :] = (
                    emodes_aligned
                )

            else:
                eigenvalues[i, n_vertices // 2 * (j) : n_vertices // 2 * (j + 1), :] = (
                    np.loadtxt(
                        f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_label-eigenvalues.txt"
                    )
                )
                eigenmodes[i, n_vertices // 2 * (j) : n_vertices // 2 * (j + 1), :] = (
                    nib.load(
                        f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_space-fsLR-32k_label-eigenmodes.func.gii"
                    )
                    .darrays[0]
                    .data
                )
        print(f"Eigenmodes processed: sub-{sub} ses-{ses}")

    print()
    print("Save data")
    print("-------------------------")
    np.savez(
        "../data/processed/demographics",
        age=age,
        sex=sex,
        lateralization=lateralization,
        diagnosis=diagnosis,
    )

    np.savez(
        "../data/processed/geometric_eigemodes",
        eigenvalues=eigenvalues,
        eigenmodes=eigenmodes,
    )


if __name__ == "__main__":
    main()
