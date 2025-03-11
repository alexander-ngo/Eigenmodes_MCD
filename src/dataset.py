import os
import subprocess
import brainspace.mesh as mesh
import nibabel as nib
import numpy as np
import pandas as pd
from lapy import Solver, TriaMesh


# Helper functions
def laplace_beltrami(tria, n_modes):
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
    idx = np.zeros([np.shape(surface_new.Points)[0], 1])
    for i in range(np.shape(surface_new.Points)[0]):
        idx[i] = np.where(
            np.all(np.equal(surface_new.Points[i, :], surface_old.Points), axis=1)
        )[0][0]
    idx = idx.astype(int)

    return idx


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
    medial = np.loadtxt(medial_fname)
    new_surface = mesh.mesh_operations.mask_points(surface, medial)

    # calculate eigenvalues and eigenmodes
    vtk_fname = (
        f"../data/tmp/{os.path.basename(surface_fname).replace('.surf.gii', '.vtk')}"
    )
    if not os.path.isfile(vtk_fname):
        vtk_fname = f"../data/tmp/{os.path.basename(surface_fname).replace('.surf.gii', '.vtk')}"
        cmd = [
            "mris_convert",
            surface_fname,
            vtk_fname,
        ]
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    tria = TriaMesh.read_vtk(vtk_fname)
    tria.v = new_surface.Points
    tria.t = np.reshape(new_surface.Polygons, [new_surface.n_cells, 4])[:, 1:4]
    evals, emodes = laplace_beltrami(tria, n_modes)

    # reshape emodes to match vertices of original surface
    idx = indices(surface, new_surface)
    emodes_reshaped = np.zeros([surface.n_points, np.shape(emodes)[1]])
    for mode in range(np.shape(emodes)[1]):
        emodes_reshaped[idx, mode] = np.expand_dims(emodes[:, mode], axis=1)

    return evals, emodes_reshaped


# Main function
def main():
    print()
    print("---------------")
    print("Loading dataset")
    print("---------------")

    print()
    print("Demographics")
    print("---------------")
    demographics = pd.read_csv(
        "../data/raw/demographics.csv", converters={"session": str}
    )
    age = demographics["age"]
    sex = demographics["sex"]
    lateralization = demographics["lateralization"]
    diagnosis = demographics["diagnosis"]

    print()
    print("Eigenmodes")
    print("---------------")
    n_modes = 200
    n_vertices = 64984
    processed = True

    eigenvalues = np.zeros([len(demographics["subject"]), n_modes])
    eigenmodes = np.zeros([len(demographics["subject"]), n_vertices, n_modes])

    # get eigenmodes for each subject
    for i, (sub, ses, group) in enumerate(
        zip(demographics["subject"], demographics["session"], demographics["group"])
    ):
        print(f"Eigenmodes processing: sub-{sub} ses-{ses}")

        for j, hemi in enumerate(["L", "R"]):
            if processed:
                # run laplace-beltrami operator
                surface_fname = f"../data/raw/{group}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii"
                medial_fname = f"../data/surfaces/fsLR-32k.{hemi}.medialwall.txt"
                evals, emodes = surface_eigenmodes(surface_fname, medial_fname, n_modes)

                # save eigenvalues and eigenmodes as files
                os.makedirs(
                    f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/",
                    exist_ok=True,
                )

                evals_fname = f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_label-eigenvalues.txt"
                np.savetxt(evals_fname, emodes)

                data = nib.gifti.gifti.GiftiImage()
                data.add_gifti_data_array(
                    nib.gifti.gifti.GiftiDataArray(data=emodes, datatype="float32")
                )
                emodes_fname = f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_hemi-{hemi}_space-fsLR-32k_label-eigenmodes.func.gii"
                nib.save(data, emodes_fname)

            else:
                evals = np.loadtxt(
                    f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/"
                    f"sub-{sub}_ses-{ses}_hemi-{hemi}_label-eigenvalues.txt"
                )
                emodes = (
                    nib.load(
                        f"../data/processed/eigenmodes/{group}/sub-{sub}/ses-{ses}/"
                        f"sub-{sub}_ses-{ses}_hemi-{hemi}_space-fsLR-32k_label-eigenmodes.func.gii"
                    )
                    .darrays[0]
                    .data
                )

            eigenvalues[i, :] = evals
            eigenmodes[i, n_vertices // 2 * (j) : n_vertices // 2 * (j + 1), :] = emodes

        print()

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
