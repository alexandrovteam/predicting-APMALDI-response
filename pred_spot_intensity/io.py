import pandas as pd
from sklearn.preprocessing import PowerTransformer


def load_molecule_features(mol_features_csv_path,
                           molecule_name_column="molecule_name",
                           normalize=True):
    mol_properties = pd.read_csv(mol_features_csv_path, index_col=molecule_name_column)
    mol_properties.sort_index(inplace=True)
    mol_properties.drop_duplicates(inplace=True)

    # Validate loaded data:
    assert mol_properties.index.is_unique

    # Molecule features used for training (in the right order):
    column_names = ['pka_strongest_acidic', 'pka_strongest_basic', 'polar_surface_area',
                    'polarizability', 'acceptor_count', 'donor_count',
                    'physiological_charge']
    assert all(col in mol_properties.columns.tolist() for col in column_names)
    # Reorder:
    mol_properties = mol_properties[column_names]

    # Check for NaN values:
    null_mask_pka_acidic = mol_properties.pka_strongest_acidic.isnull()
    mol_properties.loc[null_mask_pka_acidic, "pka_strongest_acidic"] = mol_properties.pka_strongest_acidic[
        ~null_mask_pka_acidic].max()

    null_mask_pka_basic = mol_properties.pka_strongest_basic.isnull()
    mol_properties.loc[null_mask_pka_basic, "pka_strongest_basic"] = mol_properties.pka_strongest_basic[
        ~null_mask_pka_basic].min()

    # Normalize:
    if normalize:
        pt = PowerTransformer()
        mol_properties = pd.DataFrame(pt.fit_transform(mol_properties),
                                      index=mol_properties.index,
                                      columns=mol_properties.columns)

    return mol_properties
