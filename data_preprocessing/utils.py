import pandas as pd
import plotly.express as px
from collections import Counter
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import tqdm
import os
from deepchem.splits import ScaffoldSplitter
from deepchem.data import NumpyDataset
import seaborn as sns
import matplotlib.pyplot as plt



def barplot_cuantity_comps_binary_bioactivity(df,cuantity,log_y=True):
    bioact_count = df.groupby("target")["binary_bioactivity"].value_counts()
    best_X_count = df.groupby("target")["comp_id"].count().sort_values(ascending=False).head(cuantity)
    best_X_count.name = "comp_cuantity"
    bin_bioact_count_bestX = bioact_count[best_X_count.index]
    bin_bioact_count_bestX = bin_bioact_count_bestX.reset_index()
    bin_bioact_count_bestX.binary_bioactivity = bin_bioact_count_bestX.binary_bioactivity.astype("str")
    bin_bioact_count_bestX = pd.merge(bin_bioact_count_bestX,df[["target","organism"]],how="inner",left_on="target",right_on="target").drop_duplicates()
    bin_bioact_count_bestX = pd.merge(bin_bioact_count_bestX,best_X_count,how="inner",left_on="target",right_on="target").drop_duplicates()
    fig_best_X_bbioact = px.bar(bin_bioact_count_bestX, x='target', y='count', hover_data=['organism','comp_cuantity',], color='binary_bioactivity', title='Binary Bioactivity Count per Target',
             labels={'count': 'Count', 'target_name':'Target','target': 'Id', 'binary_bioactivity': 'Binary Bioactivity', "organism": "Organism","comp_cuantity":"Total Assays"},
             category_orders={"binary_bioactivity": ["1", "0"]},
             color_discrete_map={"1": 'orange', "0": 'blue'},
             barmode='stack',
             log_y=log_y)

    return fig_best_X_bbioact

def compid_target_desduplication_binary_bioact(df):
    print(f"Datapoints antes de desduplicar {len(df)}")
    
    df.sort_values('type')

    groups = df.groupby(['comp_id', 'target'])

    counter = Counter(unmodified=0,saved=0,droped=0)

    # define a function to apply to each group
    def drop_duplicates(group,counter_obj):
        # get the unique values of binary_bioactivity in the group
        bioactivity_values = group['binary_bioactivity'].unique()

        # if there is only one unique value (no hay discordancias de bioactividad), keep the first row in the group (cualquiera)
        if len(bioactivity_values) == 1:
            counter_obj["unmodified"] += 1
            return group.head(1)

        # if there are multiple unique values, prioritize rows with Ki or Kd in the type column
        else:
            ki_or_kd_rows = group[group['type'].isin(['Ki', 'Kd'])]
            if len(ki_or_kd_rows) > 0:
                counter_obj["saved"] += 1
                return ki_or_kd_rows.head(1)
            else:
                counter_obj["droped"] += 1
                return pd.DataFrame()

    # apply the function to each group and concatenate the results
    df_clean = pd.concat([drop_duplicates(group,counter) for _, group in groups])

    # reset the index of the DataFrame
    df_clean = df_clean.reset_index(drop=True)

    print(f"Assays droped: {counter}")
    print(f"Datapoints despues de desduplicar {len(df_clean)}")

    return df_clean

def gat_tanimoto_similarity_triangle(df):
    df_internal = df.copy()

    if "fps" not in df_internal.columns:
        df_internal["ROMol"] = df_internal.loc[:,"smiles"].apply(Chem.MolFromSmiles)
        df_internal["fps"] = df_internal.loc[:,"ROMol"].apply(FingerprintMols.FingerprintMol)
        df_internal = df_internal.reset_index(drop=True)

    nfgrps = len(df_internal.loc[:,"fps"])
    fgrps = df_internal.loc[:,"fps"]

    similarities = np.empty((nfgrps, nfgrps))

    for i in tqdm.trange(1, nfgrps):

            similarity = [DataStructs.FingerprintSimilarity(fgrps[i], target_fgrps) for target_fgrps in fgrps[:i]]
            similarities[i, :i] = similarity
            similarities[:i, i] = similarity


    # Calculating similarities of molecules
    tri_lower_diag = np.tril(similarities, k=0)
    tri_lower_diag[np.triu_indices(tri_lower_diag.shape[0])] = np.nan

    similarity_triangle_df = pd.DataFrame(tri_lower_diag,index=df_internal["comp_id"],columns=df_internal["comp_id"])
    
    return similarity_triangle_df


def plot_similarity_histogram(similarity_triangle, target_name: str, export_folder="data_plots", width=6, height=4):
    flat_similarities = similarity_triangle.values.flatten()

    title = f"Similitud de {target_name}"

    # Crear el histograma con seaborn
    plt.figure(figsize=(width, height))
    sns.histplot(flat_similarities, bins=30, kde=False)
    plt.title(title)
    plt.xlabel("Tanimoto")
    plt.ylabel("Cantidad de tanimotos")

    # Guardar la imagen
    file_name = "_".join(title.lower().split(" ")) + ".png"
    path = os.path.join(export_folder, file_name)
    
    plt.savefig(path, format='png')
    plt.show()
    plt.close()

def plot_histogram_with_two_y_axes(actives, inactives, target_name: str, export_folder="data_plots", width=6, height=4):
    
    title = f"Similitud entre activos e inactivos de {target_name}"

    # Crear la figura y los dos ejes
    fig, ax1 = plt.subplots(figsize=(width, height))

    # Configurar el segundo eje Y
    ax2 = ax1.twinx()

    # Filtrar los datos por bioactividad
    bioactivity_positive = actives.dropna()
    bioactivity_negative = inactives.dropna()

    # Graficar el histograma para bioactividad positiva en el primer eje
    sns.histplot(bioactivity_positive["Tanimoto"], bins=50, ax=ax1, color="blue", kde=False, alpha=0.5, label="Active")
    ax1.set_ylabel("Cantidad (Activas)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Graficar el histograma para bioactividad negativa en el segundo eje
    sns.histplot(bioactivity_negative["Tanimoto"], bins=50, ax=ax2, color="#40c365", kde=False, alpha=0.5, label="Inactive")
    ax2.set_ylabel("Cantidad (Inactivas)", color="#40c365")
    ax2.tick_params(axis='y', labelcolor="#40c365")

    # Configurar el título y las etiquetas
    ax1.set_xlabel("Tanimoto")
    plt.title(title)

    # Crear la leyenda combinada
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles=handles_1 + handles_2, labels=labels_1 + labels_2, loc="upper right")

    # Guardar la imagen
    file_name = "_".join(title.lower().split(" ")) + ".png"
    path = os.path.join(export_folder, file_name)

    # Mostrar el gráfico
    plt.savefig(path, format='png')
    plt.show()
    plt.close()


def scaffold_dataset_splitter(df):
    df_target = df.copy()
    scaffoldsplitter = ScaffoldSplitter()
    smiles = df_target["smiles"].to_numpy()
    comp_id = df_target["comp_id"].to_numpy()
    label_bioact = df_target["binary_bioactivity"].to_numpy()
    dc_dataset_split = NumpyDataset(X=smiles,y=label_bioact,ids=smiles)
    train,valid,test = scaffoldsplitter.train_valid_test_split(dc_dataset_split,frac_train=0.64,frac_valid=0.16,frac_test=0.20,seed=123)
    df_target.loc[df_target.smiles.isin(test.ids),"data_split"] = "test"
    df_target.loc[df_target.smiles.isin(train.ids),"data_split"] = "train"
    df_target.loc[df_target.smiles.isin(valid.ids),"data_split"] = "validation"
    dataset_deepscreen = df_target[["comp_id","binary_bioactivity","data_split","smiles"]]
    dataset_deepscreen = dataset_deepscreen.rename(columns={"binary_bioactivity":"bioactivity"})
    return dataset_deepscreen