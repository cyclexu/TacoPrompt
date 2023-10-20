# TacoPrompt

The source code used for self-supervised taxonomy completion method TacoPrompt, published in EMNLP 2023.

## Data Preparation

### For datasets used in the paper


You can download the three datasets we used in the paper from 
[Datasets](https://drive.google.com/drive/folders/150CJsXvxzPEIUy011FQhwVJyJ3jIhoHR?usp=drive_link)
and put them under `data/` for the experiments.


Each of the dataset folders <TAXO_DIR> contain at least the following files:

1. ``<TAXO_NAME>.terms``. Each line in this file represents one term / concept / node in the taxonomy, 
   including its TAXON_ID and Surface Name.

    ```
    taxon1_id \t taxon1_surface_name
    taxon2_id \t taxon2_surface_name
    taxon3_id \t taxon3_surface_name
    ...
    ```

2. ``<TAXO_NAME>.taxo``. Each line in this file represents one relation / edge in the taxonomy, including the Parent TAXON_ID and Child TAXON_ID.

    ```
    parent_taxon1_id \t child_taxon1_id
    parent_taxon2_id \t child_taxon2_id
    parent_taxon3_id \t child_taxon3_id
    ...
    ```

3. ``<TAXO_NAME>.desc``. Each line in this file represents the extracted term description of each term, mapped to each term's Surface Name.

    ```
    taxon1_surface_name \t taxon1_description
    taxon2_surface_name \t taxon2_description
    taxon3_surface_name \t taxon3_description
    ...
    ```

4. ``<TAXO_NAME>.pickle.bin``. This file is created by ``dataloader/dataset`` when encountering a raw dataset. The pickled dataset contains taxonomy information as well as train/val/test splits.

### For your own taxonomy

* Step 1: Organize your input taxonomy along with node features into the format of ``<TAXO_NAME>.terms``, ``<TAXO_NAME>.taxo`` and ``<TAXO_NAME>.desc`` mentioned in the previous section.

* Step 2:  Truncate long concept description to the fixed token length to satisfy the max sequence length limitation of the backbone LM.

    ```
    python truncate_desc.py -d <TAXO_DIR> -n <TAXO_NAME>
    ```

* Step 3: (Optional) Generate the train/val/test splits in files called ``<TAXO_NAME>.terms.train``, ``<TAXO_NAME>.terms.validation`` and ``<TAXO_NAME>.terms.test``.

* Step 4: Run the training script by setting the argument `raw` for the class `MAGDataset` in `data_loader/dataset.py` to `True`. After training once, the script will generate the pickled dataset and `raw` can be set to `False` for future experiments.

## Model Training

For reproducing the results in the paper, please use the provided configs in `config_files/<TAXO_NAME>/`.

```
python train.py --config config_files/<TAXO_NAME>/config.test.chain.json
python train.py --config config_files/<TAXO_NAME>/config.test.hidden.json
```

For running TacoPrompt for your own taxonomy, please create a config file similar to the ones provided in the corresponding folder, and run the experiments with the above script.


## Model Inference
We provide inference scripts in `inference_scripts/`.

For end-to-end inference, please use the script `infer_chain.py` or `infer_hidden.py`.
```
python infer_chain.py  --config <CONFIG_PATH> -m <MODEL_PATH>
python infer_hidden.py --config <CONFIG_PATH> -m <MODEL_PATH>
```

For retrieval and reranking inference, please use the script `infer_chain_topk.py` or `infer_hidden_topk.py`.
```
python infer_chain_topk.py  --config <CONFIG_PATH> -m <MODEL_PATH> -rl <RETRIEVAL_LIST_PATH>
python infer_hidden_topk.py --config <CONFIG_PATH> -m <MODEL_PATH> -rl <RETRIEVAL_LIST_PATH>
```
The retrieval list used in the paper is provided in `inference_scripts/`.

You can construct your own retrieval list in this format:
```
{query_id: [(parent_id, child_id), (parent_id, child_id), ...]}
```

Our trained models are available [here](https://drive.google.com/drive/folders/1h7fj7GwlCMfbrT5-IDiqF8pm8jbQE30_?usp=drive_link).

## Model Organization

For all implementations, we follow the project organization in [pytorch-template](https://github.com/victoresque/pytorch-template).