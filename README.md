# ChangeFormer Core Code

This repository contains the core code for ChangeFormer. The `index_seleciton_evaluation/ChangeFormer` directory includes the model definitions and training files for ChangeFormer and the `ARQG` directory contains the code for randomly generated queries. The `index_seleciton_evaluation` comes from [Magic mirror in my hand](https://github.com/hyrise/index_selection_evaluation), and we conduct experiments with changes based on it.

## Usage

Next we describe how to use the code. The rest of the content and detailed will be gradually added to the project.

### Generate Random Queries

The settings of randomly generated queries are listed in `ARQG\settings.py`, including the following content:
- the number of queries
- the length of the sql
- the database schema ( table name, join key and attributes )

Run `ARQG\main.py` to get the randomly generated queries, and the results are stored in `ARQG\rand.sql`.

### Train the ChangeFormer

The ChangeFormer training code is located in `index_selection_evaluation/ChangeFormer/trainer.py`.

### Running index selection

```
python -m selection
```