from pathlib import Path

def get_config():
    return {
        "batch_size" : 8,
        "num_epochs" : 10,
        "lr" : 10**-4,
        "seq_len" : 350,
        "d_model" : 512,
        "datasource" : "opus_books",
        "lang_src" : "en",
        "lang_tgt" : "it",
        "model_folder" : "weights",
        "model_basename" : "transformers_model_",
        "preload" : "latest",
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/transformers_experiment"
    }

def get_weights_file(config, epoch: str):
    # get model folder
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    # get model filename
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

