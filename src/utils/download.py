import os

checkpoint_url = 'https://erda.ku.dk/archives/1a91453689c691c242f78268ff2fe1aa/SecretoGen/secretogen.pt'

def download_checkpoint(destination_dir: str):
    """Download the secretogen checkpoint from ERDA."""

    print(f'Downloading checkpoint to {destination_dir}...')

    os.makedirs(destination_dir, exist_ok=True)
    os.system(f'wget {checkpoint_url} -P {destination_dir}/')