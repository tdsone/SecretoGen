import os

checkpoint_url = 'https://sid.erda.dk/share_redirect/e9OpPDduHg/secretogen.pt'

def download_checkpoint(destination_dir: str):
    """Download the secretogen checkpoint from ERDA."""

    print(f'Downloading checkpoint to {destination_dir}...')

    os.makedirs(destination_dir, exist_ok=True)
    os.system(f'wget {checkpoint_url} -P {destination_dir}/')