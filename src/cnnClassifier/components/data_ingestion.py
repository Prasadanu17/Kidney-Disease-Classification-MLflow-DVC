import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Fetch data from the url
        '''
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?export=download&id='

            # Retry up to 3 times for unreliable network connections
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Download attempt {attempt}/{max_retries}")
                    gdown.download(prefix + file_id, str(zip_download_dir), quiet=False)

                    if os.path.exists(zip_download_dir) and os.path.getsize(zip_download_dir) > 0:
                        logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
                        return
                    else:
                        logger.warning(f"Attempt {attempt}: Download file missing or empty")
                except Exception as download_error:
                    logger.warning(f"Attempt {attempt} failed: {download_error}")
                    # Clean up partial files before retry
                    for f in os.listdir("artifacts/data_ingestion"):
                        if f.endswith(".part"):
                            os.remove(os.path.join("artifacts/data_ingestion", f))
                    if attempt == max_retries:
                        raise download_error

        except Exception as e:
            raise e
        

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
