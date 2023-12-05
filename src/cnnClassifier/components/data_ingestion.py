import os
import urllib.request as request
import zipfile
from src.cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnnClassifier.utils import read_yaml
from from_root import from_root

class DataIngestionConfig:
    config = read_yaml(os.path.join(from_root(), CONFIG_FILE_PATH))
    param = read_yaml(os.path.join(from_root(), PARAMS_FILE_PATH))
    source_url = config.data_ingestion.source_URL
    root_dir = os.path.join(from_root(), config.data_ingestion.root_dir)
    local_data = os.path.join(from_root(), config.data_ingestion.local_data_file)
    unzip_dir = os.path.join(from_root(), config.data_ingestion.unzip_dir)


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def download_file(self):
        # Check if the local data file exists
        if not os.path.exists(self.data_ingestion_config.local_data):
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.local_data), exist_ok=True)

            # Download the file from the source URL
            filename, headers = request.urlretrieve(
                url=self.data_ingestion_config.source_url,
                filename=self.data_ingestion_config.local_data
            )
            print(f"{filename} downloaded! with following info: \n{headers}")
        else:
            print(f"File already exists of size: {os.path.getsize(self.data_ingestion_config.local_data)} bytes")


    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.data_ingestion_config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.data_ingestion_config.local_data, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


if __name__=='__main__':
    data_ingestion = DataIngestion()
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()