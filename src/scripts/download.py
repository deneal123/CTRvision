import shutil
from pathlib import Path
from typing import Optional

import kagglehub
from utils.config_parser import ConfigParser
from utils.custom_logging import get_logger

logger = get_logger(__name__)


class DatasetDownloader:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_name = self.config['dataset_download']['dataset_name']
        self.overwrite = self.config['dataset_download']['overwrite_existing']
        self.required_files = self.config['dataset_download']['required_files']
        self._validate_config()
        
    def _validate_config(self) -> None:
        if not isinstance(self.required_files, list):
            raise ValueError("required_files must be a list")
        if not self.required_files:
            raise ValueError("required_files cannot be empty")
            
    def download(self) -> Optional[str]:
        try:
            logger.info(f"Starting download of dataset: {self.dataset_name}")
            
            cache_path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset downloaded to cache: {cache_path}")
            
            self._log_cache_contents(cache_path)
            target_path = self._get_target_path()
            
            if self._should_skip_download(target_path):
                return str(target_path)
            
            self._copy_required_files(cache_path, target_path)
            self._validate_result(target_path)
            
            logger.info(f"Dataset successfully prepared at: {target_path}")
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise

    def _get_target_path(self) -> Path:
        from src import path_to_project
        project_path = Path(path_to_project())
        return project_path / 'data' / self.dataset_name.split('/')[-1]

    def _should_skip_download(self, target_path: Path) -> bool:
        if target_path.exists():
            if not self.overwrite:
                logger.info("Dataset already exists and overwrite is disabled. Skipping.")
                return True
            logger.info("Dataset exists but will be overwritten.")
        return False

    def _log_cache_contents(self, cache_path: str) -> None:
        cache_dir = Path(cache_path)
        if cache_dir.exists():
            contents = [f.name for f in cache_dir.iterdir()]
            logger.debug(f"Cache contents: {contents}")

    def _copy_required_files(self, cache_path: str, target_path: Path) -> None:
        if target_path.exists():
            shutil.rmtree(target_path)
        target_path.mkdir(parents=True, exist_ok=True)
        
        cache_dir = Path(cache_path)
        copied_files = []
        missing_files = []
        
        for required_file in self.required_files:
            source_path = cache_dir / required_file
            
            if not source_path.exists():
                missing_files.append(required_file)
                continue
                
            dest_path = target_path / required_file
            
            try:
                if source_path.is_dir():
                    shutil.copytree(source_path, dest_path)
                    copied_files.append(f"📁 {required_file}")
                else:
                    shutil.copy2(source_path, dest_path)
                    copied_files.append(f"📄 {required_file}")
            except Exception as e:
                logger.error(f"Failed to copy {required_file}: {str(e)}")
                raise

        if copied_files:
            logger.info(f"Successfully copied:\n" + "\n".join(copied_files))
        if missing_files:
            logger.warning(f"Missing files in dataset: {missing_files}")

    def _validate_result(self, target_path: Path) -> None:
        for required_file in self.required_files:
            check_path = target_path / required_file
            if not check_path.exists():
                logger.warning(f"Validation failed: {required_file} not found in target")
            else:
                if check_path.is_dir():
                    file_count = len(list(check_path.glob('*')))
                    logger.info(f"Directory {required_file} contains {file_count} files")
                else:
                    file_size = check_path.stat().st_size
                    logger.info(f"File {required_file} size: {file_size} bytes")


def main():
    from src import path_to_config
    config = ConfigParser().parse(path_to_config())
    try:
        downloader = DatasetDownloader(config)
        result_path = downloader.download()
        logger.info(f"✅ Dataset ready at: {result_path}")
    except Exception as e:
        logger.error(f"❌ Failed to download dataset: {str(e)}")
        raise


if __name__ == "__main__":
    main()