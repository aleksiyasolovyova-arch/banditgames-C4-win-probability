import time
import requests
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from src.train_integrated import train_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Watcher")


def main():
    load_dotenv()

    data_dir = Path(os.getenv("DATA_DIR", "/workspace/datasets"))
    models_dir = Path(os.getenv("MODELS_DIR", "/workspace/models"))
    deploy_api_url = os.getenv("DEPLOY_API_URL", "").strip()
    self_play = os.getenv("SELF_PLAY", "true").lower() == "true"

    logger.info(f"Watching {data_dir} for new parquet files...")
    processed_files = set()

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            files = sorted(list(data_dir.glob("*.parquet")))

            for file_path in files:
                if file_path.name in processed_files:
                    continue

                logger.info(f"Detected new dataset: {file_path.name}")

                # dataset_v1.parquet -> v1
                version = file_path.stem.split("_")[-1]

                results = train_job(
                    dataset_path=str(file_path),
                    output_dir=str(models_dir),
                    version=version,
                    self_play=self_play
                )

                if results is not None and deploy_api_url:
                    try:
                        requests.post(deploy_api_url, json={"version": version}, timeout=5)
                        logger.info(f"Deployment signal sent for {version}")
                    except Exception as e:
                        logger.error(f"Deployment signal failed: {e}")

                processed_files.add(file_path.name)

            time.sleep(10)

        except Exception as e:
            logger.error(f"Watcher loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
