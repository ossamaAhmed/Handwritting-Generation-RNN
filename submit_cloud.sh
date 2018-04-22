now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="unconditional_model_$now"
OUTPUT_PATH=gs://handwritting-generation-ml/output/
INPUT_PATH=gs://handwritting-generation-ml/input/
gcloud ml-engine jobs submit training $JOB_NAME --package-path train --module-name train.conditional_training --staging-bucket gs://handwritting-generation-ml --job-dir gs://handwritting-generation-ml/output --region us-east1 --config config.yaml --runtime-version=1.6 -- --data_dir="${INPUT_PATH}" --output_dir="${OUTPUT_PATH}"