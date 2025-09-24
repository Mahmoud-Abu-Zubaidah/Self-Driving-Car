import os # library to interact with the operating system
import wget # library to download files from the internet
import subprocess # library to run subprocesses
import tarfile # library to handle tar files
import shutil # library for high-level file operations


# Libraries for model training
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Variables
CUSTOM_MODEL_NAME = 'my_ssd_mobnet_tuned_more_images'  # Where the model will be saved
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'  # Pretrained model name
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'  # URL to download model
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'  # TFRecord generator script
LABEL_MAP_NAME = 'label_map.pbtxt'  # Label map file name

# Directory paths
paths = {
    'SCRIPTS_PATH': os.path.join('..', 'scripts'),
    'APIMODEL_PATH': os.path.join('..', 'models'),
    'ANNOTATION_PATH': os.path.join('annotations'),
    'IMAGE_PATH': os.path.join('images'),
    'MODEL_PATH': os.path.join('models'),
    'PRETRAINED_MODEL_PATH': os.path.join('pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join('models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('..', 'protoc')
}

# File paths
files = {
    'PIPELINE_CONFIG': os.path.join(paths['CHECKPOINT_PATH'], 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Create directories if they don't exist
for path in paths.values():
    os.makedirs(path, exist_ok=True)
print("All necessary directories are created.")

# Check if the object_detection folder exists
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    try:
        # Clone the TensorFlow models repository
        subprocess.run(
            ['git', 'clone', 'https://github.com/tensorflow/models', paths['APIMODEL_PATH']],
            check=True
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")



# Download model if on Windows (os.name == 'nt')
if os.name == 'nt':
    model_tar_gz = PRETRAINED_MODEL_NAME + '.tar.gz'
    model_tar_path = os.path.join(os.getcwd(), model_tar_gz)

    # Download the model
    if not os.path.exists(model_tar_path):
        print(f"Downloading {PRETRAINED_MODEL_NAME}...")
        wget.download(PRETRAINED_MODEL_URL, model_tar_path)
        print("\nDownload complete.")
    else:
        print("Model archive already downloaded.")

    # Move the tar.gz to the pre-trained models directory
    dest_tar_path = os.path.join(paths['PRETRAINED_MODEL_PATH'], model_tar_gz)
    os.makedirs(paths['PRETRAINED_MODEL_PATH'], exist_ok=True)
    if not os.path.exists(dest_tar_path):
        shutil.move(model_tar_path, dest_tar_path)
        print("Moved model archive to pre-trained model path.")
    else:
        print("Model archive already exists in destination.")

    # Extract the .tar.gz file
    extract_path = paths['PRETRAINED_MODEL_PATH']
    tar_file_path = os.path.join(extract_path, model_tar_gz)
    if os.path.exists(tar_file_path):
        print("Extracting model...")
        with tarfile.open(tar_file_path) as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
    else:
        print("Model archive not found for extraction.")

## Start to train the model
#1. Create `labelmap.pbtxt` for classes
labels = [{'name':'vehicle', 'id':1}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

#2. Create TF records
# Generate train.record
train_command = [
    'python',
    files['TF_RECORD_SCRIPT'],
    '-x', os.path.join(paths['IMAGE_PATH'], 'train'),
    '-l', files['LABELMAP'],
    '-o', os.path.join(paths['ANNOTATION_PATH'], 'train.record')
]

# Generate test.record
test_command = [
    'python',
    files['TF_RECORD_SCRIPT'],
    '-x', os.path.join(paths['IMAGE_PATH'], 'test'),
    '-l', files['LABELMAP'],
    '-o', os.path.join(paths['ANNOTATION_PATH'], 'test.record')
]

# Run the commands
try:
    print("Creating train.record...")
    subprocess.run(train_command, check=True)
    print("train.record created successfully.")

    print("Creating test.record...")
    subprocess.run(test_command, check=True)
    print("test.record created successfully.")

except subprocess.CalledProcessError as e:
    print(f"Error occurred while generating TFRecord files: {e}")


#3. Copy the config file from pre-trained-model to your model file.
# Source and destination paths
src_pipeline_config = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')
dst_pipeline_config = os.path.join(paths['CHECKPOINT_PATH'], 'pipeline.config')

# Ensure destination directory exists
os.makedirs(paths['CHECKPOINT_PATH'], exist_ok=True)

# Copy the config file
try:
    shutil.copy(src_pipeline_config, dst_pipeline_config)
    print(f"Copied pipeline.config to {dst_pipeline_config}")
except FileNotFoundError:
    print(f"pipeline.config not found at {src_pipeline_config}")
except Exception as e:
    print(f"Error copying pipeline.config: {e}")

#4. Update the pipeline config for transfer learning
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
#
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)
# Update fields
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 5
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
# Write the updated config back to file
config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)
# Paths (assuming you've defined `paths`, `files`, etc. already)
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

# Command string
command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(
    TRAINING_SCRIPT,
    paths['CHECKPOINT_PATH'],
    files['PIPELINE_CONFIG']
)

# Run the command
try:
    print("Starting training...")
    subprocess.run(command, shell=True, check=True)
    print("Training finished successfully.")
except subprocess.CalledProcessError as e:
    print(f"Training failed with error: {e}")

# Eval model
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(
    TRAINING_SCRIPT,
    paths['CHECKPOINT_PATH'],
    files['PIPELINE_CONFIG'],
    paths['CHECKPOINT_PATH']
)

# Run the command
try:
    print("Starting evaluation...")
    subprocess.run(command, shell=True, check=True)
    print("Evaluation finished successfully.")
except subprocess.CalledProcessError as e:
    print(f"Evaluation failed with error: {e}")