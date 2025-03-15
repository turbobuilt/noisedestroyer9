import logging
from multiprocessing import queues
import time
import pedalboard
import glob
import multiprocessing as mp
import concurrent.futures
from pedalboard import Pedalboard, Chorus, Reverb
from pedalboard.io import AudioFile, ResampledReadableAudioFile
import random
import numpy as np
import os
import gc
import boto3
import tempfile
from botocore.client import Config
from dotenv import load_dotenv

# Configure logging - set this to True for verbose logging
VERBOSE_LOGGING = False

# Set up logging configuration
logging_level = logging.DEBUG if VERBOSE_LOGGING else logging.INFO
# disable completely
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("noisedestroyer")

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# Cloud storage configuration from environment variables
endpoint_url = os.getenv('endpoint_url')
access_key_id = os.getenv('access_key_id')
secret_access_key = os.getenv('secret_access_key')
bucket_name = os.getenv('bucket_name')

# Initialize S3 client for Cloudflare R2
s3_client = boto3.client(
    's3',
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    config=Config(signature_version='s3v4')
)
logger.info(f"S3 client initialized with endpoint: {endpoint_url}")

# Define cloud paths and create local cache directory for temp files
speech_cloud_prefix = 'archive_files/'  # Contains MP3 files
noise_cloud_prefix = 'noise_files/'      # Contains OGG files
os.makedirs('temp_cache', exist_ok=True)
logger.info("Created temp_cache directory")

# For fallback to local files if needed - get all files directly
local_speech_files = glob.glob('./speech/**/*.mp3', recursive=True)
local_noise_files = glob.glob('./noise/**/*.ogg', recursive=True)
logger.info(f"Found {len(local_speech_files)} local speech files and {len(local_noise_files)} local noise files")

# Function to list files from cloud storage
def list_cloud_files(prefix):
    logger.debug(f"Listing cloud files with prefix: {prefix}")
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            file_list = [item['Key'] for item in response['Contents'] 
                   if not item['Key'].endswith('/')]  # Skip directory entries
            logger.debug(f"Found {len(file_list)} cloud files with prefix {prefix}")
            return file_list
        logger.warning(f"No contents found in bucket {bucket_name} with prefix {prefix}")
        return []
    except Exception as e:
        logger.warning(f"Error listing cloud files: {e}")
        return []

# Cache the file lists to avoid repeated API calls
speech_cloud_files = list_cloud_files(speech_cloud_prefix)
noise_cloud_files = list_cloud_files(noise_cloud_prefix)
logger.info(f"Cached {len(speech_cloud_files)} speech files and {len(noise_cloud_files)} noise files from cloud")

def get_cloud_file(cloud_path, local_temp_path=None):
    """Download a file from cloud storage to a temporary local file"""
    logger.debug(f"Attempting to download cloud file: {cloud_path}")
    if local_temp_path is None:
        fd, local_temp_path = tempfile.mkstemp(dir='temp_cache')
        os.close(fd)
        logger.debug(f"Created temporary file: {local_temp_path}")
    
    try:
        s3_client.download_file(bucket_name, cloud_path, local_temp_path)
        logger.debug(f"Successfully downloaded {cloud_path} to {local_temp_path}")
        return local_temp_path
    except Exception as e:
        logger.warning(f"Error downloading {cloud_path}: {e}")
        return None

def get_clean_file(lock, sample_rate, segment_length, max_examples_per_file):
    end_safe_space = 20*sample_rate
    temp_file = None
    
    try:
        while True:
            # Try cloud files first, fall back to local if needed
            if speech_cloud_files:
                file_path = speech_cloud_files[random.randint(0, len(speech_cloud_files)-1)]
                logger.debug(f"Selected cloud speech file: {file_path}")
                temp_file = get_cloud_file(file_path)
                if temp_file is None:
                    logger.debug(f"Failed to download {file_path}, trying another file")
                    continue  # Try another file
            else:
                # Fallback to local files - use direct file paths
                if not local_speech_files:
                    logger.warning("No local speech files found")
                    time.sleep(1)
                    continue
                    
                file_path = local_speech_files[random.randint(0, len(local_speech_files)-1)]
                logger.debug(f"Selected local speech file: {file_path}")
                temp_file = file_path

            try:
                with AudioFile(temp_file) as raw_file:
                    logger.debug(f"Opened speech file: {file_path}, samplerate: {raw_file.samplerate}")
                    with ResampledReadableAudioFile(raw_file, sample_rate) as file:
                        if file.frames < segment_length + end_safe_space:
                            logger.debug(f"File too short ({file.frames} frames), skipping")
                            continue
                        segments_this_file = min((file.frames - end_safe_space) // segment_length, max_examples_per_file)
                        logger.debug(f"Will extract {segments_this_file} segments from file")

                        # select a chunk that is sample_rate*segment_length frames long, starting at any random place that leaves end_safe_space seconds at the end
                        start = random.randint(0, file.frames - segment_length*segments_this_file - end_safe_space)
                        file.seek(start)
                        logger.debug(f"Reading {segment_length*segments_this_file} frames starting from position {start}")
                        data = file.read(segment_length*segments_this_file)
                        # convert to mono, currently (channels, samples), need (samples)
                        data = data.mean(axis=0)

                        logger.debug(f"Successfully read and processed speech file")
                        return data*1.3
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                continue
    finally:
        # Clean up temporary file if it was created
        if temp_file and temp_file.startswith('temp_cache') and os.path.exists(temp_file):
            logger.debug(f"Removing temporary file: {temp_file}")
            os.remove(temp_file)

def augment_noise_file(data):
    # randomly vary volume between .1 and 2
    scale = random.uniform(.1, .9)
    logger.debug(f"Augmenting noise with volume scale factor: {scale:.2f}")
    data = data*scale
    return data


def get_noise_background(lock, segment_length):
    samples = []
    total_read = 0
    temp_file = None
    
    try:
        while True:
            # Try cloud files first, fall back to local if needed
            if noise_cloud_files:
                file_path = noise_cloud_files[random.randint(0, len(noise_cloud_files)-1)]
                logger.debug(f"Selected cloud noise file: {file_path}")
                temp_file = get_cloud_file(file_path)
                if temp_file is None:
                    continue  # Try another file
            else:
                # Fallback to local files
                if not local_noise_files:
                    logger.warning("No local noise files found")
                    time.sleep(1)
                    continue
                    
                file_path = local_noise_files[random.randint(0, len(local_noise_files)-1)]
                logger.debug(f"Selected local noise file: {file_path}")
                temp_file = file_path

            try:
                with AudioFile(temp_file) as raw_file:
                    # random number between .5 and 2 to scale
                    scale = random.uniform(.5, 1.1)
                    sample_rate = raw_file.samplerate*scale
                    logger.debug(f"Opened noise file: {file_path}, original samplerate: {raw_file.samplerate}, scaled to: {sample_rate}")
                    
                    with ResampledReadableAudioFile(raw_file, sample_rate) as file:
                        # select a chunk that is sample_rate*segment_length frames long, starting at any random place that leaves end_safe_space seconds at the end
                        amount_to_select = min(segment_length, file.frames)
                        if file.frames <= 0:
                            logger.warning(f"Empty noise file: {file_path}")
                            continue
                        start = random.randint(0, max(0, file.frames - amount_to_select - 1))
                        file.seek(start)
                        logger.debug(f"Reading {amount_to_select} frames from position {start}")
                        data = file.read(amount_to_select)
                        if data.shape[0] == 0:  # No channels
                            logger.warning("No channels in noise file")
                            continue
                        data = data.mean(axis=0)
                        logger.debug(f"Converted noise to mono, shape: {data.shape}")

                        # now cut out quiet parts - parts where the volume is averages less than .05 for 100 samples
                        size = 100
                        # Handle cases where data is shorter than size
                        if len(data) < size:
                            logger.debug(f"Noise segment too short ({len(data)} < {size}), using as is")
                            samples.append(data)
                            total_read += len(data)
                            if total_read >= segment_length:
                                break
                            continue
                            
                        # cut to multiple of size
                        data = data[:-(data.shape[0] % size)] if data.shape[0] % size != 0 else data
                        data = data.reshape(-1, size)
                        new_data = []
                        filtered_segments = 0
                        for i in range(data.shape[0]):
                            if np.mean(np.abs(data[i])) > .02:
                                new_data.append(data[i])
                            else:
                                filtered_segments += 1
                        
                        logger.debug(f"Filtered out {filtered_segments} quiet segments from noise")
                        if len(new_data) == 0:
                            logger.warning("All noise segments were too quiet, skipping file")
                            continue
                        data = np.concatenate(new_data)

                        samples.append(data)
                        total_read += len(data)
                        logger.debug(f"Added {len(data)} frames, total: {total_read}/{segment_length}")
                        if total_read >= segment_length:
                            break
                
                # Clean up temporary file after successful use
                if temp_file and temp_file.startswith('temp_cache') and os.path.exists(temp_file):
                    logger.debug(f"Removing temporary file: {temp_file}")
                    os.remove(temp_file)
                    temp_file = None
            
            except Exception as e:
                logger.warning(f"Error opening noise file {file_path}: {e}")
                # Don't delete the file here, as it might be a cloud path issue
                continue
    finally:
        # Make sure temp file is cleaned up
        if temp_file and temp_file.startswith('temp_cache') and os.path.exists(temp_file):
            logger.debug(f"Removing temporary file: {temp_file}")
            os.remove(temp_file)
            
    # Ensure we have enough samples or pad if needed
    if len(samples) == 0:
        # Return silence if we couldn't get any samples
        logger.warning("Couldn't get any noise samples, returning silence")
        return np.zeros(segment_length)
        
    data = np.concatenate(samples)
    # Pad or trim to exact segment_length
    if len(data) < segment_length:
        logger.debug(f"Padding noise data from {len(data)} to {segment_length}")
        return np.pad(data, (0, segment_length - len(data)), 'wrap')
    
    logger.debug(f"Trimming noise data from {len(data)} to {segment_length}")
    return data[:segment_length]


def get_audio_samples(lock, done_flag, output_queue: mp.Queue, sample_rate, segment_length, max_examples_per_file):
    i = 0
    worker_id = mp.current_process().name
    logger.info(f"Worker {worker_id} started")
    
    while True:
        if done_flag.value == 1:
            logger.info(f"Worker {worker_id}: Done flag set, exiting worker process")
            return
            
        parent = mp.parent_process()
        if parent is None or not parent.is_alive():
            logger.info(f"Worker {worker_id}: Parent process is no longer alive, exiting worker process")
            return

        try:
            logger.debug(f"Worker {worker_id}: Getting clean file sample #{i+1}")
            clean_file = get_clean_file(lock, sample_rate, segment_length, max_examples_per_file)
            if clean_file is None:
                logger.warning(f"Worker {worker_id}: Failed to get clean file, retrying...")
                time.sleep(1)
                continue
                
            if i % 1 == 0:  # speech + noise for now
                logger.debug(f"Worker {worker_id}: Getting noise background for sample #{i+1}")
                noise_file = get_noise_background(lock, clean_file.shape[0])
                noise_file = augment_noise_file(noise_file)
                logger.debug(f"Worker {worker_id}: Creating mixed audio sample")
                x = clean_file + noise_file*.4
                y = clean_file
                
                # Reshape into segments
                x = x.reshape(-1, segment_length)
                y = y.reshape(-1, segment_length)
                
                logger.debug(f"Worker {worker_id}: Created {x.shape[0]} segments for sample #{i+1}")
                for j in range(x.shape[0]):
                    while True:
                        try:
                            logger.debug(f"Worker {worker_id}: Putting segment {j+1}/{x.shape[0]} in queue")
                            output_queue.put((x[j].copy(), y[j].copy()), block=True, timeout=1)
                            break
                        except queues.Full:
                            if done_flag.value == 1:
                                logger.info(f"Worker {worker_id}: Done flag set while waiting for queue space, exiting")
                                return
                                
                            parent = mp.parent_process()
                            if parent is None or not parent.is_alive():
                                logger.info(f"Worker {worker_id}: Parent process died while waiting for queue space")
                                return
                                
                            logger.debug(f"Worker {worker_id}: Queue full, waiting...")
                            time.sleep(0.1)  # Shorter sleep to be more responsive
            i += 1
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error in worker process: {e}")
            time.sleep(1)

class DataIterator():
    def __init__(self, sample_rate=16000, segment_length=2**18, max_examples_per_file=5, queue_depth=5, num_workers=2, step=0, batch_size=1):
        self.sample_rate = sample_rate
        self.max_examples_per_file = max_examples_per_file
        self.num_workers = num_workers
        self.segment_length = segment_length
        self.pool = None
        self.workers = []
        self.manager = None
        self.lock = None
        self.done_flag = None
        self.step = mp.Value('i', step)
        self.queue_depth = queue_depth
        self.output_queue = None
        self.batch_size = batch_size
        logger.info(f"DataIterator initialized with {num_workers} workers, queue depth {queue_depth}")
        logger.debug(f"Sample rate: {sample_rate}, segment length: {segment_length}, max examples per file: {max_examples_per_file}, batch size: {batch_size}")

    def start_workers(self):
        if self.pool is not None:
            logger.debug("Clearing existing workers before starting new ones")
            self.clear()
            
        logger.info("Starting worker processes")
        self.manager = mp.Manager()
        self.lock = self.manager.Lock()
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
        self.workers = []
        self.output_queue = self.manager.Queue(self.queue_depth)
        self.done_flag = self.manager.Value('i', 0)
        
        for i in range(self.num_workers):
            logger.debug(f"Submitting worker {i+1}")
            worker = self.pool.submit(get_audio_samples, self.lock, self.done_flag, self.output_queue, self.sample_rate, self.segment_length, self.max_examples_per_file)
            self.workers.append(worker)
        logger.info(f"{self.num_workers} worker processes started")

    def clear(self):
        if self.pool is not None:
            logger.info("Clearing worker processes")
            if self.done_flag is not None:
                logger.debug("Setting done flag to terminate workers")
                self.done_flag.value = 1
            logger.debug("Shutting down process pool")
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
            self.workers = []
            self.lock = None
            self.manager = None
            self.output_queue = None
            # Force garbage collection to free resources
            logger.debug("Running garbage collection")
            gc.collect()
            logger.info("Worker processes cleared")

    def __iter__(self):
        return self
    
    def __next__(self):
        x_batch = []
        y_batch = []
        
        while len(x_batch) < self.batch_size:
            if self.pool is None:
                logger.debug("No worker pool, starting workers")
                self.start_workers()
            try:
                logger.debug("Waiting for next sample from queue")
                x_sample, y_sample = self.output_queue.get(True, 10)
                logger.debug("Got sample from queue")
                
                # Reshape samples to [length, channels] - adding channel dimension
                x_sample = x_sample.reshape(-1, 1)  # [length, 1]
                y_sample = y_sample.reshape(-1, 1)  # [length, 1]
                
                x_batch.append(x_sample)
                y_batch.append(y_sample)
            except queues.Empty:
                logger.info("Queue empty, waiting for more samples")
                time.sleep(.5)
                continue
            except Exception as e:
                logger.error(f"Error getting next sample: {e}")
                time.sleep(.5)
                continue
        
        # Stack the samples to create batches with shape [batch_size, length, channels]
        x_stacked = np.stack(x_batch, axis=0)
        y_stacked = np.stack(y_batch, axis=0)
        
        logger.debug(f"Returning batch shapes: x={x_stacked.shape}, y={y_stacked.shape}")
        return x_stacked, y_stacked
    
    def __len__(self):
        if speech_cloud_files:
            return len(speech_cloud_files)
        return len(local_speech_files)
    
    def __del__(self):
        logger.info("DataIterator being deleted")
        self.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("DataIterator exiting context")
        self.clear()
    
if __name__ == "__main__":
    sample_rate = 16000
    sample_length = 2**18
    
    # Create a sampling folder
    sample_dir = "audio_samples"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        logger.info(f"Created directory: {sample_dir}")
    else:
        # Clear existing sample files
        for file in glob.glob(f"{sample_dir}/*"):
            logger.debug(f"Removing existing sample file: {file}")
            os.remove(file)
        logger.info(f"Cleared existing files in {sample_dir}")
    
    print(f"Found {len(speech_cloud_files)} speech files in cloud")
    print(f"Found {len(noise_cloud_files)} noise files in cloud")
    print(f"Found {len(local_speech_files)} local speech files (MP3)")
    print(f"Found {len(local_noise_files)} local noise files (OGG)")
    
    # Create a data iterator with batch_size=1 for this demo
    logger.info("Creating data iterator")
    data = DataIterator(sample_rate=sample_rate, segment_length=sample_length, 
                       max_examples_per_file=3, queue_depth=5, num_workers=2, batch_size=1)
    
    # Generate 3 samples
    for i in range(3):
        try:
            logger.info(f"Generating sample {i+1}/3")
            # Get a sample pair
            x, y = next(data)  # Now shape is [batch, length, channels]
            logger.debug(f"Sample {i+1}: x shape {x.shape}, y shape {y.shape}")
            # print(f"Sample {i+1}: x shape {x.shape}, y shape {y.shape}")
            
            # Calculate the noise component
            noise = x - y
            
            logger.info(f"Saving clean file for sample {i+1}")
            with AudioFile(f"{sample_dir}/sample{i+1}_clean.mp3", 'w', sample_rate, 1) as o:
                # AudioFile expects [channels, length]
                o.write(y[0].reshape(1, -1))  # First batch item
            
            logger.info(f"Saving noise file for sample {i+1}")
            with AudioFile(f"{sample_dir}/sample{i+1}_noise.mp3", 'w', sample_rate, 1) as o:
                o.write(noise[0].reshape(1, -1))
            
            logger.info(f"Saving mixed file for sample {i+1}")
            with AudioFile(f"{sample_dir}/sample{i+1}_mixed.mp3", 'w', sample_rate, 1) as o:
                o.write(x[0].reshape(1, -1))
        except Exception as e:
            logger.error(f"Error generating sample {i+1}: {e}")
            print(f"Error generating sample {i+1}: {e}")
    
    # Clean up
    logger.info("Cleanup: shutting down data iterator")
    data.clear()
    print(f"Done! 3 samples saved to {sample_dir}/")
    logger.info(f"Processing complete. 3 samples saved to {sample_dir}/")