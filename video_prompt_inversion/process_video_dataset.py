# Import required libraries
import numpy as np  # For numerical operations
import torch  # PyTorch deep learning framework
import gc  # For garbage collection
import os  # For operating system operations
import argparse  # For parsing command line arguments
import json  # For JSON file operations
from glob import glob  # For file pattern matching
from tqdm import tqdm  # For progress bars
from longvu.builder import load_pretrained_model  # Load LongVU model
from longvu.constants import (  # Import constants
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle  # For conversation handling
from longvu.mm_datautils import (  # Data utilities
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader  # For video reading


def process_video(video_path, model, tokenizer, image_processor, question="What's the prompt to generate the video?"):
    """Process a single video and return the prediction.
    
    Args:
        video_path: Path to the video file
        model: The LongVU model
        tokenizer: Tokenizer for text processing
        image_processor: Processor for video frames
        question: Question to ask about the video
    """
    try:
        # Initialize video reader
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        
        # Sample frames at 1 frame per second
        frame_indices = np.array([i for i in range(0, len(vr), round(fps))])
        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)
        video = np.stack(video)
        
        # Process video frames
        image_sizes = [video[0].shape[:2]]
        video = process_images(video, image_processor, model.config)
        video = [item.unsqueeze(0) for item in video]

        # Prepare conversation prompt
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates["llama3"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        # Generate prediction
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        
        # Decode prediction
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Clean up memory
        del video, input_ids, output_ids
        torch.cuda.empty_cache()
        
        return pred
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return f"Error: {str(e)}"


def batch_process_videos(video_dir, model_path, output_file, question="What's the prompt to generate the video?", 
                         file_pattern="*.mp4", start_idx=0, end_idx=None, resume=False, model_loader=None):
    """Process videos in a directory and save results to a file.
    
    Args:
        video_dir: Directory containing videos
        model_path: Path to pretrained model
        output_file: Path to save results
        question: Question to ask about videos
        file_pattern: Pattern to match video files
        start_idx: Starting index for processing
        end_idx: Ending index for processing
        resume: Whether to resume from existing results
        model_loader: Tuple of (tokenizer, model, image_processor, context_len) if already loaded
    """
    
    # Get list of video paths
    video_paths = sorted(glob(os.path.join(video_dir, file_pattern)))
    
    # Set processing range
    if end_idx is None:
        end_idx = len(video_paths)
    else:
        end_idx = min(end_idx, len(video_paths))
    
    video_paths = video_paths[start_idx:end_idx]
    print(f"Processing videos {start_idx} to {end_idx-1} out of {len(video_paths)}")
    
    # Handle resuming from previous results
    results = []
    processed_videos = set()
    
    if resume and os.path.exists(output_file + '.json'):
        try:
            with open(output_file + '.json', 'r') as f:
                results = json.load(f)
                for result in results:
                    processed_videos.add(result['video_path'])
            print(f"Resuming from existing results with {len(results)} already processed videos")
        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
            results = []
    
    # Load model if not provided
    if model_loader is None:
        print(f"Loading model from {model_path}")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, "cambrian_llama"
        )
        model.eval()
    else:
        tokenizer, model, image_processor, context_len = model_loader
    
    print(f"Found {len(video_paths)} videos to process")
    
    # Process each video
    for i, video_path in enumerate(tqdm(video_paths)):
        # Skip already processed videos when resuming
        if video_path in processed_videos:
            print(f"Skipping already processed video: {video_path}")
            continue
            
        video_name = os.path.basename(video_path)
        pred = process_video(video_path, model, tokenizer, image_processor, question)
        
        # Store result
        result = {
            "video_name": video_name,
            "video_path": video_path,
            "prediction": pred
        }
        
        results.append(result)
        
        # Save intermediate results
        with open(output_file, 'a') as f:
            f.write(f"Video: {video_name}\nPath: {video_path}\nPrediction: {pred}\n\n")
        
        with open(output_file + '.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(video_paths)} videos")
    
    print(f"Processing complete for batch. Results saved to {output_file}")
    
    return results, (tokenizer, model, image_processor, context_len)


def process_in_batches(video_dir, model_path, output_file, question="What's the prompt to generate the video?", 
                      file_pattern="*.mp4", batch_size=100):
    """Process all videos in batches to better manage memory
    
    Args:
        video_dir: Directory containing videos
        model_path: Path to pretrained model
        output_file: Path to save results
        question: Question to ask about videos
        file_pattern: Pattern to match video files
        batch_size: Number of videos to process in each batch
    """
    
    # Get total video count
    video_paths = sorted(glob(os.path.join(video_dir, file_pattern)))
    total_videos = len(video_paths)
    
    print(f"Total videos to process: {total_videos}")
    
    # Load model once
    print(f"Loading model from {model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, "cambrian_llama"
    )
    model.eval()
    model_loader = (tokenizer, model, image_processor, context_len)
    
    # Process videos in batches
    all_results = []
    for start_idx in range(0, total_videos, batch_size):
        end_idx = min(start_idx + batch_size, total_videos)
        print(f"\nProcessing batch from {start_idx} to {end_idx-1}")
        
        batch_results, model_loader = batch_process_videos(
            video_dir, 
            model_path, 
            output_file, 
            question=question,
            file_pattern=file_pattern,
            start_idx=start_idx,
            end_idx=end_idx,
            resume=True,
            model_loader=model_loader
        )
        
        all_results.extend(batch_results)
    
    # Final cleanup
    tokenizer, model, image_processor, context_len = model_loader
    del model, tokenizer, image_processor
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"All processing complete. Total videos processed: {len(all_results)}")
    return all_results


def main():
    """Main function to handle command line arguments and start processing"""
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Process a directory of videos with LongVU model')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos to process')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--output_file', type=str, default="output/batch_video_results.txt", help='Path to save results')
    parser.add_argument('--question', type=str, default="What's the prompt to generate the video?", help='Question to ask about each video')
    parser.add_argument('--file_pattern', type=str, default="*.mp4", help='Pattern to match video files')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of videos to process in each batch for better memory management')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results')
    parser.add_argument('--start_idx', type=int, default=0, help='Index to start processing from (for manual resuming)')
    parser.add_argument('--end_idx', type=int, default=None, help='Index to end processing (None for all videos)')
    
    args = parser.parse_args()
    
    # Initialize output files
    if not args.resume:
        if os.path.exists(args.output_file):
            os.remove(args.output_file)
        if os.path.exists(args.output_file + '.json'):
            os.remove(args.output_file + '.json')
    
    # Process videos based on arguments
    if args.start_idx == 0 and args.end_idx is None:
        results = process_in_batches(
            args.video_dir,
            args.model_path,
            args.output_file,
            args.question,
            args.file_pattern,
            args.batch_size
        )
    else:
        # Process specific range
        results, _ = batch_process_videos(
            args.video_dir,
            args.model_path,
            args.output_file,
            args.question,
            args.file_pattern,
            args.start_idx,
            args.end_idx,
            args.resume
        )


if __name__ == "__main__":
    main() 
