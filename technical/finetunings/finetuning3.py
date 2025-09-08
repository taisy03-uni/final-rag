import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import whoami
from tqdm import tqdm
import time
from rouge_score import rouge_scorer
from bert_score import score
import torch
import numpy as np
import gc
import math
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class ImprovedFinetuner():
    def __init__(self, max_context_length=100000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_context_length = max_context_length
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct" 
        
    def preprocess_data(self, train=True):
        """Enhanced preprocessing with better error handling and validation split"""
        if train: 
            print("Processing Training Data...")
            data = "dataset/UK-Abs/train-data/"
            data_judgement = data + "judgement/"
            data_summary = data + "summary/"
            output = "dataset/output/train/"
        else: 
            print("Processing Testing Data...")
            data = "dataset/UK-Abs/test-data/"
            data_judgement = data + "judgement/"
            data_summary = data + "summary/full/"
            output = "dataset/output/test/"
            
        os.makedirs(output, exist_ok=True)
        data = []
        skipped_files = []

        for filename in tqdm(os.listdir(data_judgement)):
            judgement_path = os.path.join(data_judgement, filename)
            summary_path = os.path.join(data_summary, filename)
            
            if os.path.exists(summary_path):
                try:
                    with open(judgement_path, 'r', encoding='utf-8') as f:
                        judgement = f.read().strip()
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = f.read().strip()
                    
                    # Basic validation
                    if len(judgement) > 100 and len(summary) > 50:
                        data.append({
                            "filename": filename, 
                            "full_text": judgement, 
                            "summary": summary,
                            "full_text_length": len(judgement),
                            "summary_length": len(summary)
                        })
                    else:
                        skipped_files.append(f"{filename}: Too short")
                        
                except Exception as e:
                    skipped_files.append(f"{filename}: {str(e)}")
            else:
                skipped_files.append(f"{filename}: No matching summary")

        # Print statistics
        num_judgement_files = len(os.listdir(data_judgement))
        print(f"Found {len(data)} valid pairs out of {num_judgement_files} judgement files")
        print(f"Skipped {len(skipped_files)} files")
        
        if data:
            lengths = [item['full_text_length'] for item in data]
            summary_lengths = [item['summary_length'] for item in data]
            print(f"Document length stats: mean={np.mean(lengths):.0f}, min={min(lengths)}, max={max(lengths)}")
            print(f"Summary length stats: mean={np.mean(summary_lengths):.0f}, min={min(summary_lengths)}, max={max(summary_lengths)}")

        # Create train/validation split for training data
        if train and len(data) > 1:
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
            
            # Save training data
            train_path = os.path.join(output, "train_data.json")
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            
            # Save validation data
            val_path = os.path.join(output, "val_data.json")
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, indent=2, ensure_ascii=False)
                
            print(f"Training data: {len(train_data)} samples saved to {train_path}")
            print(f"Validation data: {len(val_data)} samples saved to {val_path}")
        else:
            # Save test data
            output_filename = "test_data.json"
            output_path = os.path.join(output, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Test data: {len(data)} samples saved to {output_path}")
        
        # Save skipped files log
        if skipped_files:
            skipped_path = os.path.join(output, "skipped_files.txt")
            with open(skipped_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(skipped_files))
            print(f"Skipped files logged to: {skipped_path}")

        return len(data)
    
    def load_data(self, split="train"):
        """Load data with support for train/val/test splits"""
        if split == "train":
            path = "dataset/output/train/train_data.json"
        elif split == "val":
            path = "dataset/output/train/val_data.json"
        elif split == "test":
            path = "dataset/output/test/test_data.json"
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} samples from {split} split")
            return data
        except FileNotFoundError:
            print(f"No data found at {path}. Please run preprocess_data first.")
            return []
    
    def prep_data_for_training(self, data, tokenizer):
        """Enhanced data preparation with proper chunking for long documents"""
        texts = []
        skipped = 0
        
        # Calculate available tokens for input (leaving room for summary and special tokens)
        max_input_tokens = self.max_context_length - 2000 # Reserve space for summary and special tokens
        
        system_prompt = "You are a legal expert. Summarize the following legal document concisely while preserving key legal points, decisions, and reasoning."
        
        for item in tqdm(data, desc="Preparing training data"):
            try:
                # For very long documents, we need to be more strategic
                full_text = item['full_text'].strip()
                summary = item['summary'].strip()
                
                # Check if we need to chunk the input
                input_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                
                if len(input_tokens) > max_input_tokens:
                    # For training, take the first chunk that fits
                    # This preserves the beginning which often contains key case information
                    truncated_tokens = input_tokens[:max_input_tokens]
                    full_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    print(f"Truncated document {item.get('filename', 'unknown')} from {len(input_tokens)} to {len(truncated_tokens)} tokens")
                
                # Create training text in chat format for instruct model
                text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

                {full_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                {summary}<|eot_id|>"""
                
                # Verify the full text fits in context
                full_tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(full_tokens) <= self.max_context_length:
                    texts.append(text)
                else:
                    print(f"Skipping document {item.get('filename', 'unknown')}: still too long after truncation ({len(full_tokens)} tokens)")
                    skipped += 1
                    
            except Exception as e:
                print(f"Error processing item {item.get('filename', 'unknown')}: {str(e)}")
                skipped += 1
                continue
        
        print(f"Prepared {len(texts)} training examples, skipped {skipped}")
        return Dataset.from_dict({"text": texts})

    def finetune_model(self):
        """Improved fine-tuning with validation monitoring and better hyperparameters"""
        print("Starting improved fine-tuning process...")
        
        # Load tokenizer first to determine context length
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load and prepare data
        print("Loading training data...")
        train_data = self.load_data("train")
        val_data = self.load_data("val")
        
        if not train_data:
            print("No training data found. Please run preprocess_data first.")
            return
        
        print("Preparing training dataset...")
        train_dataset = self.prep_data_for_training(train_data, tokenizer)
        
        val_dataset = None
        if val_data:
            print("Preparing validation dataset...")
            val_dataset = self.prep_data_for_training(val_data, tokenizer)
        
        # Load model with proper configuration
        print(f"Loading model: {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
            use_cache=False  # Disable cache for training
        )
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        # Configure LoRA with more conservative settings
        lora_config = LoraConfig(
            r=16,                    # Increased rank for better capacity
            lora_alpha=16,           # Match with rank
            target_modules=["q_proj", "v_proj"],  # Target more modules
            lora_dropout=0.05,       # Reduced dropout
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Calculate training steps
        num_epochs = 2  # Reduced epochs to prevent overfitting
        batch_size = 1
        gradient_accumulation_steps = 8  # Effective batch size = 8
        total_steps = 500
        warmup_steps = max(1, total_steps // 10)  # 10% warmup
        
        print(f"Training configuration:")
        print(f"  - Total samples: {len(train_dataset)}")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Warmup steps: {warmup_steps}")
        
        # Enhanced training configuration
        training_args = SFTConfig(
            output_dir="./results_improved_lora",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # Learning rate and optimization
            learning_rate=2e-4,              # Much lower learning rate
            weight_decay=0.01,               # Increased weight decay
            optim="paged_adamw_8bit",        # More memory efficient
            
            # Scheduling
            lr_scheduler_type="cosine",      # Better learning rate schedule
            warmup_steps=warmup_steps,
            
            # Memory and stability
            fp16=True,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            
            # Monitoring and saving
            logging_steps=10,
            eval_strategy="steps", 
            eval_steps=50,
            save_steps=100,
            save_strategy="steps",  
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            
            # Other settings
            remove_unused_columns=False,
            group_by_length=True,
            report_to="tensorboard",
            dataset_text_field="text",
            
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora_config,
        )
        
        # Add early stopping callback if we have validation data
        if val_dataset:
            from transformers import EarlyStoppingCallback
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Start training with error handling
        try:
            print("Starting training...")
            trainer.train()
            
            # Save the final model
            save_path = "./fLARGEine_tuned_improved_lora"
            os.makedirs(save_path, exist_ok=True)
            
            print("Saving model...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            print(f"Fine-tuned model saved to: {save_path}")
            
            # Save training logs
            if trainer.state.log_history:
                with open(os.path.join(save_path, "training_logs.json"), 'w') as f:
                    json.dump(trainer.state.log_history, f, indent=2)
            
            return save_path
            
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            return None
        finally:
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_fine_tuned_model(self, model_path="./LARGEfine_tuned_improved_lora"):
        """Load the improved fine-tuned model"""
        print(f"Loading fine-tuned model from {model_path}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
            
            # Load LoRA weights
            model = PeftModel.from_pretrained(model, model_path)
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading fine-tuned model: {str(e)}")
            return None, None
    
    def load_original_model(self):
        """Load the original instruct model"""
        print(f"Loading original model: {self.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        return model, tokenizer
    
    def generate_summary(self, model, tokenizer, text, max_new_tokens=1024):
        """Enhanced summary generation with proper handling of long documents"""
        # Check available context
        max_input_tokens = self.max_context_length - max_new_tokens - 200  # Reserve space for response and special tokens
        # Chunk the text if necessary
        input_tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(input_tokens) > max_input_tokens:
            print(f"Document too long ({len(input_tokens)} tokens), truncating to {max_input_tokens}")
            input_tokens = input_tokens[:max_input_tokens]
            text = tokenizer.decode(input_tokens, skip_special_tokens=True)

        # Format for instruct model
        system_prompt = "You are a legal expert. Summarise the following legal document concisely while preserving key legal points, decisions, and reasoning."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(
            formatted_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_context_length - max_new_tokens
        )
        
        # Move to model device
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Extract only the generated tokens (after the input)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up the summary
            summary = summary.strip()

            # Remove any remaining chat template artifacts
            if summary.startswith("<|eot_id|>"):
                summary = summary.replace("<|eot_id|>", "").strip()
            
            return summary
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return ""
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def calculate_rouge_scores(self, predictions, references):
        """Enhanced ROUGE calculation with error handling"""
        if len(predictions) == 0 or len(references) == 0:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        # Filter out empty predictions
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip()]
        
        if not valid_pairs:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in valid_pairs:
            try:
                result = scorer.score(ref, pred)
                scores['rouge1'].append(result['rouge1'].fmeasure)
                scores['rouge2'].append(result['rouge2'].fmeasure)
                scores['rougeL'].append(result['rougeL'].fmeasure)
            except Exception as e:
                print(f"Error calculating ROUGE for pair: {str(e)}")
                continue
        
        return {
            'rouge1': np.mean(scores['rouge1']) if scores['rouge1'] else 0.0,
            'rouge2': np.mean(scores['rouge2']) if scores['rouge2'] else 0.0,
            'rougeL': np.mean(scores['rougeL']) if scores['rougeL'] else 0.0
        }
    
    def calculate_bert_score(self, predictions, references):
        """Enhanced BERTScore calculation"""
        # Filter out empty predictions
        valid_preds = []
        valid_refs = []
        
        for pred, ref in zip(predictions, references):
            if pred.strip():
                valid_preds.append(pred)
                valid_refs.append(ref)
        
        if not valid_preds:
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
        
        try:
            P, R, F1 = score(valid_preds, valid_refs, lang="en", verbose=False)
            return {
                'bert_precision': P.mean().item(),
                'bert_recall': R.mean().item(),
                'bert_f1': F1.mean().item()
            }
        except Exception as e:
            print(f"Error calculating BERTScore: {str(e)}")
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
    
    def evaluate(self, eval_finetuned=True, model_path="./LARGEfine_tuned_improved_lora", output_dir="evaluation_results_improved", max_samples=None):
        """Enhanced evaluation with better memory management and error handling"""
        print("Starting enhanced evaluation...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        if eval_finetuned:
            model, tokenizer = self.load_fine_tuned_model(model_path)
            model_type = "finetuned"
        else:
            model, tokenizer = self.load_original_model()
            model_type = "original"
        
        if model is None or tokenizer is None:
            print("Failed to load model. Exiting evaluation.")
            return None
        
        # Load test data
        test_data = self.load_data("test")
        if not test_data:
            print("No test data found.")
            return None
        
        # Limit samples if specified
        if max_samples and len(test_data) > max_samples:
            test_data = test_data[:max_samples]
            print(f"Limited evaluation to {max_samples} samples")
        
        print(f"Evaluating on {len(test_data)} test samples using {model_type} model")
        
        predictions = []
        references = []
        results = []
        failed_generations = 0
        
        print("Generating summaries...")
        for i, item in enumerate(tqdm(test_data)):
            try:
                # Generate summary
                predicted_summary = self.generate_summary(
                    model, tokenizer, item['full_text']
                )
                
                if predicted_summary.strip():
                    predictions.append(predicted_summary)
                    references.append(item['summary'])
                    
                    results.append({
                        'filename': item['filename'],
                        'reference_summary': item['summary'],
                        'predicted_summary': predicted_summary,
                        'original_length': item.get('full_text_length', len(item['full_text'])),
                        'reference_length': item.get('summary_length', len(item['summary'])),
                        'predicted_length': len(predicted_summary)
                    })
                else:
                    failed_generations += 1
                    print(f"Failed to generate summary for {item['filename']}")
                
                # Memory cleanup every 10 samples
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing {item['filename']}: {str(e)}")
                failed_generations += 1
                continue
        
        print(f"Successfully generated {len(predictions)} summaries")
        print(f"Failed generations: {failed_generations}")
        
        if len(predictions) == 0:
            print("No successful predictions generated.")
            return None
        
        # Calculate metrics
        print("Calculating evaluation metrics...")
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        bert_scores = self.calculate_bert_score(predictions, references)
        
        # Compile results
        evaluation_metrics = {
            'model_type': model_type,
            'num_samples_attempted': len(test_data),
            'num_successful_predictions': len(predictions),
            'num_failed_generations': failed_generations,
            'success_rate': len(predictions) / len(test_data) if test_data else 0,
            'rouge_scores': rouge_scores,
            'bert_scores': bert_scores,
            'evaluation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Model: {model_type}")
        print(f"Samples: {len(predictions)}/{len(test_data)} (success rate: {evaluation_metrics['success_rate']:.1%})")
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        print(f"BERTScore F1: {bert_scores['bert_f1']:.4f}")
        print(f"BERTScore Precision: {bert_scores['bert_precision']:.4f}")
        print(f"BERTScore Recall: {bert_scores['bert_recall']:.4f}")
        print("="*50)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        detailed_results_path = os.path.join(output_dir, f"detailed_results_{model_type}_{timestamp}.json")
        with open(detailed_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        metrics_path = os.path.join(output_dir, f"evaluation_metrics_{model_type}_{timestamp}.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to: {detailed_results_path}")
        print(f"Evaluation metrics saved to: {metrics_path}")
        
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        return evaluation_metrics
    
    def check_cuda_memory(self):
        """Enhanced memory checking with recommendations"""
        if not torch.cuda.is_available():
            print("CUDA not available")
            return False
        
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_gb = free_mem / (1024**3)
            total_gb = total_mem / (1024**3)
            used_gb = total_gb - free_gb
            
            print(f"GPU Memory: {used_gb:.1f}GB used / {free_gb:.1f}GB free / {total_gb:.1f}GB total")
            
            if free_gb < 3.0:
                print("⚠ Low GPU memory! Consider:")
                print("  - Reducing batch size")
                print("  - Using gradient checkpointing")
                print("  - Clearing CUDA cache")
                
            return free_gb > 1.0
            
        except Exception as e:
            print(f"Error checking GPU memory: {str(e)}")
            return False
    
    def generate_single_summary(self, use_finetuned=True, example_index=0, model_path="./LARGEfine_tuned_improved_lora"):
        """Enhanced single summary generation with better error handling"""
        print("=" * 80)
        print("ENHANCED SINGLE SUMMARY GENERATION")
        print("=" * 80)
        
        try:
            # Check memory first
            if not self.check_cuda_memory():
                print("Insufficient GPU memory for generation")
                return None
            
            # Load test data
            test_data = self.load_data("test")
            if not test_data or example_index >= len(test_data):
                print(f"Invalid example index {example_index} or no test data")
                return None
            
            example = test_data[example_index]
            print(f"Processing: {example['filename']}")
            print(f"Document length: {len(example['full_text']):,} characters")
            print(f"Reference summary length: {len(example['summary']):,} characters")
            
            # Load model
            if use_finetuned:
                print("\nLoading fine-tuned model...")
                model, tokenizer = self.load_fine_tuned_model(model_path)
                model_type = "Fine-tuned"
            else:
                print("\nLoading original model...")
                model, tokenizer = self.load_original_model()
                model_type = "Original"
            
            if model is None or tokenizer is None:
                print("Failed to load model")
                return None
            
            print("\n" + "="*60)
            print("REFERENCE SUMMARY:")
            print("="*60)
            print(example['summary'])
            
            print(f"\nGenerating summary using {model_type} model...")
            start_time = time.time()
            
            predicted_summary = self.generate_summary(
                model, tokenizer, example['full_text']
            )
            
            generation_time = time.time() - start_time
            
            print(f"\n" + "="*60)
            print(f"GENERATED SUMMARY ({model_type} Model):")
            print("="*60)
            print(predicted_summary)
            
            print(f"\n" + "="*60)
            print("GENERATION INFO:")
            print("="*60)
            print(f"Generation time: {generation_time:.2f} seconds")
            print(f"Generated length: {len(predicted_summary):,} characters")
            
            # Calculate metrics
            if predicted_summary.strip():
                rouge_scores = self.calculate_rouge_scores([predicted_summary], [example['summary']])
                bert_scores = self.calculate_bert_score([predicted_summary], [example['summary']])
                
                print(f"\n" + "="*60)
                print("EVALUATION METRICS:")
                print("="*60)
                print(f"ROUGE-1 F1: {rouge_scores['rouge1']:.4f}")
                print(f"ROUGE-2 F1: {rouge_scores['rouge2']:.4f}")
                print(f"ROUGE-L F1: {rouge_scores['rougeL']:.4f}")
                print(f"BERTScore F1: {bert_scores['bert_f1']:.4f}")
                print(f"BERTScore Precision: {bert_scores['bert_precision']:.4f}")
                print(f"BERTScore Recall: {bert_scores['bert_recall']:.4f}")
            else:
                print("\nNo summary generated - cannot calculate metrics")
                rouge_scores = None
                bert_scores = None
            
            # Prepare result
            result = {
                'filename': example['filename'],
                'model_type': model_type,
                'example_index': example_index,
                'original_text': example['full_text'],
                'reference_summary': example['summary'],
                'generated_summary': predicted_summary,
                'original_text_length': len(example['full_text']),
                'reference_summary_length': len(example['summary']),
                'generated_summary_length': len(predicted_summary),
                'generation_time_seconds': generation_time,
                'rouge_scores': rouge_scores,
                'bert_scores': bert_scores,
                'generation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save result
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"single_summary_{model_type.lower()}_{timestamp}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\nResult saved to: {output_filename}")
            return result
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
            print("\nMemory cleanup completed")


if __name__ == "__main__":
    # Initialize the improved finetuner
    finetuner = ImprovedFinetuner(max_context_length=25000)
    
    # Example usage:
    
    # 1. Preprocess data (uncomment to run)
    #print("\n1. Preprocessing data...")
    #finetuner.preprocess_data(train=True)
    #finetuner.preprocess_data(train=False)
    
    # 2. Fine-tune the model (uncomment to run)
    print("\n2. Starting fine-tuning...")
    model_path = finetuner.finetune_model()
    
    # 3. Evaluate original model
    print("\n3. Evaluating original model...")
    original_results = finetuner.evaluate(eval_finetuned=False, max_samples=100)
    
    # 4. Evaluate fine-tuned model
    print("\n4. Evaluating fine-tuned model...")
    finetuned_results = finetuner.evaluate(eval_finetuned=True, max_samples=100)
    
    # 5. Generate single example
    print("\n5. Generating single summary example...")
    result = finetuner.generate_single_summary(
        use_finetuned=True,  # Set to True to use fine-tuned model
        example_index=0
    )
    