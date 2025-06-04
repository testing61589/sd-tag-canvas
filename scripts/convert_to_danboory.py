#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import fnmatch
import re
import shutil
import csv
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional, Union
from PIL import Image
import onnxruntime
import numpy as np
import torch
from tqdm import tqdm
import huggingface_hub

from modules.module.BaseImageCaptionModel import BaseImageCaptionModel, CaptionSample
from scripts.tag_mappings import (
    MAPPING,
    COMBO_MAPPING,
    PARTIAL_MAPPING,
    PARTIAL_EXCEPTIONS,
    REMOVE_MAPPING
)


class WDModel(BaseImageCaptionModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        model_path = huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2", "model.onnx"
        )
        provider = ("CUDAExecutionProvider" 
                   if device.type == "cuda" and "CUDAExecutionProvider" in onnxruntime.get_available_providers() 
                   else "CPUExecutionProvider")
        self.model = onnxruntime.InferenceSession(model_path, providers=[provider])

        label_path = huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2", "selected_tags.csv"
        )
        self.tag_names = []
        self.general_indexes = []
        with open(label_path, newline='') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if row["category"] == "0":
                    self.general_indexes.append(i)
                self.tag_names.append(row["name"])

    def generate_caption(self, caption_sample: CaptionSample, 
                        initial_caption: str = "", caption_prefix: str = "", 
                        caption_postfix: str = "") -> str:
        _, height, width, _ = self.model.get_inputs()[0].shape
        image = caption_sample.get_image().resize((width, height)).convert("RGB")
        image = np.asarray(image)[:, :, ::-1].astype(np.float32)  # RGB -> BGR
        image = np.expand_dims(image, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image})[0][0].astype(float)

        general_labels = [(self.tag_names[i], probs[i]) for i in self.general_indexes if probs[i] > 0.35]
        sorted_general_labels = sorted(general_labels, key=lambda x: x[1], reverse=True)
        predicted_caption = ", ".join(label.replace("_", " ") for label, _ in sorted_general_labels)

        return (caption_prefix + predicted_caption + caption_postfix).strip()


class SimpleCaptionSample(CaptionSample):
    def __init__(self, image_path: str):
        self.image_path = image_path
    
    def get_image(self):
        return Image.open(self.image_path).convert("RGB")


class TagProcessor:
    """Handles all tag processing operations"""
    
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    
    def __init__(self, mapping: Dict, combo_mapping: Dict, partial_mapping: Dict, 
                 partial_exceptions: Dict, remove_mapping: Set):
        self.mapping = mapping
        self.combo_mapping = combo_mapping
        self.partial_mapping = partial_mapping
        self.partial_exceptions = partial_exceptions
        self.remove_mapping = remove_mapping

    @staticmethod
    def normalize_token(token: str) -> str:
        """Replace newlines with spaces, collapse multiple spaces, and convert to lowercase."""
        return ' '.join(token.replace('\n', ' ').replace('\r', ' ').split()).lower()

    @staticmethod
    def canonical(token: str) -> str:
        """Return canonical form of token for comparison."""
        return re.sub(r'[^a-z]', '', token.lower())

    @staticmethod
    def sanitize_token(token: str) -> str:
        """Sanitize a token by removing BREAK markers and colon-number sequences."""
        token = token.replace("BREAK", ",")
        token = re.sub(r'(?<!\\):\d+(\.\d+)?', '', token)
        token = re.sub(r'(?<!\\)[()]', '', token)
        token = token.replace("\\(", "(").replace("\\)", ")")
        return token.strip(" ,")

    @staticmethod
    def matches_wildcard(pattern: str, token_list: List[str]) -> bool:
        """Check if pattern matches any token in the list (supports wildcards)."""
        if "*" in pattern or "?" in pattern:
            return any(fnmatch.fnmatch(t, pattern) for t in token_list)
        return pattern in token_list

    def get_file_paths(self, folder_path: str) -> Dict[str, str]:
        """Get mapping of image files to their corresponding txt files."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"The folder '{folder_path}' does not exist or is not a directory.")
        
        file_pairs = {}
        image_files = [f for f in glob.glob(os.path.join(folder_path, '*')) 
                      if f.lower().endswith(self.IMAGE_EXTENSIONS)]
        
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(folder_path, base_name + '.txt')
            file_pairs[img_path] = txt_path
            
        return file_pairs

    def read_tokens_from_file(self, filepath: str) -> List[str]:
        """Read and process tokens from a text file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = ""
            
            return self.convert_tokens_in_text(content, return_list=True)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return []

    def write_tokens_to_file(self, filepath: str, tokens: List[str]) -> None:
        """Write tokens to a text file."""
        content = ", ".join(tokens)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def convert_tokens_in_text(self, text: str, prepend_token: Optional[str] = None, 
                             return_list: bool = False) -> Union[str, List[str]]:
        """Convert and process tokens according to mappings."""
        run_sanitize = bool(re.search(r":\d", text))
        
        # Split, sanitize, and normalize tokens
        raw_tokens = text.split(',')
        norm_tokens = []
        for token in raw_tokens:
            if run_sanitize:
                token = self.sanitize_token(token)
            token_clean = self.normalize_token(token)
            if token_clean:
                norm_tokens.append(token_clean)
        
        # Apply combo mappings first
        for combo, replacement in self.combo_mapping.items():
            if all(item in norm_tokens for item in combo):
                norm_tokens = [t for t in norm_tokens if t not in combo]
                norm_tokens.append(replacement)
        
        # Apply other mappings
        processed_tokens = []
        for token in norm_tokens:
            if token in self.mapping:
                new_token = self.mapping[token]
            else:
                new_token = token
                # Apply partial mapping
                for substr, rep in self.partial_mapping.items():
                    if substr in new_token:
                        if substr in self.partial_exceptions and token in self.partial_exceptions[substr]:
                            continue
                        new_token = new_token.replace(substr, rep)
            
            # Skip tokens in removal mapping
            if new_token not in self.remove_mapping:
                processed_tokens.append(new_token)
        
        # Sort and remove duplicates
        unique_tokens = self._remove_duplicates(sorted(processed_tokens))
        
        # Add prepend token if specified
        if prepend_token:
            if prepend_token in unique_tokens:
                unique_tokens.remove(prepend_token)
            unique_tokens.insert(0, prepend_token)
        
        # Apply solo filtering
        unique_tokens = self._apply_solo_filtering(unique_tokens)
        
        return unique_tokens if return_list else ", ".join(unique_tokens)

    def _remove_duplicates(self, tokens: List[str]) -> List[str]:
        """Remove duplicate tokens while preserving order."""
        unique_tokens = []
        prev = None
        for token in tokens:
            if token != prev:
                unique_tokens.append(token)
                prev = token
        return unique_tokens

    def _apply_solo_filtering(self, tokens: List[str]) -> List[str]:
        """Apply solo filtering for size-based tokens."""
        if "solo" not in tokens:
            return tokens
        
        size_order = {"small": 1, "medium": 2, "big": 3, "large": 4, "huge": 5, "gigantic": 6}
        size_groups = {}
        non_size_tokens = []
        
        for token in tokens:
            parts = token.split(maxsplit=1)
            if len(parts) == 2 and parts[0].lower() in size_order:
                size = parts[0].lower()
                item = parts[1].strip()
                rank = size_order[size]
                if item not in size_groups or rank > size_groups[item][0]:
                    size_groups[item] = (rank, token)
            else:
                non_size_tokens.append(token)
        
        return sorted(non_size_tokens + [val[1] for val in size_groups.values()])

    def process_txt_file(self, filepath: str, prepend_token: Optional[str] = None) -> Tuple[bool, str]:
        """Process a single txt file and return whether it changed and diff info."""
        try:
            # Read original content
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                original_tokens = [self.normalize_token(tok) for tok in original_content.split(',') 
                                 if self.normalize_token(tok)]
            else:
                original_content = ""
                original_tokens = []
                print(f"📄 File '{filepath}' not found. Creating new one.")

            # Process content
            new_content = self.convert_tokens_in_text(original_content, prepend_token)
            new_tokens = [self.normalize_token(tok) for tok in new_content.split(',') 
                         if self.normalize_token(tok)]

            # Calculate differences
            diff_str = self._calculate_diff(original_tokens, new_tokens)

            # Write if changed
            if new_content != original_content or (not original_content and new_content.strip()):
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True, diff_str or "(new file)"

            return False, ""
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return False, ""

    def _calculate_diff(self, original_tokens: List[str], new_tokens: List[str]) -> str:
        """Calculate the difference between two token lists."""
        orig_set = set(original_tokens)
        new_set = set(new_tokens)
        added = new_set - orig_set
        removed = orig_set - new_set
        modified = set()

        # Find modifications (same canonical form but different text)
        for a in list(added):
            for r in list(removed):
                if self.canonical(a) == self.canonical(r):
                    modified.add(a)
                    added.discard(a)
                    removed.discard(r)
                    break

        diff_list = []
        for token in sorted(added):
            diff_list.append(f"{token} (+)")
        for token in sorted(removed):
            diff_list.append(f"{token} (-)")
        for token in sorted(modified):
            diff_list.append(f"{token} (~)")

        return ", ".join(diff_list)

    def process_folder(self, folder_path: str, prepend_token: Optional[str] = None) -> Dict[str, str]:
        """Process all txt files in a folder."""
        changed_files = {}
        file_pairs = self.get_file_paths(folder_path)
        
        if not file_pairs:
            print("No image files found in the specified folder.")
            return changed_files

        for img_path, txt_path in file_pairs.items():
            changed, diff_str = self.process_txt_file(txt_path, prepend_token)
            if changed:
                changed_files[txt_path] = diff_str

        return changed_files


class TagAnalyzer:
    """Handles tag analysis operations"""
    
    def __init__(self, tag_processor: TagProcessor):
        self.processor = tag_processor

    def list_tags(self, folder_path: str, priority_tag: Optional[str] = None, 
                  priority_mode: str = "a") -> None:
        """List all tags with frequency counts and optionally prioritize a tag."""
        tag_counts = self._collect_tag_frequencies(folder_path)
        
        # Print frequency table
        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
        for tag, count in sorted_tags:
            print(f"{tag}: {count}")
        print(f"Total unique tags: {len(tag_counts)}")

        # Handle priority tag
        if priority_tag:
            self._apply_priority_tag(folder_path, priority_tag, priority_mode)

    def _collect_tag_frequencies(self, folder_path: str) -> Dict[str, int]:
        """Collect frequency counts for all tags in folder."""
        tag_counts = {}
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        if not txt_files:
            print("No .txt files found in the specified folder.")
            return tag_counts

        for filepath in txt_files:
            try:
                tokens = self.processor.read_tokens_from_file(filepath)
                for token in tokens:
                    tag_counts[token] = tag_counts.get(token, 0) + 1
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        
        return tag_counts

    def _apply_priority_tag(self, folder_path: str, priority_tag: str, mode: str) -> None:
        """Apply priority tag to all files in folder."""
        normalized_priority = self.processor.normalize_token(priority_tag)
        print(f"\nUpdating files so that '{priority_tag}' is at the front...")
        
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        for filepath in txt_files:
            try:
                tokens = self.processor.read_tokens_from_file(filepath)
                
                if mode == "p":  # Priority mode - only if exists
                    if normalized_priority in tokens:
                        tokens.remove(normalized_priority)
                        tokens.insert(0, normalized_priority)
                elif mode == "a":  # Add mode - always ensure at front
                    if normalized_priority in tokens:
                        tokens.remove(normalized_priority)
                    tokens.insert(0, normalized_priority)
                
                self.processor.write_tokens_to_file(filepath, tokens)
                print(f"Updated {filepath}")
            except Exception as e:
                print(f"Error updating {filepath}: {e}")

    def search_files_for_token(self, folder_path: str, search_token: str) -> None:
        """Search for files containing a specific token (supports wildcards)."""
        found_files = []
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        if not txt_files:
            print("No .txt files found in the specified folder.")
            return

        for filepath in txt_files:
            try:
                tokens = self.processor.read_tokens_from_file(filepath)
                matched_tokens = []
                
                if '*' in search_token:
                    for token in tokens:
                        if fnmatch.fnmatch(token, search_token):
                            matched_tokens.append(token)
                else:
                    if search_token in tokens:
                        matched_tokens.append(search_token)
                
                if matched_tokens:
                    found_files.append((filepath, matched_tokens))
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        self._print_search_results(found_files, search_token)

    def _print_search_results(self, found_files: List[Tuple[str, List[str]]], search_token: str) -> None:
        """Print search results."""
        if found_files:
            print(f"Files containing token pattern '{search_token}':")
            for filepath, matches in found_files:
                print(f"{filepath}  ->  Matches: {', '.join(matches)}")
            print(f"\nTotal matching files: {len(found_files)}")
        else:
            print(f"No files found containing token pattern '{search_token}'.")

    def find_missing_files(self, folder_path: str, search_tokens: List[str]) -> None:
        """Find files missing specified tokens."""
        missing_files = []
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        if not txt_files:
            print("No .txt files found in the specified folder.")
            return

        for filepath in txt_files:
            try:
                tokens = self.processor.read_tokens_from_file(filepath)
                
                if len(search_tokens) == 1:
                    if not self._token_matches(search_tokens[0], tokens):
                        missing_files.append(filepath)
                elif len(search_tokens) >= 2:
                    if (self._token_matches(search_tokens[0], tokens) and 
                        not self._token_matches(search_tokens[1], tokens)):
                        missing_files.append(filepath)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        self._print_missing_results(missing_files, search_tokens)

    def _token_matches(self, search_token: str, token_list: List[str]) -> bool:
        """Check if a token matches any in the list (supports wildcards)."""
        if '*' in search_token:
            return any(fnmatch.fnmatch(t, search_token) for t in token_list)
        return search_token in token_list

    def _print_missing_results(self, missing_files: List[str], search_tokens: List[str]) -> None:
        """Print missing files results."""
        if missing_files:
            print("Files missing the specified pattern(s):")
            for f in missing_files:
                print(f)
        else:
            if len(search_tokens) == 1:
                print(f"All files contain token pattern '{search_tokens[0]}'.")
            else:
                print(f"All files contain '{search_tokens[0]}' or also contain '{search_tokens[1]}'.")

    def compare_tags_to_csv(self, folder_path: str, csv_path: str) -> None:
        """Compare tags in folder to allowed tags in CSV file."""
        try:
            allowed_tags, tag_lookup = self._load_allowed_tags(csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file '{csv_path}' not found.")
            return

        disallowed_tags = {}
        mismatch_warnings = []
        
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        for filepath in txt_files:
            try:
                tokens = self.processor.read_tokens_from_file(filepath)
                bad_tokens = []
                
                for tok in tokens:
                    if tok in allowed_tags:
                        canonical = tag_lookup[tok]
                        if '_' in tok and ' ' in canonical:
                            mismatch_warnings.append(
                                f"⚠️  In file '{filepath}': tag '{tok}' matched CSV tag '{canonical}' "
                                f"(underscore → space)"
                            )
                    else:
                        bad_tokens.append(tok)

                if bad_tokens:
                    disallowed_tags[filepath] = bad_tokens
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        self._print_comparison_results(disallowed_tags, mismatch_warnings)

    def _load_allowed_tags(self, csv_path: str) -> Tuple[Set[str], Dict[str, str]]:
        """Load allowed tags from CSV file."""
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            allowed_tags_raw = set()
            for row in reader:
                if row:
                    allowed_tags_raw.add(self.processor.normalize_token(row[0]))

        allowed_tags = set()
        tag_lookup = {}
        
        for tag in allowed_tags_raw:
            variants = self._get_tag_variants(tag)
            for variant in variants:
                allowed_tags.add(variant)
                tag_lookup[variant] = tag

        return allowed_tags, tag_lookup

    def _get_tag_variants(self, tag: str) -> Set[str]:
        """Get normalized variants of a tag with spaces and underscores."""
        tag = self.processor.normalize_token(tag)
        return {tag, tag.replace('_', ' '), tag.replace(' ', '_')}

    def _print_comparison_results(self, disallowed_tags: Dict[str, List[str]], 
                                mismatch_warnings: List[str]) -> None:
        """Print comparison results."""
        if mismatch_warnings:
            print("\n⚠️ Underscore/space mismatch warnings:")
            for warning in sorted(set(mismatch_warnings)):
                print(warning)

        if disallowed_tags:
            all_disallowed = []
            for tags in disallowed_tags.values():
                all_disallowed.extend(tags)

            tag_counts = Counter(all_disallowed)
            print("\n📊 Disallowed tag frequency (across all files):")
            for tag, count in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"  - {tag}: {count} file(s)")


class FileManager:
    """Handles file operations"""
    
    def __init__(self, tag_processor: TagProcessor):
        self.processor = tag_processor

    def reject_files_with_token(self, folder_path: str, reject_token: str) -> None:
        """Move files containing reject token to rejects subdirectory."""
        reject_token = self.processor.normalize_token(reject_token)
        reject_dir = os.path.join(folder_path, "rejects")
        os.makedirs(reject_dir, exist_ok=True)

        rejected_count = 0
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

        for txt_path in txt_files:
            try:
                tokens = self.processor.read_tokens_from_file(txt_path)
                if reject_token in tokens:
                    self._move_file_pair(txt_path, reject_dir)
                    rejected_count += 1
            except Exception as e:
                print(f"Error processing {txt_path}: {e}")

        print(f"✅ Moved {rejected_count} file(s) to '{reject_dir}' based on token '{reject_token}'.")

    def _move_file_pair(self, txt_path: str, reject_dir: str) -> None:
        """Move both txt and corresponding image file."""
        base = os.path.splitext(os.path.basename(txt_path))[0]
        folder_path = os.path.dirname(txt_path)
        
        # Move txt file
        shutil.move(txt_path, os.path.join(reject_dir, os.path.basename(txt_path)))
        
        # Move corresponding image file
        for ext in self.processor.IMAGE_EXTENSIONS:
            img_path = os.path.join(folder_path, base + ext)
            if os.path.exists(img_path):
                shutil.move(img_path, os.path.join(reject_dir, os.path.basename(img_path)))
                break

    def prepare_folder(self, original_path: str, prep_tag: Optional[str] = None) -> str:
        """Prepare a folder by copying, clearing tags, auto-tagging, and optionally adding a tag."""
        prepared_path = original_path.rstrip('/\\') + "-prepared"

        print(f"📁 Copying folder to: {prepared_path}")
        if os.path.exists(prepared_path):
            shutil.rmtree(prepared_path)
        shutil.copytree(original_path, prepared_path)

        print("🧹 Emptying all .txt files...")
        self._empty_txt_files(prepared_path)

        print("🤖 Running auto-tagging...")
        auto_tagger = AutoTagger()
        auto_tagger.auto_tag_images(prepared_path)

        if prep_tag:
            print(f"🏷️ Adding tag '{prep_tag}' to front of all files...")
            changed_files = self.processor.process_folder(
                prepared_path, prepend_token=self.processor.normalize_token(prep_tag)
            )
            print(f"✅ Prepare complete. Modified {len(changed_files)} files.")
        else:
            print("✅ Prepare complete. Skipped prepending tag.")

        return prepared_path

    def _empty_txt_files(self, folder_path: str) -> None:
        """Empty all txt files corresponding to image files."""
        file_pairs = self.processor.get_file_paths(folder_path)
        for _, txt_path in file_pairs.items():
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("")

    def clean_txt_files(self, folder_path: str) -> None:
        """Remove all tags from txt files corresponding to image files."""
        print("🧹 Cleaning all .txt files...")
        
        file_pairs = self.processor.get_file_paths(folder_path)
        cleaned_count = 0
        
        for _, txt_path in file_pairs.items():
            try:
                # Check if file exists and has content
                file_existed = os.path.exists(txt_path)
                had_content = False
                
                if file_existed:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    had_content = bool(content)
                
                # Write empty content
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("")
                
                if had_content:
                    cleaned_count += 1
                    print(f"Cleaned: {txt_path}")
                elif not file_existed:
                    print(f"Created empty: {txt_path}")
                    
            except Exception as e:
                print(f"Error cleaning {txt_path}: {e}")
        
        print(f"✅ Cleaned {cleaned_count} file(s) with existing content.")


class AutoTagger:
    """Handles automatic tagging with WD14 model"""
    
    def auto_tag_images(self, folder_path: str, target_tags: Optional[List[str]] = None) -> None:
        """Auto-tag images using WD14 VIT v2 model."""
        print(f"🔍 Auto-tagging images using WD14 VIT v2...")
        if target_tags:
            if len(target_tags) == 1:
                print(f"🎯 Only tagging images that match: '{target_tags[0]}'")
            else:
                tag_list = ', '.join(f"'{tag}'" for tag in target_tags)
                print(f"🎯 Only tagging images that match any of: {tag_list}")
                
        processor = TagProcessor(MAPPING, COMBO_MAPPING, PARTIAL_MAPPING, 
                               PARTIAL_EXCEPTIONS, REMOVE_MAPPING)
        match_tokens = [processor.normalize_token(tag) for tag in target_tags] if target_tags else None

        model = WDModel(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        updated_count = 0
        file_pairs = processor.get_file_paths(folder_path)

        for image_path, txt_path in tqdm(file_pairs.items(), desc="Tagging images", unit="img"):
            try:
                sample = SimpleCaptionSample(image_path)
                caption = model.generate_caption(sample)
                tags = [processor.normalize_token(t) for t in caption.split(",")]

                if not tags:
                    continue

                # Check if any of the match tokens are found in the image tags
                if match_tokens and not self._matches_any_target(match_tokens, tags, processor):
                    continue

                existing_tags = processor.read_tokens_from_file(txt_path)
                new_tags = self._merge_tags(existing_tags, tags, match_tokens, processor)

                if new_tags and set(new_tags) != set(existing_tags):
                    processor.write_tokens_to_file(txt_path, new_tags)
                    updated_count += 1

            except Exception as e:
                print(f"⚠️ Error processing {image_path}: {e}")

        print(f"\n✅ Total files updated: {updated_count}")

    def _matches_any_target(self, match_tokens: List[str], tags: List[str], processor: TagProcessor) -> bool:
        """Check if any of the target tags match any of the image tags."""
        return any(processor.matches_wildcard(match_token, tags) for match_token in match_tokens)

    def _merge_tags(self, existing_tags: List[str], new_tags: List[str], 
                   match_tokens: Optional[List[str]], processor: TagProcessor) -> List[str]:
        """Merge existing and new tags."""
        if match_tokens:
            matched = []
            for tag in new_tags:
                if tag not in existing_tags:
                    # Check if this tag matches any of our target patterns
                    if any(processor.matches_wildcard(match_token, [tag]) for match_token in match_tokens):
                        matched.append(tag)
            return existing_tags + matched if matched else existing_tags
        else:
            result = existing_tags[:]
            for tag in new_tags:
                if tag not in result:
                    result.append(tag)
            return result


def main():
    parser = argparse.ArgumentParser(
        description="Process comma-separated tokens in .txt files with various operations"
    )
    parser.add_argument("folder", help="Folder path containing .txt files")
    parser.add_argument("-a", "--add", help="Tag to add to the beginning of the list")
    parser.add_argument("-p", "--priority", help="Tag to move to the front (only if it exists)")
    parser.add_argument("-l", "--list", action="store_true", help="List tag totals (default mode)")
    parser.add_argument("-f", "--find", help="Search for files containing the specified token")
    parser.add_argument("-m", "--missing", nargs="+", 
                       help="Search for files that do NOT contain the specified token(s)")
    parser.add_argument("-r", "--remove", help="Remove the specified token from all files")
    parser.add_argument("-c", "--compare", nargs="?", const="popular_tags.csv",
                       help="Compare tags to CSV file (default: popular_tags.csv)")
    parser.add_argument("-j", "--reject", help="Move files with token to 'rejects' subdirectory")
    parser.add_argument("--auto-tag", nargs="?", const="", metavar="TAG",
                       help="Use WD14 VIT v2 to tag images. Optional TAG filter.")
    parser.add_argument("--prepare", nargs="?", const="", metavar="TAG",
                       help="Copy folder, clear tags, auto-tag, and optionally prepend TAG")
    parser.add_argument("--clean", action="store_true", 
                       help="Remove all tags from .txt files")

    args = parser.parse_args()

    # Initialize components
    tag_processor = TagProcessor(MAPPING, COMBO_MAPPING, PARTIAL_MAPPING, 
                                PARTIAL_EXCEPTIONS, REMOVE_MAPPING)
    tag_analyzer = TagAnalyzer(tag_processor)
    file_manager = FileManager(tag_processor)
    auto_tagger = AutoTagger()

    folder_path = args.folder

    # Handle remove operation first (modifies remove_mapping)
    if args.remove:
        token_to_remove = tag_processor.normalize_token(args.remove)
        # Create a new processor with the additional remove token
        extended_remove_mapping = REMOVE_MAPPING.copy()
        extended_remove_mapping.add(token_to_remove)
        
        remove_processor = TagProcessor(MAPPING, COMBO_MAPPING, PARTIAL_MAPPING,
                                      PARTIAL_EXCEPTIONS, extended_remove_mapping)
        
        changed_files = remove_processor.process_folder(folder_path)
        print(f"Token '{args.remove}' removed from all files.")
        print(f"Total files changed: {len(changed_files)}")
        if changed_files:
            print("Files changed:")
            for filepath, diff in changed_files.items():
                print(f"{filepath} - {diff}")
        return

    # Handle prepare operation
    if args.prepare is not None:
        prepared_path = file_manager.prepare_folder(
            folder_path, args.prepare if args.prepare else None
        )
        return

    # Handle auto-tag operation
    if args.auto_tag is not None:
        # Parse comma-separated tags if provided
        if args.auto_tag.strip():
            # Split by comma and normalize each tag
            target_tags = [tag.strip() for tag in args.auto_tag.split(',') if tag.strip()]
        else:
            target_tags = None
        
        auto_tagger.auto_tag_images(folder_path, target_tags)
        return

    # Handle reject operation
    if args.reject:
        file_manager.reject_files_with_token(folder_path, args.reject)
        return

    # Handle comparison operation
    if args.compare:
        tag_analyzer.compare_tags_to_csv(folder_path, args.compare)
        return

    # Determine priority settings
    priority_tag = None
    priority_mode = None
    if args.priority:
        priority_tag = args.priority
        priority_mode = "p"
    elif args.add:
        priority_tag = args.add
        priority_mode = "a"

    # Process files and collect changes
    changed_files = tag_processor.process_folder(folder_path)

    # Handle search and analysis operations
    if args.find:
        tag_analyzer.search_files_for_token(folder_path, args.find)
    elif args.missing:
        tag_analyzer.find_missing_files(folder_path, args.missing)
    else:
        # Default operation: list tags
        tag_analyzer.list_tags(folder_path, priority_tag, priority_mode)
        print(f"\nTotal files changed: {len(changed_files)}")
        if changed_files:
            print("Files changed:")
            for filepath, diff in changed_files.items():
                print(f"{filepath} - {diff}")

    # Handle clean operation
    if args.clean:
        file_manager.clean_txt_files(folder_path)
        return

if __name__ == "__main__":
    main()