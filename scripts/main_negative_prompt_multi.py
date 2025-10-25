import os
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum

# Node names
NODE_MODEL_SELECTION = "model_selection"
NODE_EVALUATION = "evaluation"
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import base64
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from diffusers import PipelineQuantizationConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")
import argparse
import random
from scripts.utils import setup_logging


from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')

# load models
import torch
from PIL import Image
import numpy as np
import sys

from diffusers import (
    QwenImagePipeline, QwenImageEditPipeline,
    QwenImageTransformer2DModel
)
from transformers import Qwen2_5_VLForConditionalGeneration

import re
import json
from prompts.system_prompts import make_intention_analysis_prompt, make_gen_image_judge_prompt

import time
from tqdm import tqdm

# Initialize model variables
qwen_image_pipe = None
qwen_edit_pipe = None

class CreativityLevel(Enum):
    LOW = "low"      # Ask for every unclear element, minimal creative fill
    MEDIUM = "medium"  # Fill some details, ask for important ones
    HIGH = "high"    # Autonomously fill in most details

class T2IConfig:
    """Configuration and state management for T2I workflow."""
    def __init__(self, human_in_loop: bool = True):
        # Global settings
        self.is_human_in_loop = human_in_loop
        self.save_dir = ""
        self.seed = None
        self.image_index = None
        self.logger = None
        
        # Open LLM configurations
        self.use_open_llm = False 
        self.open_llm_model = ""  
        self.open_llm_host = ""  
        self.open_llm_port = ""  

        # Prompt understanding configuration
        self.prompt_understanding = {
            "creativity_level": None,  # Will be determined dynamically based on prompt analysis
            "original_prompt": "",
            "prompt_analysis": "",  # JSON string
            "questions": None,
            "user_clarification": None,
            "refined_prompt": "",
        }

        # Initialize first regeneration config as default config
        self.regeneration_count = 0
        self.regeneration_configs = {
            "count_0": {
                "selected_model": "",
                "generating_prompt": "",
                "reference_content_image": "",
                "reasoning": "",
                "confidence_score": 0.0,
                "gen_image_path": "",
                "evaluation_score": 0.0,
                "user_feedback": None,
                "improvement_suggestions": None
            }
        }

    def add_regeneration_config(self):
        """Create a new regeneration configuration."""
        index = self.regeneration_count + 1
        
        # Get reference content from previous config
        prev_config = self.regeneration_configs[f"count_{self.regeneration_count}"]
        prev_gen_image_path = prev_config["gen_image_path"]

        self.regeneration_configs[f"count_{index}"] = {
            "selected_model": "",
            "generating_prompt": "",
            "reference_content_image": prev_gen_image_path,
            "reasoning": "",
            "confidence_score": 0.0,
            "gen_image_path": "",
            "evaluation_score": 0.0,
            "user_feedback": None,
            "improvement_suggestions": None
        }
        self.regeneration_count = index
        return f"count_{index}"

    def get_current_config(self):
        """Get the current regeneration configuration."""
        return self.regeneration_configs[f"count_{self.regeneration_count}"]

    def get_prev_config(self):
        """Get the current regeneration configuration."""
        return self.regeneration_configs[f"count_{self.regeneration_count-1}"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for storage."""
        # Convert the CreativityLevel enum to its string value
        prompt_understanding = self.prompt_understanding.copy()
        if prompt_understanding["creativity_level"] is not None:
            prompt_understanding["creativity_level"] = prompt_understanding["creativity_level"].value
        else:
            prompt_understanding["creativity_level"] = "MEDIUM"  # Default fallback

        # Ensure prompt_analysis is stored as a dictionary, not a JSON string
        if isinstance(prompt_understanding["prompt_analysis"], str):
            # Handle empty string case first
            if not prompt_understanding["prompt_analysis"].strip():
                self.logger.debug("Before prompt analyze, so getting empty prompt_analysis string, using empty dict")
                prompt_understanding["prompt_analysis"] = {}
            else:
                try:
                    prompt_understanding["prompt_analysis"] = json.loads(prompt_understanding["prompt_analysis"])
                    self.logger.debug("Successfully parsed prompt_analysis JSON string")
                except json.JSONDecodeError:
                    self.logger.error("Invalid JSON string in prompt_analysis")
                    try:
                        # Try to preserve the string content if it's not JSON
                        prompt_understanding["prompt_analysis"] = {
                            "raw_content": prompt_understanding["prompt_analysis"]
                        }
                        self.logger.debug("Preserved raw prompt_analysis content")
                    except Exception as e:
                        self.logger.error(f"Failed to preserve prompt analysis content: {e}")
                        prompt_understanding["prompt_analysis"] = {}
        elif isinstance(prompt_understanding["prompt_analysis"], dict):
            self.logger.debug("prompt_analysis is already a dictionary")
        else:
            self.logger.warning(f"Unexpected prompt_analysis type: {type(prompt_understanding['prompt_analysis'])}")
            prompt_understanding["prompt_analysis"] = {}

        return {
            "is_human_in_loop": self.is_human_in_loop,
            "save_dir": self.save_dir,
            "seed": self.seed,
            "image_index": self.image_index,
            "use_open_llm": self.use_open_llm,
            "open_llm_model": self.open_llm_model,
            "open_llm_host": self.open_llm_host,
            "open_llm_port": self.open_llm_port,
            "prompt_understanding": prompt_understanding,
            "regeneration_configs": self.regeneration_configs,
            "regeneration_count": self.regeneration_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'T2IConfig':
        """Create config from stored dictionary."""
        instance = cls()
        
        # Load global settings directly from the dictionary
        instance.is_human_in_loop = data["is_human_in_loop"]
        instance.save_dir = data["save_dir"]
        instance.seed = data["seed"]
        instance.image_index = data["image_index"]
        
        # Load use_open_llm if it exists in the data
        if "use_open_llm" in data:
            instance.use_open_llm = data["use_open_llm"]
        
        # Load open LLM settings if they exist
        if "open_llm_model" in data:
            instance.open_llm_model = data["open_llm_model"]
        if "open_llm_host" in data:
            instance.open_llm_host = data["open_llm_host"]
        if "open_llm_port" in data:
            instance.open_llm_port = data["open_llm_port"]
        
        # Load prompt understanding config
        instance.prompt_understanding = data["prompt_understanding"]
        creativity_level_value = instance.prompt_understanding["creativity_level"]
        if creativity_level_value is not None and creativity_level_value != "":
            try:
                instance.prompt_understanding["creativity_level"] = CreativityLevel(creativity_level_value)
            except ValueError:
                # Fallback to MEDIUM if invalid value
                instance.prompt_understanding["creativity_level"] = CreativityLevel.MEDIUM
        else:
            # If None or empty, will be determined dynamically later
            instance.prompt_understanding["creativity_level"] = None

        # Load regeneration configs
        instance.regeneration_configs = data["regeneration_configs"]
        instance.regeneration_count = data["regeneration_count"]

        return instance

    def save_to_file(self, filename: str):
        """Save current config state to a JSON file."""
        try:
            config_data = self.to_dict()
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            self.logger.debug(f"Successfully saved config to: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving config to {filename}: {str(e)}")

    @classmethod
    def load_from_file(cls, filename: str) -> 'T2IConfig':
        """Load config state from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading config from {filename}: {str(e)}")
            raise

# Function to initialize LLMs based on configuration
def initialize_llms(use_open_llm=False, open_llm_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503", local_host="0.0.0.0", local_port="8000"):
    """Initialize the LLM models based on configuration"""
    global llm, llm_json
    
    if use_open_llm:
        openai_api_base = f"http://{local_host}:{local_port}/v1"
        openai_api_key = "eqr3k3jlk21jdlkdmvli23rjnflwejnfikjcn"  # Not actually validated
        
        llm = ChatOpenAI(
            model=open_llm_model,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            temperature=0.15
        )
        
        llm_json = ChatOpenAI(
            model=open_llm_model,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            temperature=0.15,
            response_format={"type": "json_object"}
        )
        
        print(f"Initialized OpenSource LLM: {open_llm_model}")
    else:
        # Using Qwen-2.5
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            openai_api_base="https://openrouter.ai/api/v1"
        )
        llm_json = ChatOpenAI(
            model="openai/gpt-4o-mini",
            openai_api_base="https://openrouter.ai/api/v1",
            response_format={"type": "json_object"}
        )
        print("Initialized GPT Instruct")
    
    return llm, llm_json

class IntentionAnalyzer:
    """Helper class for intention understanding operations"""
    def __init__(self, llm):
        self.llm = llm
        self.llm_json = llm.bind(response_format={"type": "json_object"})
        self.logger = logger

    def determine_creativity_level(self, prompt: str) -> CreativityLevel:
        """Analyze the prompt to automatically determine the appropriate creativity level."""
        self.logger.debug(f"Determining creativity level for prompt: '{prompt}'")
        
        creativity_analysis_prompt = """You are an expert at analyzing image generation prompts to determine the appropriate creativity level.

Analyze the given prompt and determine the creativity level based on these criteria:

HIGH Creativity Level (system should be highly creative and autonomous):
- Very brief or vague prompts (e.g., "a cat", "landscape", "portrait")
- Abstract concepts or artistic requests (e.g., "surreal dream", "impressionist style")
- Prompts with many undefined elements or lacking specific details
- Creative or artistic enhancement requests (e.g., "make it more dramatic", "artistic interpretation")
- Prompts that invite interpretation and artistic freedom

MEDIUM Creativity Level (balanced approach):
- Moderately detailed prompts with some specifics but room for enhancement
- Prompts with clear subject but undefined context or style
- Standard scene descriptions that could benefit from creative details
- Prompts with mix of specific and general elements

LOW Creativity Level (stick closely to specifications):
- Highly detailed and specific prompts with clear requirements
- Technical or precise requests (e.g., "headshot photo", "product photography")
- Prompts with explicit style, composition, and detail specifications
- Professional or commercial image requests
- Prompts that leave little room for interpretation

Return JSON with:
{
    "creativity_level": "LOW|MEDIUM|HIGH",
    "reasoning": "Detailed explanation of why this creativity level was chosen",
    "prompt_characteristics": {
        "detail_level": "low|medium|high",
        "specificity": "vague|moderate|precise",
        "artistic_freedom": "constrained|balanced|open"
    }
}

Examples:

Input: "a cat"
Output: {
    "creativity_level": "HIGH",
    "reasoning": "Very brief prompt with minimal details. Requires significant creative interpretation for breed, pose, setting, lighting, style, etc.",
    "prompt_characteristics": {"detail_level": "low", "specificity": "vague", "artistic_freedom": "open"}
}

Input: "Professional headshot of a 30-year-old woman with brown hair, wearing a navy blue blazer, neutral background, studio lighting"
Output: {
    "creativity_level": "LOW", 
    "reasoning": "Highly specific prompt with clear requirements for age, appearance, clothing, background, and lighting. Little room for creative interpretation.",
    "prompt_characteristics": {"detail_level": "high", "specificity": "precise", "artistic_freedom": "constrained"}
}

Input: "A medieval marketplace in a fantasy setting"
Output: {
    "creativity_level": "MEDIUM",
    "reasoning": "Clear subject and setting but many details undefined (architecture style, time of day, characters, atmosphere, specific elements). Balanced between guidance and creative freedom.",
    "prompt_characteristics": {"detail_level": "medium", "specificity": "moderate", "artistic_freedom": "balanced"}
}"""

        try:
            response = track_llm_call(self.llm_json.invoke, "creativity_determination", [
                ("system", creativity_analysis_prompt),
                ("human", f"Analyze this prompt and determine creativity level: '{prompt}'")
            ])
            
            if isinstance(response.content, str):
                result = json.loads(response.content)
            else:
                result = response.content
                
            creativity_level_str = result.get("creativity_level", "MEDIUM")
            reasoning = result.get("reasoning", "Default reasoning")
            
            # Convert string to enum
            if creativity_level_str == "HIGH":
                creativity_level = CreativityLevel.HIGH
            elif creativity_level_str == "LOW":
                creativity_level = CreativityLevel.LOW
            else:
                creativity_level = CreativityLevel.MEDIUM
                
            self.logger.info(f"Determined creativity level: {creativity_level.value}")
            self.logger.info(f"Reasoning: {reasoning}")
            
            return creativity_level
            
        except Exception as e:
            self.logger.error(f"Error in creativity determination: {str(e)}. Defaulting to MEDIUM.")
            return CreativityLevel.MEDIUM

    def identify_image_path(self, prompt: str) -> str:
        from urllib.parse import urlparse

        # Extract possible image filename or URL from the prompt
        match = re.search(r"[\w/.\-]+\.png|[\w/.\-]+\.jpg|https?://[\w/.\-]+", prompt)
        if match:
            path_or_url = match.group()
            parsed_url = urlparse(path_or_url)

            if parsed_url.scheme in ['http', 'https']:
                # It's a URL
                # print(f"Identified URL: {path_or_url}")
                self.logger.debug(f"Identified URL: {path_or_url}")
                return path_or_url, "url"
            else:
                # It's a local file path
                full_path = os.path.abspath(os.path.expanduser(path_or_url))
                if os.path.exists(full_path):
                    return full_path, "local"
                else:
                    # print(f"Image '{full_path}' not found.")
                    self.logger.error(f"Image '{full_path}' not found.")
                    return None, None
        # print("No valid image file or URL found in the prompt.")
        self.logger.debug("No valid image file or URL found in the prompt.")
        return None, None

    def analyze_prompt(self, prompt: str, creativity_level: CreativityLevel) -> Dict[str, Any]:
        """Analyze the prompt and identify elements that need clarification."""
        self.logger.debug(f"Analyzing prompt: '{prompt}' with creativity level: {creativity_level.value}")
        
        # Identify image input by ".png" or ".jpg"
        # NOTE: currently only support one image (the first identified image) in the prompt
        image_dir_in_prompt, image_type = self.identify_image_path(prompt)
        if image_dir_in_prompt:
            self.logger.debug(f"Identifying image from: {image_dir_in_prompt}; Image type: {image_type}")
            if image_type == "url":
                analysis_prompt = [
                                    (
                                        "system",
                                        make_intention_analysis_prompt()
                                    ),
                                    (
                                        "human",
                                        [
                                            {"type": "text", 
                                            "text": f"Analyze this image generation prompt: '{prompt}' with creativity level: {creativity_level.value}"},
                                            {"type": "image_url", 
                                            "image_url": {"url": f"{image_dir_in_prompt}"}}
                                        ]
                                    )
                                ]
                
            elif image_type == "local":
                # read image and convert to base64
                with open(image_dir_in_prompt, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    analysis_prompt = [
                                    (
                                        "system",
                                        make_intention_analysis_prompt()
                                    ),
                                    (
                                        "human",
                                        [
                                            {"type": "text", 
                                            "text": f"Analyze this image generation prompt: '{prompt}' with creativity level: {creativity_level.value}"},
                                            {"type": "image_url", 
                                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                        ]
                                    )
                                ]
        else:
            analysis_prompt = [
                (
                    "system",
                    make_intention_analysis_prompt()
                ),
                (
                    "human",
                    f"Analyze this image generation prompt: '{prompt}' with creativity level: {creativity_level.value}"
                )
            ]
        
        # Get response as string and parse it to dict
        response = track_llm_call(self.llm_json.invoke, "intention_analysis", analysis_prompt)
        self.logger.debug(f"Raw LLM response: {response}")
        
        # response is <class 'langchain_core.messages.ai.AIMessage'>
        # response.content is <class 'str'>

        try:
            if isinstance(response.content, str):
                parsed_response = json.loads(response.content)
            elif isinstance(response.content, dict):
                parsed_response = response.content
            elif isinstance(response.content, json):
                parsed_response = response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
            self.logger.debug(f"Parsed response: {json.dumps(parsed_response, indent=2)}")
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            self.logger.error(f"Response was: {response}")
            raise

    def retrieve_reference(self, analysis: Dict[str, Any]):
        """Retrieve refenrece content or style based on the analysis."""

        # get current config
        current_config = config.get_current_config()
        if "references" in analysis["identified_elements"] and analysis["identified_elements"]["references"].get("content"):
            current_config["reference_content_image"] = analysis["identified_elements"]["references"]["content"]
        
        self.logger.debug(f"Retrieved reference content image: {current_config['reference_content_image']}")
        
    def retrieve_questions(self, analysis: Dict[str, Any], creativity_level: CreativityLevel) -> str:
        """Retrieve questions based on the analysis and creativity level."""
        self.logger.debug(f"Generating questions for creativity level: {creativity_level.value}")
        
        if creativity_level == CreativityLevel.HIGH:
            self.logger.debug("High creativity mode - returning AUTOCOMPLETE")
            return "AUTOCOMPLETE"

        questions = []
        
        for ambiguous_element in analysis["ambiguous_elements"]:
            questions.extend(ambiguous_element["suggested_questions"])
        
        # For LOW creativity, ask for ALL unclear elements (more questions)
        if creativity_level == CreativityLevel.LOW:
            # Add additional detail-oriented questions
            if "identified_elements" in analysis:
                for element_category, element_data in analysis["identified_elements"].items():
                    if isinstance(element_data, dict) and element_data.get("needs_clarification", False):
                        questions.append(f"Please specify details for {element_category}: {element_data.get('description', '')}")
            
            # Always ask questions for LOW creativity unless there are really none
            if not questions:
                questions.append("Please provide any additional specific details you'd like to include.")
        
        if not questions:
            self.logger.debug("No questions needed - returning SUFFICIENT_DETAIL")
            return "SUFFICIENT_DETAIL"
        
        formatted_questions = "\n".join([f"- {q}" for q in questions])
        self.logger.debug(f"Retrieved questions:\n{formatted_questions}")
        return formatted_questions

    def refine_prompt_with_analysis(self,
                                    original_prompt: str, 
                                    analysis: Dict[str, Any], 
                                    user_responses: Optional[Dict[str, str]] = None,
                                    creativity_level: CreativityLevel = CreativityLevel.MEDIUM
                                ) -> Dict[str, Any]:
        """
        Refine the prompt using the analysis and any user responses.
        Also evaluates detail level when user responses are provided.
        
        Returns:
            Dict containing:
            - refined_prompt: str
            - suggested_creativity_level: CreativityLevel (only when user_responses provided)
        """
        self.logger.debug(f"Original prompt: '{original_prompt}'")
        self.logger.debug(f"User responses: {user_responses}")
        self.logger.debug(f"Creativity level: {creativity_level.value}")
        
        if user_responses:
            refinement_prompt = f"""
            Original prompt: "{original_prompt}"
            Analysis: {json.dumps(analysis, indent=2)}
            User responses: {json.dumps(user_responses, indent=2)}
            Current creativity level: {creativity_level.value}
            
            You are a Qwen-Image prompt expert. Your PRIMARY GOAL is to stay faithful to the original prompt while incorporating user clarifications. CRITICAL: The refined prompt must preserve the core intent, subjects, and atmosphere of the original prompt.
            
            GROUNDING PRINCIPLES:
            - PRESERVE ALL original subjects, objects, and key elements mentioned in the original prompt
            - MAINTAIN the original scene's core atmosphere, mood, and context
            - ONLY ADD details that directly support or clarify the original prompt
            - AVOID introducing new subjects, objects, or concepts not implied by the original
            - USER RESPONSES should only clarify ambiguities, not replace original elements
            
            Steps: 
            0. Convert negative statements into positive ones by rephrasing to focus on what should be included, without mentioning what should not be included. Examples:
               * "Do not wear a coat" -> "Wear a light sweater"
               * "No trees in background" -> "Clear blue sky background"
               * "Remove the hat" -> "Show full hair styling"
               * "Not smiling" -> "Serious expression"
               * "No bright colors" -> "Muted, subtle tones"
            1. START with the original prompt as the foundation - preserve its exact wording where possible
            2. INCORPORATE user responses ONLY to resolve specific ambiguities identified in the analysis
            3. ADD minimal creative details from analysis ONLY if they directly support the original prompt's intent
            4. ENSURE the refined prompt sounds like an enhanced version of the original, not a different prompt
            5. If there is reference image, must keep the its directory
            6. Suggest creativity level based on detail completeness of user responses:
               - LOW: If user provided very specific details for most aspects
               - MEDIUM: If some details are provided but some flexibility is needed
               - HIGH: If many details are still open to interpretation
            
            Return a JSON with:
            {{
                "refined_prompt": "A refined version that stays closely grounded to the original prompt while incorporating user clarifications. The refined prompt should read as a natural enhancement of the original, not a replacement.",
                "suggested_creativity_level": "LOW|MEDIUM|HIGH",
                "reasoning": "Explain why the suggested creativity level was chosen based on the detail completeness of user responses."
            }}
            """
        else:
            refinement_prompt = f"""
            Original prompt: "{original_prompt}"
            Analysis: {json.dumps(analysis, indent=2)}
            Creativity level: {creativity_level.value}

            You are a Qwen prompt expert. Your PRIMARY GOAL is to stay faithful to the original prompt while resolving ambiguities. CRITICAL: The refined prompt must preserve the core intent, subjects, and atmosphere of the original prompt.
            
            GROUNDING PRINCIPLES:
            - PRESERVE ALL original subjects, objects, and key elements mentioned in the original prompt
            - MAINTAIN the original scene's core atmosphere, mood, and context
            - ONLY ADD details that directly support or clarify the original prompt
            - AVOID introducing new subjects, objects, or concepts not clearly implied by the original
            - Creative filling should ENHANCE, not REPLACE or OVERSHADOW original elements
            
            Steps: 
            0. Convert negative statements into positive ones by rephrasing to focus on what should be included, without mentioning what should not be included. Examples:
               * "Do not wear a coat" -> "Wear a light sweater"
               * "No trees in background" -> "Clear blue sky background"
               * "Remove the hat" -> "Show full hair styling"
               * "Not smiling" -> "Serious expression"
               * "No bright colors" -> "Muted, subtle tones"
            1. START with the original prompt as the foundation - preserve its core structure and intent
            2. RESOLVE ambiguous elements using creative_fill from analysis, but only for true ambiguities
            3. ADD minimal supporting details ONLY if creativity_level is MEDIUM or HIGH AND they enhance the original concept
            4. ENSURE the refined prompt feels like a clearer version of the original, not a different scene
            5. If there is reference image, must keep the its directory
            
            Return a JSON with:
            {{
                "refined_prompt": "A refined version that stays closely grounded to the original prompt while resolving necessary ambiguities. The result should read as a natural clarification of the original, maintaining its core essence.",
                "reasoning": "Explain how the refinement preserves the original prompt's intent while addressing ambiguities."
            }}
            """
        
        response = track_llm_call(self.llm_json.invoke, "refine_prompt", refinement_prompt)
        self.logger.debug(f"Refinement result: {response}")

        try:
            if isinstance(response.content, str):
                parsed_response = json.loads(response.content)
            elif isinstance(response.content, dict):
                parsed_response = response.content
            elif isinstance(response.content, json):
                parsed_response = response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
            self.logger.debug(f"Parsed analysis: {json.dumps(parsed_response, indent=2)}")
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            self.logger.error(f"Response was: {response}")
            raise


class NegativePromptGenerator:
    """Helper class for generating negative prompts"""
    def __init__(self, llm):
        self.llm = llm
        self.llm_json = llm.bind(response_format={"type": "json_object"})
        self.logger = logger

    def generate_negative_prompt(self, positive_prompt: str, analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a negative prompt based on the positive prompt and analysis."""
        self.logger.debug(f"Generating negative prompt for: '{positive_prompt}'")
        
        negative_prompt_system = """You are an expert at generating negative prompts for image generation models like Qwen-Image and Qwen-Image-Edit.

A negative prompt specifies what should NOT appear in the generated image. It helps avoid common issues like:
- Poor quality artifacts (blurry, distorted, low quality, pixelated)
- Unwanted objects or elements that commonly appear in similar scenes
- Inappropriate content or style mismatches
- Technical issues (watermarks, text overlays, borders)

Guidelines:
1. Keep negative prompts concise but comprehensive
2. Focus on common unwanted elements for the specific scene type
3. Include general quality-related terms
4. Avoid being too restrictive - don't negate core elements of the positive prompt
5. Consider the context and style of the positive prompt

Return a JSON with:
{
    "negative_prompt": "comma-separated negative prompt terms",
    "reasoning": "explanation of why these negative elements were chosen"
}

Examples:
- Portrait: "blurry, low quality, distorted face, multiple heads, extra limbs, watermark, text"
- Landscape: "people, buildings, text, watermark, low quality, blurry, oversaturated"
- Fantasy scene: "modern objects, realistic style, low quality, blurry, watermark, text"
"""

        if analysis:
            analysis_text = f"\nPrompt analysis context: {json.dumps(analysis, indent=2)}"
        else:
            analysis_text = ""

        negative_prompt_user = f"""Generate a negative prompt for this image generation request:
Positive prompt: "{positive_prompt}"{analysis_text}

Consider the scene type, style, and content to determine what should be avoided."""

        try:
            response = track_llm_call(
                self.llm_json.invoke, 
                "negative_prompt_generation", 
                [
                    ("system", negative_prompt_system),
                    ("human", negative_prompt_user)
                ]
            )
            
            self.logger.debug(f"Raw negative prompt response: {response}")
            
            if isinstance(response.content, str):
                parsed_response = json.loads(response.content)
            elif isinstance(response.content, dict):
                parsed_response = response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response.content)}")
                
            self.logger.debug(f"Generated negative prompt: {parsed_response['negative_prompt']}")
            self.logger.debug(f"Reasoning: {parsed_response['reasoning']}")
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Failed to generate negative prompt: {str(e)}")
            # Fallback to generic negative prompt
            return {
                "negative_prompt": "low quality, blurry, distorted, watermark, text, bad anatomy",
                "reasoning": "Fallback generic negative prompt due to generation error"
            }


def load_models(use_quantization=True):
    """Pre-load models with proper GPU memory management
    
    Args:
        use_quantization (bool): Whether to use quantization and FP16 for reduced memory usage.
            If False, models will be loaded in FP32 without quantization for higher precision.
    """
    global qwen_image_pipe, qwen_edit_pipe

    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        if use_quantization:
            
            print("Loading Qwen Image model with quantization...")
            model_id = "Qwen/Qwen-Image"
            torch_dtype = torch.bfloat16
            device = "cuda"
            
            quantization_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
            )

            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )
            transformer = transformer.to("cpu")

            quantization_config = TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                subfolder="text_encoder",
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )

            text_encoder = text_encoder.to("cpu")

            qwen_image_pipe = QwenImagePipeline.from_pretrained(
                model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
            )

            qwen_image_pipe.to("cuda")

            print("GPU Name:", torch.cuda.get_device_name(0))
            print("Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**3, 2), "GB")
            print("Memory Cached:", round(torch.cuda.memory_reserved(0)/1024**3, 2), "GB")
            print("Total Memory:", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), "GB")
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()

            print("Loading Qwen Image Edit model with specialized quantization...")
            model_id = "Qwen/Qwen-Image-Edit"
            
            # Specialized quantization for transformer
            transformer_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
            )
            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=transformer_config,
                torch_dtype=torch.bfloat16
            )
            transformer = transformer.to("cpu")

            # Specialized quantization for text encoder
            text_encoder_config = TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            #     model_id,
            #     subfolder="text_encoder",
            #     quantization_config=text_encoder_config,
            #     torch_dtype=torch.bfloat16
            # )
            # text_encoder = text_encoder.to("cpu")

            # Create pipeline with specialized components
            qwen_edit_pipe = QwenImageEditPipeline.from_pretrained(
                model_id,
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=torch.bfloat16
            )
            qwen_edit_pipe.to("cuda")
            
        else:
            print("Loading Qwen Image model without quantization...")
            with torch.cuda.device(0):
                qwen_image_pipe = QwenImagePipeline.from_pretrained(
                    "Qwen/Qwen-Image", 
                    torch_dtype=torch.bfloat16,
                    device_map="cuda"
                )

            print("Loading Qwen Image Edit model without quantization...")
            with torch.cuda.device(0):
                qwen_edit_pipe = QwenImageEditPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit", 
                    torch_dtype=torch.bfloat16,
                    device_map="cuda"
                )

        # Enable model memory efficiency
        # if hasattr(qwen_image_pipe, "enable_model_cpu_offload"):
        #     qwen_image_pipe.enable_model_cpu_offload()
        # if hasattr(qwen_edit_pipe, "enable_model_cpu_offload"):
        #     qwen_edit_pipe.enable_model_cpu_offload()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU out of memory. Trying to free up memory...")
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory. Try reducing batch size or image dimensions.")
        raise
    

@tool("Qwen-Image")
def generate_with_qwen_image(prompt: str, negative_prompt: str, seed: int) -> str:
    """
    Given a prompt, Qwen-Image generates general purpose images with high quality and consistency.
    
    Args:
        prompt: The text prompt describing the image to generate
        negative_prompt: Text describing what should not be included in the image
        seed: Random seed for reproducibility
        
    Returns:
        Path to generated images
    """
    logger.info(f"Executing Qwen-Image with prompt: {prompt}; negative prompt: {negative_prompt}; seed {seed}")
    logger.info("Using Qwen-Image model...")

    try:
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Ensure we're on the right device
        generator = torch.Generator("cuda").manual_seed(seed)
        
        with torch.inference_mode():
            negative_prompt_ = "blucolors, overexposedrred details, subtitles, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still image, cluttered background, three legs"
            negative_prompt = negative_prompt_ + negative_prompt
            # Use native negative prompt support
            # print("Negative Prompt: ",negative_prompt)
            # print("Positive Prompt: ",prompt)
            image = qwen_image_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=1328,
                width=1328,
                true_cfg_scale=4.0,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=generator
            ).images[0]
            
        # Normalize image before saving
        image = normalize_image(image)

        # Set output path based on regeneration count
        if config.regeneration_count > 0:
            output_path = os.path.join(config.save_dir, f"{config.image_index}_regen{config.regeneration_count}_Qwen-Image.png")
        else:
            output_path = os.path.join(config.save_dir, f"{config.image_index}_Qwen-Image.png")
        image.save(output_path)
        
        # Log the negative prompt in the config
        current_config = config.get_current_config()
        current_config["negative_prompt"] = negative_prompt
        
        logger.debug(f"Successfully generated image at: {output_path}\n")
        return output_path
            
    except Exception as e:
        logger.error(f"Error in Qwen-Image generation: {str(e)}")
        return f"Error generating image with Qwen-Image: {str(e)}"


@tool("Qwen-Image-Edit")
def generate_with_qwen_edit(prompt: str, negative_prompt: str, existing_image_dir: str, seed: int = 42, guidance_scale: float = 4.0) -> str:
    """
    Qwen-Image-Edit: High-Quality Image Editing. Directly edit images using text instructions.
    
    Args:
        prompt: The text prompt describing the desired modifications
        negative_prompt: Text describing what should not be included in the image
        existing_image_dir: The path to the existing image to edit
        seed: Random seed for reproducibility
        guidance_scale: The guidance scale for the Qwen-Image-Edit model (higher values increase prompt adherence)

    Returns:
        Path to generated image
    """
    global qwen_edit_pipe
    
    logger.info(f"Executing Qwen Image Edit with prompt: {prompt}; negative prompt: {negative_prompt}; seed: {seed}")
    
    # Set up output path
    if config.regeneration_count > 0:
        output_path = os.path.join(config.save_dir, f"{config.image_index}_regen{config.regeneration_count}_Qwen-Image-Edit.png")
    else:
        output_path = os.path.join(config.save_dir, f"{config.image_index}_Qwen-Image-Edit.png")
    
    try:
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Load and verify input image
        try:
            input_img = Image.open(existing_image_dir).convert("RGB")
            if input_img.size[0] > 2048 or input_img.size[1] > 2048:
                logger.warning("Large image detected. Resizing to max 2048x2048 to prevent OOM.")
                input_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Error loading input image: {str(e)}")
            raise ValueError(f"Failed to load input image from {existing_image_dir}: {str(e)}")
        
        # Ensure we're on the right device
        generator = torch.Generator("cuda").manual_seed(seed)
        
        with torch.inference_mode():
            # Use native negative prompt support
            # print("Negative Prompt: ",negative_prompt)
            # print("Positive Prompt: ",prompt)
            result = qwen_edit_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_img,
                height=1328,
                width=1328,
                true_cfg_scale=4.0,
                num_inference_steps=50,
                generator=generator
            ).images[0]

        # Normalize image before saving
        result = normalize_image(result)

        # Save the result
        result.save(output_path)
        
        # Log the negative prompt in the config
        current_config = config.get_current_config()
        current_config["negative_prompt"] = negative_prompt
        
        logger.debug(f"Successfully generated image at: {output_path}\n")
        return output_path
    except Exception as e:
        logger.error(f"Error in Qwen Image Edit generation: {str(e)}; return input image path: {existing_image_dir}")
        return existing_image_dir




class ModelSelector:
    """Helper class for model selection and execution"""
    def __init__(self, llm):
        self.llm = llm
        self.llm_json = llm.bind(response_format={"type": "json_object"})
        self.logger = logger
        self.negative_prompt_generator = NegativePromptGenerator(llm)  # Add this line
        self.tools = {
            "Qwen-Image": generate_with_qwen_image,
            "Qwen-Image-Edit": generate_with_qwen_edit,
        }
        self.available_models = {
            "Qwen-Image": generate_with_qwen_image.func.__doc__,
            "Qwen-Image-Edit": generate_with_qwen_edit.func.__doc__,
        }

    def _create_system_prompt(self) -> str:
        """Create the system prompt with all examples and guidelines."""
        if config.regeneration_count > 0:
            return f"""Select the most suitable model for the given task and generate both positive and negative prompts.
            
            Available models:
            1. Qwen-Image: {self.available_models['Qwen-Image']}
            2. Qwen-Image-Edit: {self.available_models['Qwen-Image-Edit']}

            Note:
            - Best cases for selecting Qwen-Image:
                - For general image generation without reference images
                - For atmosphere/mood/lighting/style improvements or enhancing visual qualities:
                    * Integrates specific atmospheric details into the original prompt
                    * Uses descriptive language for the desired mood or visual effect
                    * Examples:
                        - "make it more dramatic" -> "A dramatic scene with high contrast lighting, deep shadows, and intense atmosphere, showing [original elements]"
                        - "enhance cozy feeling" -> "A warm and intimate setting with soft, golden lighting and comfortable ambiance, featuring [original elements]"
                        - "more professional atmosphere" -> "A polished and sophisticated environment with clean lines and professional lighting, showcasing [original elements]"
                        - "enhance ghostly features" -> "A haunting scene with ethereal, translucent ghostly elements that emit a subtle glow, featuring [original elements]"
                        - "make waves more dramatic" -> "A scene with powerful, dynamic waves with detailed foam and spray, showing [original elements]"
                        - "increase texture detail" -> "A scene with highly detailed surfaces, emphasizing intricate textures and fine details in [original elements]"
                        - "enhance lighting" -> "A scene with dramatic lighting effects, creating bold contrasts and atmospheric illumination for [original elements]"
                    
            - Best cases for selecting Qwen-Image-Edit:
                - When there's a reference image to edit
                - For precise object removal with natural background filling
                - For adding new objects to specific regions
                - For seamless texture replacement
                - For complex scene modifications requiring natural blending
                    
            - Guidelines for generating_prompt:
                - CRITICAL: Stay faithful to the original prompt's core intent, subjects, and atmosphere
                - PRESERVE all original subjects, objects, and key elements mentioned in the original prompt
                - ENHANCE clarity and detail while maintaining the original scene's essence
                - Convert negative statements into positive ones:
                    * "Do not wear a coat" -> "Wear a light sweater"
                    * "No trees in background" -> "Clear blue sky background"
                    * "Remove the hat" -> "Show full hair styling"
                    * "Not smiling" -> "Serious expression"
                    * "No bright colors" -> "Muted, subtle tones"
                - Structure prompts with clear spatial relationships:
                    * Bad: "A vintage armchair. A sleeping cat. A Persian rug. Antique books. Wooden shelves."
                    * Good: "A vintage leather armchair dominates the corner, its worn texture catching the ambient light. A cat sleeps peacefully on the Persian rug spread before it, while antique books line the wooden shelves along the wall."
                - The generating_prompt should read like an enhanced version of the original, not a replacement
            
            - Guidelines for negative_prompt:
                - Include quality-related terms: "low quality, blurry, distorted, pixelated"
                - Include unwanted technical elements: "watermark, text, signature, border"
                - Include scene-specific unwanted elements based on the positive prompt
                - Keep it concise but comprehensive
                - Don't negate core elements of the positive prompt
            
            - Return a JSON with:
            {{
                "selected_model": "model_name",
                "reference_content_image": "path to the reference content image",
                "generating_prompt": "model-specific prompt",
                "negative_prompt": "negative prompt to avoid unwanted elements",
                "reasoning": "Detailed explanation of why this model was chosen",
                "confidence_score": float  # 0.0 to 1.0
            }}

            # Example 1 (General Image Generation):
            - Given prompt: "An ancient dragon perched on a cliff overlooking a stormy ocean."
            - Given improvement: "Make the storm more intense and add more detail to the dragon's scales."
            - Returned JSON:
                {{
                    "selected_model": "Qwen-Image",
                    "reference_content_image": null,
                    "generating_prompt": "A colossal ancient dragon with highly detailed, weathered scales perches on a jagged cliff, its wings partially extended. Each scale features deep grooves and sharp ridges, reflecting flickers of ambient lightning. The background consists of a stormy ocean with hurricane-force winds and diagonal rain streaks. The sky is filled with dense, turbulent storm clouds, illuminated by intermittent lightning flashes that cast stark highlights on the dragon's body. Towering waves below crash violently against the rocky shore, creating dynamic white foam and spray patterns. The dragon is positioned as the dominant foreground element, with a cinematic contrast between its dark silhouette and the electric blue highlights from the storm.",
                    "negative_prompt": "low quality, blurry, distorted, cartoon style, cute dragon, bright colors, sunny weather, calm ocean, people, buildings, text, watermark",
                    "reasoning": "This is a general image generation task that requires detailed atmospheric effects and texturing, best handled by Qwen-Image",
                    "confidence_score": 0.96
                }}

            # Example 2 (Image Editing):
            - Given prompt: "Remove one person from the image"
            - Returned JSON:
                {{
                    "selected_model": "Qwen-Image-Edit",
                    "reference_content_image": "PATH/TO/GIVEN_IMAGE",
                    "generating_prompt": "Remove one person from the image",
                    "negative_prompt": "low quality, blurry, distorted, artifacts, seams, unnatural blending, watermark, text",
                    "reasoning": "This task requires precise object removal and background inpainting, which Qwen-Image-Edit excels at",
                    "confidence_score": 0.98
                }}

            # Example 3 (Style Enhancement):
            - Given prompt: "A portrait of a woman without makeup, no jewelry, not smiling, with no bright colors"
            - Returned JSON:
                {{
                    "selected_model": "Qwen-Image",
                    "reference_content_image": null,
                    "generating_prompt": "Professional portrait photography of a woman with natural, bare skin showing realistic texture and subtle imperfections. She gazes directly at the camera with a serene, thoughtful expression. Her hair is styled simply, falling naturally around her shoulders. She wears a plain, solid-colored top in a neutral tone. The background features soft bokeh in muted sage green and warm gray tones. The lighting is soft, diffused studio lighting from the front-left, creating gentle shadows that accentuate her facial structure.",
                    "negative_prompt": "makeup, jewelry, smiling, bright colors, harsh lighting, oversaturated, low quality, blurry, multiple faces, distorted features, watermark, text",
                    "reasoning": "This is a new portrait generation with specific style requirements, best handled by Qwen-Image",
                    "confidence_score": 0.95
                }}
            """
        else:
            return f"""Generate the most suitable prompt for the given task using Qwen-Image, including both positive and negative prompts.
            
            # Qwen-Image: {self.available_models['Qwen-Image']}

            Guidelines for generating_prompt:
            - CRITICAL: Stay faithful to the original prompt's core intent, subjects, and atmosphere
            - PRESERVE all original subjects, objects, and key elements mentioned in the original prompt
            - ENHANCE clarity and detail while maintaining the original scene's essence
            - For atmosphere/mood/lighting/style improvements:
                * Integrate specific atmospheric details into the original prompt
                * Use descriptive language for the desired mood or visual effect
                * Examples:
                    - "make it more dramatic" -> "A dramatic scene with high contrast lighting, deep shadows, and intense atmosphere, showing [original elements]"
                    - "enhance cozy feeling" -> "A warm and intimate setting with soft, golden lighting and comfortable ambiance, featuring [original elements]"
                    - "more professional atmosphere" -> "A polished and sophisticated environment with clean lines and professional lighting, showcasing [original elements]"
                    - "enhance ghostly features" -> "A haunting scene with ethereal, translucent ghostly elements that emit a subtle glow, featuring [original elements]"
                    - "make waves more dramatic" -> "A scene with powerful, dynamic waves with detailed foam and spray, showing [original elements]"
                    - "increase texture detail" -> "A scene with highly detailed surfaces, emphasizing intricate textures and fine details in [original elements]"
                    - "enhance lighting" -> "A scene with dramatic lighting effects, creating bold contrasts and atmospheric illumination for [original elements]"
             
            - Convert negative statements to positive ones:
                * "Do not wear a coat" -> "Wear a light sweater"
                * "No trees in background" -> "Clear blue sky background"
                * "Remove the hat" -> "Show full hair styling"
                * "Not smiling" -> "Serious expression"
                * "No bright colors" -> "Muted, subtle tones"
                
            - Structure prompts with clear spatial relationships:
                * Bad: "A vintage armchair. A sleeping cat. A Persian rug. Antique books. Wooden shelves."
                * Good: "A vintage leather armchair dominates the corner, its worn texture catching the ambient light. A cat sleeps peacefully on the Persian rug spread before it, while antique books line the wooden shelves along the wall."
            
            - The generating_prompt should read like an enhanced version of the original, not a replacement

            Guidelines for negative_prompt:
            - Include quality-related terms: "low quality, blurry, distorted, pixelated"
            - Include unwanted technical elements: "watermark, text, signature, border"
            - Include scene-specific unwanted elements based on the positive prompt
            - Keep it concise but comprehensive
            - Don't negate core elements of the positive prompt

            Return a JSON with:
            {{
                "selected_model": "Qwen-Image",
                "reference_content_image": "path to the reference content image",
                "generating_prompt": "model-specific prompt",
                "negative_prompt": "negative prompt to avoid unwanted elements",
                "reasoning": "Detailed explanation of why this model was chosen",
                "confidence_score": float  # 0.0 to 1.0
            }}

            # Example 1 (Scene with Atmosphere):
            - Given prompt: "An ancient dragon perched on a cliff overlooking a stormy ocean."
            - Returned JSON:
                {{
                    "selected_model": "Qwen-Image",
                    "reference_content_image": null,
                    "generating_prompt": "A colossal ancient dragon with highly detailed, weathered scales perches on a jagged cliff, its wings partially extended. Each scale features deep grooves and sharp ridges, reflecting flickers of ambient lightning. The background consists of a stormy ocean with hurricane-force winds and diagonal rain streaks. The sky is filled with dense, turbulent storm clouds, illuminated by intermittent lightning flashes that cast stark highlights on the dragon's body. Towering waves below crash violently against the rocky shore, creating dynamic white foam and spray patterns. The dragon is positioned as the dominant foreground element, with a cinematic contrast between its dark silhouette and the electric blue highlights from the storm.",
                    "negative_prompt": "low quality, blurry, distorted, cartoon style, cute dragon, bright sunny weather, calm ocean, people, buildings, modern objects, text, watermark",
                    "reasoning": "The prompt requires detailed atmospheric effects and complex texturing which Qwen-Image excels at",
                    "confidence_score": 0.96
                }}

            # Example 2 (Detailed Scene):
            - Given prompt: "A medieval fantasy marketplace"
            - Returned JSON:
                {{
                    "selected_model": "Qwen-Image",
                    "reference_content_image": null,
                    "generating_prompt": "A vibrant medieval marketplace in a fantasy setting, photographed in natural daylight. Wooden stalls with colorful canopies are arranged across a cobblestone plaza. Merchants display practical goods - rolls of fabric in earth tones, handcrafted pottery, and baskets of fresh produce. Several visitors in period-appropriate attire browse the marketplace. Stone and timber buildings with distinctive medieval architecture frame the scene. The atmosphere is lively yet realistic, with soft shadows cast by the midday sun. The image has a balanced composition with the marketplace as the clear focal point.",
                    "negative_prompt": "modern objects, cars, electricity, plastic, neon signs, contemporary clothing, low quality, blurry, oversaturated, text, watermark",
                    "reasoning": "This prompt requires creating a detailed scene with clear spatial relationships and balanced composition",
                    "confidence_score": 0.98
                }}

            # Example 3 (Portrait with Style):
            - Given prompt: "A portrait of a woman without makeup, no jewelry, not smiling, with no bright colors"
            - Returned JSON:
                {{
                    "selected_model": "Qwen-Image",
                    "reference_content_image": null,
                    "generating_prompt": "Professional portrait photography of a woman with natural, bare skin showing realistic texture and subtle imperfections. She gazes directly at the camera with a serene, thoughtful expression. Her hair is styled simply, falling naturally around her shoulders. She wears a plain, solid-colored top in a neutral tone. The background features soft bokeh in muted sage green and warm gray tones. The lighting is soft, diffused studio lighting from the front-left, creating gentle shadows that accentuate her facial structure.",
                    "negative_prompt": "makeup, jewelry, smiling, bright colors, harsh lighting, oversaturated, multiple faces, distorted features, low quality, blurry, watermark, text",
                    "reasoning": "This prompt requires generating a portrait with specific style elements while maintaining natural appearance",
                    "confidence_score": 0.95
                }}
            """

    def _create_task_prompt(self) -> str:
        """Create the task-specific prompt based on current state."""
        if config.regeneration_count > 0:
            prev_config = config.get_prev_config()
            current_config = config.get_current_config()
            reference_image_path = current_config.get('reference_content_image')
            
            if prev_config['user_feedback']:
                return f"""Analyze this regeneration request - IMPORTANT: A reference image is available from the previous generation.
                
                Reference image available: {reference_image_path}
                Previous result: {prev_config['gen_image_path']}
                User feedback: {prev_config['user_feedback']}
                Ultimate guiding principle prompt: {config.prompt_understanding['original_prompt']}
                First Round Prompt Understanding: {config.prompt_understanding}
                
                Since a reference image is available, consider Qwen-Image-Edit for targeted improvements and modifications based on user feedback."""
            else:
                return f"""Analyze this regeneration request - IMPORTANT: A reference image is available from the previous generation.
                
                Reference image available: {reference_image_path}
                Previous result: {prev_config['gen_image_path']}
                Improvement needed: {prev_config['improvement_suggestions']}
                Ultimate guiding principle prompt: {config.prompt_understanding['original_prompt']}
                First Round Prompt Understanding: {config.prompt_understanding}
                
                Since a reference image is available, consider Qwen-Image-Edit for targeted improvements and modifications."""
        else:
            if config.is_human_in_loop:
                return f"""Analyze this initial generation request:
                Original prompt: {config.prompt_understanding['original_prompt']}
                User clarification: {config.prompt_understanding['user_clarification']}
                Prompt understanding: {config.prompt_understanding}"""
            else:
                return f"""Analyze this initial generation request:
                Original prompt: {config.prompt_understanding['original_prompt']}
                Prompt understanding: {config.prompt_understanding}"""


    def select_model(self) -> Dict[str, Any]:
        """Analyze the refined prompt and select the most suitable model."""
        try:
            # Create base system prompt and task-specific prompt
            base_prompt = self._create_system_prompt()
            task_prompt = self._create_task_prompt()

            self.logger.info(f"System Prompt for model selection: {base_prompt}")
            self.logger.info(f"User Prompt for model selection: {task_prompt}")

            # Make the API call with structured prompts
            response = track_llm_call(self.llm_json.invoke, "model_selection", [
                ("system", base_prompt),
                ("human", task_prompt)
            ])

            result = self._parse_llm_response(response)
            
            # Ensure negative_prompt is present
            if "negative_prompt" not in result or not result["negative_prompt"]:
                self.logger.warning("No negative prompt in model selection result, generating one...")
                # Generate negative prompt based on the positive prompt
                analysis = None
                if hasattr(config, 'prompt_understanding') and config.prompt_understanding.get('prompt_analysis'):
                    try:
                        if isinstance(config.prompt_understanding['prompt_analysis'], str):
                            analysis = json.loads(config.prompt_understanding['prompt_analysis'])
                        else:
                            analysis = config.prompt_understanding['prompt_analysis']
                    except:
                        analysis = None
                
                neg_result = self.negative_prompt_generator.generate_negative_prompt(
                    result.get("generating_prompt", ""), 
                    analysis
                )
                result["negative_prompt"] = neg_result["negative_prompt"]
                self.logger.info(f"Generated negative prompt: {result['negative_prompt']}")

            return result

        except Exception as e:
            self.logger.error(f"Error in model selection: {str(e)}. Return basic configuration.")
            # Fallback to Qwen-Image with basic configuration including negative prompt
            fallback_prompt = config.prompt_understanding.get('refined_prompt', config.prompt_understanding['original_prompt'])
            
            # Generate a basic negative prompt for fallback
            basic_negative = "low quality, blurry, distorted, watermark, text, bad anatomy"
            
            return {
                "selected_model": "Qwen-Image",
                "reference_content_image": None,
                "generating_prompt": fallback_prompt,
                "negative_prompt": basic_negative,
                "reasoning": "Fallback selection due to error in model selection process",
                "confidence_score": 0.5
            }

    def _parse_llm_response(self, response) -> Dict[str, Any]:
        """Parse LLM response to dictionary. Handles string and dictionary response types
        and provides a fallback configuration if parsing fails."""
        try:
            if isinstance(response.content, str):
                return json.loads(response.content)
            elif isinstance(response.content, (dict, json)):
                return response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            self.logger.error(f"Raw response: {response.content}")
            # Return fallback configuration with minimal required fields
            return {
                "selected_model": "Qwen-Image",
                "reference_content_image": None,
                "generating_prompt": config.prompt_understanding.get('refined_prompt', config.prompt_understanding['original_prompt']),
                "reasoning": "Fallback selection due to JSON parsing error",
                "confidence_score": 0.5
            }

def intention_understanding_node(state: MessagesState) -> Command[str]:
    """Process user's initial prompt and handle interactions."""

    analyzer = IntentionAnalyzer(llm)
    last_message = state["messages"][-1]
    
    logger.info("-"*50)
    logger.info("INSIDE INTENTION UNDERSTANDING NODE")
    logger.info(f"Processing message: {last_message.content}")
    logger.info(f"Message count: {len(state["messages"])}")
    logger.info(f"Overall messages: {state["messages"]}")
    logger.info(f"Current config: {config.to_dict()}")

    # First interaction - analyze the prompt
    if len(state["messages"]) == 1:
        config.prompt_understanding["original_prompt"] = last_message.content
        logger.info("Step 1: Determining creativity level based on prompt")
        
        # Determine creativity level based on prompt analysis
        determined_creativity_level = analyzer.determine_creativity_level(last_message.content)
        config.prompt_understanding["creativity_level"] = determined_creativity_level
        logger.info(f"Determined creativity level: {determined_creativity_level.value}")
        
        logger.info("Step 2: Analyzing prompt")
        try:
            # Analyze prompt
            analysis = analyzer.analyze_prompt(last_message.content, config.prompt_understanding["creativity_level"])
            config.prompt_understanding["prompt_analysis"] = json.dumps(analysis)
            logger.info(f"Analysis result: {config.prompt_understanding['prompt_analysis']}")
            
            # Retrieve references info
            analyzer.retrieve_reference(analysis)
            logger.info(f"Current config for retrieved reference info:\n {config.to_dict()}")

            # Retrieve questions
            questions = analyzer.retrieve_questions(analysis, config.prompt_understanding["creativity_level"])
            logger.info(f"Suggested questions for users:\n {questions}")
            
            if questions == "SUFFICIENT_DETAIL" or questions == "AUTOCOMPLETE" or not config.is_human_in_loop:
                # Refine prompt directly
                refinement_result = analyzer.refine_prompt_with_analysis(
                    last_message.content,
                    analysis,
                    creativity_level=config.prompt_understanding["creativity_level"]
                )
                config.prompt_understanding["refined_prompt"] = refinement_result['refined_prompt']
                logger.info(f"With SUFFICIENT_DETAIL, AUTOCOMPLETE, or non-human-in-loop mode, Refinement result: {json.dumps(refinement_result, indent=2)}")
                
                command = Command(
                                update={"messages": state["messages"] + [AIMessage(content=f"Refined prompt: {config.prompt_understanding['refined_prompt']}")]},
                                goto="model_selection"
                            )
                logger.debug(f"Command: {command}")

                return command
            else:
                if config.is_human_in_loop:
                    config.prompt_understanding["questions"] = questions
                    logger.info(f"Need for information from users about: {questions}")
                    
                    # Clear the console output to make the prompt more visible
                    if os.name == 'posix':  # For Unix/Linux/MacOS
                        os.system('clear')
                    elif os.name == 'nt':   # For Windows
                        os.system('cls')

                    # Print a very distinctive prompt
                    print("\n" + "="*80)
                    print("USER INPUT REQUIRED - PLEASE RESPOND TO THE FOLLOWING QUESTIONS:")
                    print("="*80)
                    print(questions)
                    print("-"*80)

                    # Flush stdout to ensure prompt is displayed
                    sys.stdout.flush()

                    # Use a more robust input method
                    print("Enter your response below and press Enter when finished:")
                    user_responses = ""
                    try:
                        # This approach makes it clearer we're waiting for input
                        user_responses = input("> ")
                    except EOFError:
                        logger.error("Received EOF while waiting for input")
                        user_responses = "No input provided due to EOF"
                            
                    # Store user responses in config
                    config.prompt_understanding['user_clarification'] = user_responses
                    
                    # Parse user response
                    analysis = json.loads(config.prompt_understanding['prompt_analysis'])
                    
                    # Refine prompt with user input
                    refinement_result = analyzer.refine_prompt_with_analysis(
                        config.prompt_understanding['original_prompt'],
                        analysis,
                        user_responses,
                        config.prompt_understanding["creativity_level"]
                    )
                    config.prompt_understanding["refined_prompt"] = refinement_result['refined_prompt']
                    if 'suggested_creativity_level' in refinement_result:
                        if "LOW" in refinement_result['suggested_creativity_level']:
                            config.prompt_understanding["creativity_level"] = CreativityLevel.LOW
                        elif "MEDIUM" in refinement_result['suggested_creativity_level']:
                            config.prompt_understanding["creativity_level"] = CreativityLevel.MEDIUM
                        elif "HIGH" in refinement_result['suggested_creativity_level']:
                            config.prompt_understanding["creativity_level"] = CreativityLevel.HIGH
                        logger.info(f"Update creativity level to {config.prompt_understanding["creativity_level"]}")
                    
                    logger.info(f"Final refinement result: {json.dumps(refinement_result, indent=2)}")
                    
                    command = Command(
                                    update={"messages": state["messages"] + [AIMessage(content=f" User provides clarification. Refined prompt: {config.prompt_understanding['refined_prompt']}")]},
                                    goto="model_selection"
                                )
                    logger.debug(f"Command: {command}")
                    return command
                else:
                    # If not human_in_the_loop, proceed with auto-refinement
                    refinement_result = analyzer.refine_prompt_with_analysis(
                        last_message.content,
                        analysis,
                        creativity_level=CreativityLevel.HIGH
                    )
                    config.prompt_understanding["refined_prompt"] = refinement_result['refined_prompt']
                    logger.info(f"Auto-refinement result (non-human-in-loop): {json.dumps(refinement_result, indent=2)}")
                    
                    command = Command(
                                    update={"messages": state["messages"] + [AIMessage(content=f"LLM Refined prompt: {config.prompt_understanding['refined_prompt']}")]},
                                    goto="model_selection"
                                )
                    logger.debug(f"Command: {command}")
                return command
                
        except Exception as e:
            logger.error(f"Error in first interaction: {str(e)}")
            raise
    # Unexpected state
    else:
        logger.warning(f"Unexpected state: messages={last_message}")
        command = Command(
                            update={"messages": state["messages"]},
                            goto="model_selection"
                        )
        logger.debug(f"Command: {command}")
        return command

def model_selection_node(state: MessagesState) -> Command[str]:
    """Process model selection and image generation."""
    selector = ModelSelector(llm)
    
    logger.info("-"*50)
    logger.info("INSIDE MODEL SELECTION NODE")
    current_config = config.get_current_config()
    logger.info(f"Current config: {current_config}")
    if config.regeneration_count != 0:
        prev_regen_config = config.get_prev_config()
    try:
        # Select the most suitable model
        model_selection = selector.select_model()
        logger.debug(f"model_selection: {model_selection}")
        
        # Update current config with model selection
        current_config["selected_model"] = model_selection["selected_model"]
        logger.info(f"Selected model: {model_selection['selected_model']}")
        logger.info(f"Model selection reasoning: {model_selection.get('reasoning', 'No reasoning provided')}")
        
        if config.regeneration_count == 0:
            # First generation - set reference content image if provided
            if "reference_content_image" in model_selection:
                current_config["reference_content_image"] = model_selection["reference_content_image"]
            else:
                current_config["reference_content_image"] = None
            logger.info(f"First generation - reference image: {current_config['reference_content_image']}")
        else:
            # Regeneration - ensure the reference content image is preserved from add_regeneration_config
            logger.info(f"Regeneration attempt {config.regeneration_count}")
            logger.info(f"Current reference image: {current_config.get('reference_content_image')}")
            # Only update if the LLM explicitly provided a different reference
            if "reference_content_image" in model_selection and model_selection["reference_content_image"]:
                current_config["reference_content_image"] = model_selection["reference_content_image"]
                logger.info(f"Updated reference image from LLM: {current_config['reference_content_image']}")
        current_config["generating_prompt"] = model_selection["generating_prompt"]
        current_config["negative_prompt"] = model_selection["negative_prompt"]  # Add negative prompt to config
        current_config["reasoning"] = model_selection["reasoning"]
        current_config["confidence_score"] = model_selection["confidence_score"]

        # Execute the selected model with negative prompt
        gen_image_path = execute_model(
            model_name=current_config['selected_model'],
            prompt=current_config['generating_prompt'],
            negative_prompt=current_config['negative_prompt'],  # Pass negative prompt
            reference_content_image=current_config['reference_content_image'],
        )
        
        current_config["gen_image_path"] = gen_image_path

        command = Command(
            update={"messages": state["messages"] + [
                AIMessage(content=f"Generated images using {current_config['selected_model']}. "
                         f"Image path saved in: {current_config['gen_image_path']}. "
                         f"Negative prompt: {current_config['negative_prompt']}")  # Include negative prompt in output
            ]},
            goto=END
        )
        logger.debug(f"Command: {command}")
        return command

    except Exception as e:
        logger.error(f"Error in model selection: {str(e)}")

        # Check if this is a regeneration attempt
        if config.regeneration_count > 0:
            # Get previous config and use its generated image path
            prev_config = config.get_prev_config()
            prev_image_path = prev_config["gen_image_path"]
            logger.info(f"Returning previous generated image due to error: {prev_image_path}")
            
            # Update current config with previous image path
            current_config["gen_image_path"] = prev_image_path
            
            return Command(
                update={"messages": state["messages"] + [
                    AIMessage(content=f"Error occurred during regeneration. Using previous generated image: {prev_image_path}")
                ]},
                goto=END
            )
        
        # If not a regeneration, raise the error
        raise

def normalize_image(image):
    """Normalize image data to ensure valid pixel values before saving.
    
    Args:
        image: A PIL Image or tensor image
    
    Returns:
        PIL Image with valid pixel values
    """
    if isinstance(image, torch.Tensor):
        # If it's a tensor, convert to numpy first
        image = image.cpu().numpy()
    
    if isinstance(image, np.ndarray):
        # Handle NaN values
        image = np.nan_to_num(image, nan=0.5)
        # Clamp values to [0, 1] range
        image = np.clip(image, 0, 1)
        # Convert to PIL Image if necessary
        if image.dtype != np.uint8:
            image = (image * 255).round().astype(np.uint8)
        image = Image.fromarray(image)
    
    return image


def polish_prompt_en(original_prompt):
    
    SYSTEM_PROMPT = '''
You are a Prompt optimizer designed to rewrite user inputs into high-quality Prompts that are more complete and expressive while preserving the original meaning.
Task Requirements:
1. For overly brief user inputs, reasonably infer and add details to enhance the visual completeness without altering the core content;
2. Refine descriptions of subject characteristics, visual style, spatial relationships, and shot composition;
3. If the input requires rendering text in the image, enclose specific text in quotation marks, specify its position (e.g., top-left corner, bottom-right corner) and style. This text should remain unaltered and not translated;
4. Match the Prompt to a precise, niche style aligned with the user's intent. If unspecified, choose the most appropriate style (e.g., realistic photography style);
5. Please ensure that the Rewritten Prompt is less than 200 words.
'''
    original_prompt = original_prompt.strip()
    prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {original_prompt}\n\n Rewritten Prompt:"
    magic_prompt = "Ultra HD, 4K, cinematic composition"
            
    response = track_llm_call(llm.invoke, "polish_prompt", [
                    ("system", SYSTEM_PROMPT),
                    ("human", f"User Input: {original_prompt}\n\n Rewritten Prompt:")
                ])
    polished_prompt = response.content.strip()
    polished_prompt = polished_prompt.replace("\n", " ")
            
    return polished_prompt + " " + magic_prompt

def execute_model(model_name: str, prompt: str, negative_prompt: str, reference_content_image: str = None) -> str:
    """Execute the selected model and return paths to generated images."""
    global qwen_image_pipe, qwen_edit_pipe
    global model_inference_times
    selector = ModelSelector(llm)
    if model_name not in selector.tools:
        raise ValueError(f"Unknown model: {model_name}")

    # get regen count
    regen_count = config.regeneration_count
    if regen_count == 0:
        seed = config.seed
    else:
        # random seed
        seed = random.randint(0, 1000000)
    
    polished_prompt = polish_prompt_en(prompt)
    # print("Polished Prompt:", polished_prompt)
    print("="*50)
    print("Prompt",prompt)
    print("Polished Prompt:", polished_prompt)
    print("Negative Prompt:", negative_prompt)
    print("="*50)

    if model_name == "Qwen-Image":
        t0 = time.time()
        result = selector.tools[model_name].invoke({"prompt": polished_prompt, "negative_prompt": negative_prompt, "seed": seed})
        t1 = time.time()
        model_inference_times["Qwen-Image"].append(t1 - t0)
        return result
    elif model_name == "Qwen-Image-Edit":
        t0 = time.time()
        logger.info(f"existing_image_dir: {reference_content_image}")
            
        result = selector.tools[model_name].invoke({"prompt": polished_prompt, "negative_prompt": negative_prompt, "existing_image_dir": reference_content_image, "seed": seed, "guidance_scale": 4.0})
        t1 = time.time()
        model_inference_times["Qwen-Image-Edit"].append(t1 - t0)
        return result
    else:
        raise ValueError(f"Unknown model: {model_name}")

def evaluation_node(state: MessagesState) -> Command[str]:
    """Evaluate generated images and handle regeneration if needed."""
    last_message = state["messages"][-1]
    
    logger.info("-"*50)
    logger.info("INSIDE EVALUATION NODE")
    logger.info(f"Current config: {config.to_dict()}")
    
    try:
        current_config = config.get_current_config()
        
        # Prepare evaluation prompt
        with open(current_config['gen_image_path'], "rb") as image_file:
            base64_gen_image = base64.b64encode(image_file.read()).decode("utf-8")
        evaluation_prompt = [
            (
                "system",
                make_gen_image_judge_prompt(config)
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": f"original prompt: {config.prompt_understanding['original_prompt']}\n Prompt analysis: {config.prompt_understanding['prompt_analysis']}\n Prompt used for generating the image: {current_config['generating_prompt']}\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_gen_image}"
                        }
                    }
                ]
            )
        ]
        
        # Get evaluation from Qwen-VL
        evaluation_result = track_llm_call(llm_json.invoke, "evaluation", evaluation_prompt)
        evaluation_data = json.loads(evaluation_result.content)

        # Update config with evaluation score
        current_config["evaluation_score"] = evaluation_data["overall_score"]
        current_config["improvement_suggestions"] = evaluation_data["improvement_suggestions"]
        logger.info(f"Evaluation result: {json.dumps(evaluation_data, indent=2)}")
        
        # Define threshold for acceptable quality
        QUALITY_THRESHOLD = 8.0
        MAX_REGENERATION_ATTEMPTS = 3

        # Log the evaluation score and threshold
        logger.debug(f"Evaluation score: {current_config['evaluation_score']}")
        logger.debug(f"Quality threshold: {QUALITY_THRESHOLD}")
        logger.debug(f"Regeneration count: {config.regeneration_count}")
        
        if config.is_human_in_loop:
            
            logger.info("Human-in-loop is enabled, requesting user feedback.")
            print("\nCurrent evaluation score:", current_config['evaluation_score'])
            print("Evaluation feedback:", current_config['improvement_suggestions'])
            # TODO: visualize the image for user to evalute
            
            feedback_type = input("\nWould you like to provide feedback?\n0. I like the image and no need to regenerate\n1. Provide text suggestions\n2. Skip feedback but want regenerate the image\nEnter choice (0-2): ")
            if feedback_type == "0":
                logger.info("User likes the image, no need to regenerate.")
                final_message = (
                    f"Final image generated with score: {current_config['evaluation_score']}\n"
                    f"Detailed feedback: {current_config['improvement_suggestions']}\n"
                    f"User feedback: User likes the image, no need to regenerate."
                )
                comment = Command(
                    update={"messages": state["messages"] + [AIMessage(content=final_message)]},
                    goto=END
                )
                return comment
            elif feedback_type == "1":
                user_suggestion = input("Enter your suggestions: ")
                current_config["user_feedback"] = user_suggestion
            
            elif feedback_type == "2":
                logger.info("Drawing-based edits are not supported in this version.")
                print("Drawing-based edits are not supported in this version. Please provide text feedback instead.")
                user_suggestion = input("Enter your text suggestions: ")
                current_config["user_feedback"] = user_suggestion

            elif feedback_type == "3":
                logger.info("User skip feeback, but want regenerate the image")
                current_config["user_feedback"] = "User skip feeback, but want regenerate the image"
            
            regen_key = config.add_regeneration_config()
            current_config = config.get_current_config()
            
            return Command(
                update={
                    "messages": state["messages"] + [AIMessage(content= f"User suggestions: {current_config['user_feedback']}")],  
                },
                goto="model_selection"
            )

        elif current_config['evaluation_score'] < QUALITY_THRESHOLD and config.regeneration_count < (MAX_REGENERATION_ATTEMPTS-1):
            logger.info("Image quality below threshold, preparing to regenerate.")
            # Increment regeneration counter
            regen_key = config.add_regeneration_config()
            
            # Update state and return to model selection
            logger.info(f"Regenerating image (attempt {config.regeneration_count})")
            return Command(
                update={
                    "messages": state["messages"] + [
                        AIMessage(content=f"Image quality below threshold ({current_config['evaluation_score']} < {QUALITY_THRESHOLD}). Regenerating with feedback.")
                    ]
                },
                goto="model_selection"
            )
        
        # If quality is acceptable or max attempts reached
        final_message = (
            f"Final image generated with score: {current_config['evaluation_score']}\n"
            f"Detailed feedback: {current_config['improvement_suggestions']}"
        )
        
        if config.regeneration_count >= MAX_REGENERATION_ATTEMPTS and not config.is_human_in_loop:
            final_message += f"\nReached maximum regeneration attempts ({MAX_REGENERATION_ATTEMPTS})"
        
        logger.info("Image quality is acceptable or maximum regeneration attempts reached.")
        comment = Command(
            update={"messages": state["messages"] + [AIMessage(content=final_message)]},
            goto=END
        )
        return comment
    
    except Exception as e:
        logger.error(f"Error in evaluation node: {str(e)}")

        # Check if this is a regeneration attempt
        if config.regeneration_count > 0:
            # Get previous config and use its generated image path
            prev_config = config.get_prev_config()
            prev_image_path = prev_config["gen_image_path"]
            logger.info(f"Returning previous generated image due to error: {prev_image_path}")
            
            final_message = (
                f"Error occurred during evaluation. Using previous generated image: {prev_image_path}\n"
                f"Error details: {str(e)}"
            )
            
            return Command(
                update={"messages": state["messages"] + [AIMessage(content=final_message)]},
                goto=END
            )
        
        # If not a regeneration, raise the error
        raise

# Create the workflow
def create_t2i_workflow():
    """Create the T2I workflow with appropriate settings."""
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("intention", intention_understanding_node)
    workflow.add_node("model_selection", model_selection_node)
    workflow.add_node("evaluation", evaluation_node)
    
    workflow.add_edge(START, "intention")
    workflow.add_edge("intention", "model_selection")
    workflow.add_edge("model_selection", "evaluation")
    workflow.add_edge("evaluation", END)
    
    # Compile workflow
    compiled_workflow = workflow.compile()
    
    return compiled_workflow

def run_workflow(workflow: StateGraph, initial_prompt: str):
    """Run the T2I workflow with human interaction."""
    
    # Start workflow with initial state
    result = workflow.invoke(
        {"messages": [HumanMessage(content=initial_prompt)]},
    )
    
    return result

def track_llm_call(llm_func, llm_type, *args, **kwargs):
    global llm_latencies, llm_token_counts
    start = time.time()
    response = llm_func(*args, **kwargs)
    end = time.time()
    latency = end - start
    # Try to get token usage from different possible locations
    prompt_tokens = completion_tokens = total_tokens = 0
    # 1. langchain_openai new ChatOpenAI usage_metadata
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        prompt_tokens = usage.get('input_tokens', 0)
        completion_tokens = usage.get('output_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
    # 2. openai API style
    elif hasattr(response, "usage") and response.usage:
        usage = response.usage
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
    # 3. langchain_core.messages.AIMessage style (sometimes usage is in .additional_kwargs)
    elif hasattr(response, "additional_kwargs") and response.additional_kwargs:
        usage = response.additional_kwargs.get("usage", {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
    # 4. or just 0 if not found
    llm_latencies[llm_type].append(latency)
    llm_token_counts[llm_type].append((prompt_tokens, completion_tokens, total_tokens))
    return response

def main(benchmark_name, human_in_the_loop, model_version, use_open_llm=False, open_llm_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503", open_llm_host="0.0.0.0", open_llm_port="8000", calculate_latency=False, use_quantization=True):
    """Main CLI entry point."""
    # Check CUDA availability and initialize
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires a GPU to run.")
    
    # Initialize primary CUDA device
    torch.cuda.init()
    torch.cuda.set_device(0)
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nUsing GPU: {gpu_name}")
    print(f"Available GPU memory: {gpu_memory:.2f} GB")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    # Declare globals
    global logger, config
    global llm_latencies, llm_token_counts
    global model_inference_times
    global llm, llm_json

    # Initialize LLMs based on open_llm flag
    llm, llm_json = initialize_llms(use_open_llm, open_llm_model=open_llm_model, local_host=open_llm_host, local_port=open_llm_port)

    # Get workflow
    workflow = create_t2i_workflow()

    # Load models after logger and save_dir are set
    load_models(use_quantization)
    print("Models loaded")

    # store benchmark dir into one list
    benchmark_list = []
    benchmark_list.append(os.path.join("eval_benchmark/", benchmark_name))

    for benchmark_dir in benchmark_list:

        # Read prompts from the file
        with open(benchmark_dir, 'r') as file:
            if "DrawBench" in str(benchmark_name):
                if "seed" in str(benchmark_name):
                    lines = [line.strip().split('\t') for line in file]
                    prompts = [line[0] for line in lines]
                    seeds = [int(line[1]) for line in lines]
                    bench_result_folder = 'DrawBench-fixseed'
                else:
                    prompts = [line.strip().split('\t')[0] for line in file]
                    bench_result_folder = 'DrawBench'
                prompt_keys = prompts
            elif "GenAIBenchmark" in str(benchmark_name):
                prompts = json.load(file)
                prompt_keys = list(prompts.keys())
                bench_result_folder = 'GenAIBenchmark-fixseed'
            else:
                prompts = json.load(file)
                prompt_keys = list(prompts.keys())
                bench_result_folder = "123"
            # else:
            #     lines = [line.strip().split('\t') for line in file]
            #     prompts = [line[0] for line in lines]
            #     seeds = [int(line[1]) for line in lines]
            #     prompt_keys = prompts
            #     bench_result_folder = os.path.basename(benchmark_dir)

        # Create model type suffix for directory
        model_suffix = model_version
        if use_open_llm:
            # Get model name for the suffix - extract just the model name without org prefix
            model_name = open_llm_model.split('/')[-1]
            model_suffix += f"_open_llm_{model_name}"
        
        # Extract part number from benchmark_name for GPU-specific directories
        part_number = None
        if "part_" in str(benchmark_name):
            import re
            match = re.search(r'part_(\d+)', str(benchmark_name))
            if match:
                part_number = match.group(1)
        
        # Create GPU-specific directory structure
        base_dir_name = f"AgentSys_{model_suffix}_human_in_loop" if human_in_the_loop else f"AgentSys_{model_suffix}"
        
        if part_number:
            # For parallel GPU runs, create part-specific subdirectories
            save_dir = os.path.join("results", bench_result_folder, base_dir_name, f"part_{part_number}")
            print(f"Using GPU-specific results directory: {save_dir}")
        else:
            # For single runs or complete datasets
            save_dir = os.path.join("results", bench_result_folder, base_dir_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created results directory: {save_dir}")

        # Load progress if exists
        progress_file = os.path.join(save_dir, "progress.json")
        if os.path.exists(progress_file):
            with open(progress_file, "r") as file:
                progress_data = json.load(file)
                total_time = progress_data.get("total_time", 0)
                start_idx = progress_data.get("current_idx", 0)
                inference_times = progress_data.get("inference_times", [])
        else:
            total_time = 0
            start_idx = 0
            inference_times = []

        # Calculate the latency and token counts
        single_turn_times = []
        multi_turn_times = []
        end2end_times = []
        model_inference_times = {"Qwen-Image": [], "Qwen-Image-Edit": []}
        llm_latencies = {
            "intention_analysis": [], 
            "refine_prompt": [], 
            "model_selection": [], 
            "evaluation": [],
            "negative_prompt_generation": [],
            "polish_prompt": [],
            "creativity_determination": []
        }
        llm_token_counts = {
            "intention_analysis": [], 
            "refine_prompt": [], 
            "model_selection": [], 
            "evaluation": [],
            "negative_prompt_generation": [],
            "polish_prompt": [],
            "creativity_determination": []
        }
        single_turn_count = 0
        multi_turn_count = 0
        multi_turn_turns = []
        max_gpu_memories = []
        # resume stats
        stats_file = os.path.join(save_dir, "stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            single_turn_times = stats.get("single_turn_times", single_turn_times)
            multi_turn_times = stats.get("multi_turn_times", multi_turn_times)
            end2end_times = stats.get("end2end_times", end2end_times)
            model_inference_times = stats.get("model_inference_times", model_inference_times)
            llm_latencies = stats.get("llm_latencies", llm_latencies)
            llm_token_counts = stats.get("llm_token_counts", llm_token_counts)
            single_turn_count = stats.get("single_turn_count", single_turn_count)
            multi_turn_count = stats.get("multi_turn_count", multi_turn_count)
            multi_turn_turns = stats.get("multi_turn_turns", multi_turn_turns)
            total_time = stats.get("total_time", total_time)
            inference_times = stats.get("inference_times", inference_times)
            max_gpu_memories = stats.get("max_gpu_memories", max_gpu_memories)
            

        for idx, key in tqdm(enumerate(prompt_keys[start_idx:]), total=len(prompt_keys) - start_idx, desc=f"Processing {benchmark_name}"):
            # Calculate actual index
            actual_idx = idx + start_idx
            
            # Initialize a fresh config for each prompt
            config = T2IConfig(human_in_loop=human_in_the_loop)
            config.save_dir = save_dir
            config.use_open_llm = use_open_llm 
            config.open_llm_model = open_llm_model 
            config.open_llm_host = open_llm_host 
            config.open_llm_port = open_llm_port 
            
            # Handle different benchmark types consistently
            if "GenAIBenchmark" in str(benchmark_name):
                text_prompt = prompts[key]['prompt'] 
                # Always use the random_seed from the JSON file if available
                config.seed = prompts[key].get("random_seed", torch.initial_seed())
                config.image_index = prompts[key]['id']
            else:  # DrawBench or other benchmarks
                text_prompt = key
                config.seed = seeds[actual_idx] if "seed" in str(benchmark_name) and 'seeds' in locals() else torch.initial_seed()
                config.image_index = f"{actual_idx:05d}"
                print(f"Working on Benchmark name: {benchmark_name}")

            # Setup logging for this iteration
            logger = setup_logging(save_dir, filename=f"{config.image_index}.log", console_output=False)
            config.logger = logger

            # start timing
            torch.cuda.reset_peak_memory_stats(0)
            start_time = time.time()

            logger.info("\n" + "="*83)
            logger.info(f"New Session Started for index {config.image_index}: {datetime.now()}")
            logger.info(f"Processing prompt: {text_prompt}")
            logger.info(f"Using seed: {config.seed}")
            logger.info(f"Results will be saved to: {save_dir}")
            logger.info("="*83)

            try:
                logger.info(f"Starting workflow with prompt: {text_prompt}, seed: {config.seed}")
                result = run_workflow(workflow, text_prompt)

                # Save config state after generation
                config_save_path = os.path.join(save_dir, f"{config.image_index}_config.json")
                config.save_to_file(config_save_path)
                logger.info(f"Saved config state to: {config_save_path}")
                
                # Log the generated image path if available
                current_config = config.get_current_config()
                if current_config.get("gen_image_path"):
                    logger.info(f"Generated image saved to: {current_config['gen_image_path']}")
                    print(f"✅ Image {config.image_index} saved: {current_config['gen_image_path']}")
                else:
                    logger.warning(f"No image path found for {config.image_index}")
                    
            except Exception as e:
                logger.error(f"Error processing prompt {config.image_index}: {str(e)}")
                print(f"❌ Error processing image {config.image_index}: {str(e)}")
                continue
            # End timing & record stats
            end_time = time.time()
            inference_time = end_time - start_time
            end2end_times.append(inference_time)
            
            # Determine single/multi-turn
            if config.regeneration_count == 0:
                single_turn_times.append(inference_time)
                single_turn_count += 1
            else:
                multi_turn_times.append(inference_time)
                multi_turn_count += 1
                multi_turn_turns.append(config.regeneration_count + 1)
            total_time += inference_time
            inference_times.append(inference_time)
            
            # Safely collect GPU memory stats
            try:
                mem_stats = []
                for device in range(torch.cuda.device_count()):
                    mem_stats.append(torch.cuda.max_memory_allocated(device) / 1024**3)
                max_gpu_memories.append(max(mem_stats) if mem_stats else 0)
            except RuntimeError as e:
                logger.warning(f"Could not collect GPU memory stats: {e}")
                max_gpu_memories.append(0)
                
            logger.info(f"Inference time for prompt {config.image_index}: {inference_time:.4f} seconds")
            logger.info("Workflow completed")

            # Save progress after each successful completion
            progress_data = {
                "total_time": total_time,
                "current_idx": actual_idx + 1,
                "inference_times": inference_times,
                "completed_prompts": actual_idx + 1,
                "total_prompts": len(prompt_keys)
            }
            with open(progress_file, "w") as file:
                json.dump(progress_data, file, indent=2)  
                
            # Save stats.json after each completion
            stats = {
                "single_turn_times": single_turn_times,
                "multi_turn_times": multi_turn_times,
                "end2end_times": end2end_times,
                "model_inference_times": model_inference_times,
                "llm_latencies": llm_latencies,
                "llm_token_counts": llm_token_counts,
                "single_turn_count": single_turn_count,
                "multi_turn_count": multi_turn_count,
                "multi_turn_turns": multi_turn_turns,
                "total_time": total_time,
                "inference_times": inference_times,
                "max_gpu_memories": max_gpu_memories,
                "benchmark_name": benchmark_name,
                "part_number": part_number if part_number else "complete"
            }
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
                
            print(f"Progress: {actual_idx + 1}/{len(prompt_keys)} completed. Time: {inference_time:.2f}s")
        
        # Calculate and print average time
        # avg_time = total_time / progress_data["current_idx"]
        # print(f"\nAverage inference time per image: {avg_time:.4f} seconds")
        # print(f"Total time for {progress_data["current_idx"]} images: {total_time:.4f} seconds")
        
        # summary output
        if calculate_latency:
            total_prompts = single_turn_count + multi_turn_count
            print("\n==== Statistics ====")
            print(f"Single-turn avg time: {sum(single_turn_times)/len(single_turn_times) if single_turn_times else 0:.4f} s")
            print(f"Multi-turn avg time: {sum(multi_turn_times)/len(multi_turn_times) if multi_turn_times else 0:.4f} s")
            print(f"End-to-end avg time: {sum(end2end_times)/len(end2end_times) if end2end_times else 0:.4f} s")
            for model in model_inference_times:
                times = model_inference_times[model]
                print(f"{model} avg inference time: {sum(times)/len(times) if times else 0:.4f} s, total: {sum(times):.2f} s")
            for k in llm_latencies:
                if llm_latencies[k]:
                    print(f"LLM {k} avg latency: {sum(llm_latencies[k])/len(llm_latencies[k]):.4f} s")
                else:
                    print(f"LLM {k} avg latency: 0.0000 s (fail to log)")
            for k in llm_token_counts:
                if llm_token_counts[k]:
                    avg_prompt = sum(x[0] for x in llm_token_counts[k]) / len(llm_token_counts[k])
                    avg_completion = sum(x[1] for x in llm_token_counts[k]) / len(llm_token_counts[k])
                    avg_total = sum(x[2] for x in llm_token_counts[k]) / len(llm_token_counts[k])
                    print(f"LLM {k} avg prompt tokens: {avg_prompt:.2f}, completion tokens: {avg_completion:.2f}, total tokens: {avg_total:.2f}")
                else:
                    print(f"LLM {k} avg prompt tokens: 0.00, completion tokens: 0.00, total tokens: 0.00  (fail to log)")
            print(f"Single-turn end count: {single_turn_count} ({single_turn_count/total_prompts*100 if total_prompts else 0:.2f}%)")
            print(f"Multi-turn end count: {multi_turn_count} ({multi_turn_count/total_prompts*100 if total_prompts else 0:.2f}%)")
            if multi_turn_turns:
                print(f"Multi-turn average turns: {sum(multi_turn_turns)/len(multi_turn_turns):.2f}")
            else:
                print("Multi-turn average turns: 0")
            if max_gpu_memories:
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU max memory usage: {max(max_gpu_memories):.2f} GB / {total_gpu_memory:.2f} GB")
            else:
                print("GPU max memory usage: 0.00 GB (fail to log)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2I-Copilot Agent System")
    parser.add_argument('--benchmark_name', default='cool_sample.txt', type=str, help='Path to the benchmark directory: cool_sample.txt, GenAIBenchmark/genai_image_seed.json, ambiguous_seed.txt')
    parser.add_argument('--human_in_the_loop', action='store_true', help='Use human in the loop')
    parser.add_argument('--model_version', default='vRelease', type=str, help= 'Model version')
    parser.add_argument('--use_open_llm', action='store_true', help='Use open source LLM.')
    parser.add_argument('--open_llm_model', default='mistralai/Mistral-Small-3.1-24B-Instruct-2503', type=str, 
                        help='Open LLM model to use (mistralai/Mistral-Small-3.1-24B-Instruct-2503, Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct)')
    parser.add_argument('--open_llm_host', default='0.0.0.0', type=str, help='Host address for the open LLM server')
    parser.add_argument('--open_llm_port', default='8000', type=str, help='Port for the open LLM server')
    parser.add_argument('--calculate_latency', action='store_true', help='Calculate and print latency statistics')

    args = parser.parse_args()
    main(args.benchmark_name, args.human_in_the_loop, args.model_version, args.use_open_llm, args.open_llm_model, args.open_llm_host, args.open_llm_port, args.calculate_latency)