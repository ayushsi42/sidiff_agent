import os
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum
import sqlite3
import json
import os
import time
import pickle
import numpy as np
from transformers import pipeline  # AutoModel, 
import torch
import faiss
import faiss.contrib.torch_utils
NODE_MODEL_SELECTION = "negative_model_selection"
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
from prompts.system_prompts_memory import make_intention_analysis_prompt, make_gen_image_judge_prompt

import time
from tqdm import tqdm

# Initialize model variables
qwen_image_pipe = None
qwen_edit_pipe = None

# Initialize embedding model variable
embedding_model = None

# Initialize LLM variables
llm = None
llm_json = None

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
        self.category = None  # For CSV benchmarks with categories
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

        # RAG-based workflow guidance
        self.workflow_guidance = {
            "positive_guidance": {},  # What worked well for similar prompts
            "unsuccessful_patterns": {}   # What to avoid based on similar prompt failures
        }
        
        # RAG guidance retrieval flag
        self.rag_guidance_retrieved = False

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
                prompt_understanding["prompt_analysis"] = {}
            else:
                try:
                    prompt_understanding["prompt_analysis"] = json.loads(prompt_understanding["prompt_analysis"])
                except json.JSONDecodeError:
                    self.logger.error("Invalid JSON string in prompt_analysis")
                    try:
                        # Try to preserve the string content if it's not JSON
                        prompt_understanding["prompt_analysis"] = {
                            "raw_content": prompt_understanding["prompt_analysis"]
                        }
                    except Exception as e:
                        self.logger.error(f"Failed to preserve prompt analysis content: {e}")
                        prompt_understanding["prompt_analysis"] = {}
        elif isinstance(prompt_understanding["prompt_analysis"], dict):
            pass  # Already a dictionary
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

    def determine_creativity_level(self, prompt: str, workflow_context: Dict[str, Any] = None) -> CreativityLevel:
        """Analyze the prompt to automatically determine the appropriate creativity level."""
        
        # Include workflow context guidance in the analysis prompt
        guidance_text = ""
        if workflow_context:
            workflow_guidance = format_workflow_guidance_text(workflow_context, include_overview=True)
            guidance_text = f"\n\n{workflow_guidance}"
        
        base_creativity_analysis_prompt = """You are an expert at analyzing image generation prompts to determine the appropriate creativity level.

The PRIMARY RULE for assessing creativity level is: Shorter and less informed prompts require HIGH creativity to detailed and well-specified prompts require LOW creativity.

Analyze the given prompt and determine the creativity level based on these criteria:

HIGH Creativity Level (system should be highly creative and autonomous):
- Very brief or vague prompts with minimal information (e.g., "a black cat", "a blue landscape", "a beautiful sunset")
- Single-phrase or short sentence prompts lacking descriptive details
- Abstract concepts or artistic requests with minimal guidance (e.g., "surreal dream", "impressionist style")
- Prompts with numerous undefined elements requiring creative decisions
- Prompts that offer minimal context, leaving most details to be creatively determined
- Word count typically under 10 words

MEDIUM Creativity Level (balanced approach):
- Prompts with moderate detail but still containing unspecified aspects
- Prompts specifying subject and some context but lacking specific style or compositional elements
- Standard scene descriptions that mention key elements but leave secondary elements unspecified
- Prompts with a balance of specific instructions and areas requiring creative interpretation
- Word count typically between 10-25 words

LOW Creativity Level (stick closely to specifications):
- Highly detailed and comprehensive prompts with explicit requirements
- Technical or precise requests with specific parameters (e.g., "professional headshot photo with precise lighting setup")
- Prompts that explicitly specify style, composition, colors, lighting, background, and other details
- Professional or commercial image requests with clear technical specifications
- Prompts that leave very little room for creative interpretation
- Word count typically over 25 words with numerous specific details

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
    "reasoning": "Extremely brief prompt with no details about breed, color, pose, setting, lighting, or style. System must autonomously determine all visual elements and composition.",
    "prompt_characteristics": {"detail_level": "low", "specificity": "vague", "artistic_freedom": "open"}
}

Input: "sunset on the beach"
Output: {
    "creativity_level": "HIGH",
    "reasoning": "Brief prompt that only specifies basic scene elements. Requires creative decisions about color palette, composition, foreground elements, beach type, mood, and all other visual details.",
    "prompt_characteristics": {"detail_level": "low", "specificity": "vague", "artistic_freedom": "open"}
}

Input: "A medieval marketplace with people shopping and vendors selling goods"
Output: {
    "creativity_level": "MEDIUM",
    "reasoning": "Prompt has clear subject and basic activity but leaves many specifics undefined (architecture style, time of day, types of goods, clothing styles, weather, atmosphere). Contains 11 words with moderate detail level.",
    "prompt_characteristics": {"detail_level": "medium", "specificity": "moderate", "artistic_freedom": "balanced"}
}

Input: "Professional headshot of a 30-year-old woman with shoulder-length brown hair, wearing a navy blue blazer, neutral beige background, studio lighting with soft key light from left side"
Output: {
    "creativity_level": "LOW", 
    "reasoning": "Extremely detailed prompt (24 words) with explicit specifications for subject, age, hair length, hair color, clothing, background color, lighting setup and direction. Almost all creative decisions have been predetermined.",
    "prompt_characteristics": {"detail_level": "high", "specificity": "precise", "artistic_freedom": "constrained"}
}"""

        # Construct the full system prompt with guidance BEFORE the system prompt
        full_system_prompt = guidance_text + "\n\n" + base_creativity_analysis_prompt
        
        # === STEP 1: CREATIVITY LEVEL DETERMINATION ===
        # Always log creativity level determination as it's a core workflow step
        self.logger.info(f"=== STEP 1: CREATIVITY LEVEL DETERMINATION ===")
        self.logger.info(f"System Prompt Length: {len(full_system_prompt)} characters")
        self.logger.info(f"System Prompt:\n{full_system_prompt}")
        self.logger.info(f"Human Prompt: Analyze this prompt and determine creativity level: '{prompt}' (Word count: {len(prompt.split())})")

        try:
            # Count the number of words in the prompt for additional context
            word_count = len(prompt.split())
            
            # Include word count in the analysis request
            response = track_llm_call(self.llm_json.invoke, "creativity_determination", [
                ("system", full_system_prompt),
                ("human", f"Analyze this prompt and determine creativity level: '{prompt}'\n\nAdditional context: This prompt contains {word_count} words.")
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
                
            # Apply word count heuristic as a fallback or sanity check
            word_count = len(prompt.split())
            if word_count <= 5 and creativity_level != CreativityLevel.HIGH:
                creativity_level = CreativityLevel.HIGH
            elif word_count >= 25 and creativity_level != CreativityLevel.LOW:
                creativity_level = CreativityLevel.LOW
                
            # === STEP 1 OUTPUT ===
            self.logger.info(f"=== STEP 1 OUTPUT: CREATIVITY LEVEL = {creativity_level.value.upper()} ===")
            self.logger.info(f"Reasoning: {reasoning}")
            
            return creativity_level
            
        except Exception as e:
            # Apply word count heuristic as fallback when LLM fails
            word_count = len(prompt.split())
            self.logger.error(f"Error in creativity determination: {str(e)}. Using word count heuristic.")
            
            if word_count <= 5:
                return CreativityLevel.HIGH
            elif word_count >= 25:
                return CreativityLevel.LOW
            else:
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
                return path_or_url, "url"
            else:
                # It's a local file path
                full_path = os.path.abspath(os.path.expanduser(path_or_url))
                if os.path.exists(full_path):
                    return full_path, "local"
                else:
                    self.logger.error(f"Image '{full_path}' not found.")
                    return None, None
        
        return None, None

    def analyze_prompt(self, prompt: str, creativity_level: CreativityLevel, workflow_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the prompt and identify elements that need clarification."""
        
        guidance_text = ""
        # Use workflow guidance as the primary guidance source (includes processed RAG guidance)
        if workflow_context:
            workflow_guidance = format_workflow_guidance_text(workflow_context, include_overview=True)
            guidance_text += f"\n\n{workflow_guidance}"

        image_dir_in_prompt, image_type = self.identify_image_path(prompt)
        
        base_intention_prompt = make_intention_analysis_prompt()
        complete_system_prompt = guidance_text + "\n\n" + base_intention_prompt
        
        # === STEP 2: INTENTION ANALYSIS ===
        # Always log intention analysis as it's a core workflow step
        self.logger.info(f"=== STEP 2: INTENTION ANALYSIS ===")
        self.logger.info(f"System Prompt Length: {len(complete_system_prompt)} characters")
        self.logger.info(f"System Prompt:\n{complete_system_prompt}")
        
        if image_dir_in_prompt:
            if image_type == "url":
                analysis_prompt = [
                                    (
                                        "system",
                                        complete_system_prompt
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
                                        complete_system_prompt
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
                    complete_system_prompt
                ),
                (
                    "human",
                    f"Analyze this image generation prompt: '{prompt}' with creativity level: {creativity_level.value}"
                )
            ]
        
        # Get response as string and parse it to dict
        response = track_llm_call(self.llm_json.invoke, "intention_analysis", analysis_prompt)
        
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
                
            # === STEP 2 OUTPUT ===
            ambiguous_count = len(parsed_response.get('ambiguous_elements', []))
            self.logger.info(f"=== STEP 2 OUTPUT: ANALYZED {ambiguous_count} AMBIGUOUS ELEMENTS ===")
            main_subjects = parsed_response.get('identified_elements', {}).get('main_subjects', [])
            self.logger.info(f"Main Subjects: {main_subjects}")
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in prompt analysis: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in prompt analysis: {str(e)}")
            raise

    def retrieve_reference(self, analysis: Dict[str, Any]):
        """Retrieve refenrece content or style based on the analysis."""

        # get current config
        current_config = config.get_current_config()
        if "references" in analysis["identified_elements"] and analysis["identified_elements"]["references"].get("content"):
            current_config["reference_content_image"] = analysis["identified_elements"]["references"]["content"]
        
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
                                    creativity_level: CreativityLevel = CreativityLevel.MEDIUM,
                                    workflow_context: Dict[str, Any] = None
                                ) -> Dict[str, Any]:
        """
        Refine the prompt using the analysis and any user responses.
        Also evaluates detail level when user responses are provided.
        
        Returns:
            Dict containing:
            - refined_prompt: str
            - suggested_creativity_level: CreativityLevel (only when user_responses provided)
        """
        
        # Prepare guidance text - use workflow guidance (includes processed RAG guidance)
        guidance_text = ""
        # Use workflow guidance as the primary guidance source (includes processed RAG guidance)
        if workflow_context:
            workflow_guidance = format_workflow_guidance_text(workflow_context, include_overview=False)
            guidance_text += f"\n\n{workflow_guidance}"
        
        if user_responses:
            base_system_prompt = """You are a Qwen-Image prompt expert. Your PRIMARY GOAL is to stay faithful to the original prompt while incorporating user clarifications. CRITICAL: The refined prompt must preserve the core intent, subjects, and atmosphere of the original prompt.
            
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
            {
                "refined_prompt": "A refined version that stays closely grounded to the original prompt while incorporating user clarifications. The refined prompt should read as a natural enhancement of the original, not a replacement.",
                "suggested_creativity_level": "LOW|MEDIUM|HIGH",
                "reasoning": "Explain why the suggested creativity level was chosen based on the detail completeness of user responses."
            }"""
            
            # Construct system prompt with guidance BEFORE the system prompt
            system_prompt = guidance_text + "\n\n" + base_system_prompt
            
            human_prompt = f"""Original prompt: "{original_prompt}"
            Analysis: {json.dumps(analysis, indent=2)}
            User responses: {json.dumps(user_responses, indent=2)}
            Current creativity level: {creativity_level.value}"""
        else:
            base_system_prompt = """You are a Qwen prompt expert. Your PRIMARY GOAL is to stay faithful to the original prompt while resolving ambiguities. CRITICAL: The refined prompt must preserve the core intent, subjects, and atmosphere of the original prompt.
            
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
            {
                "refined_prompt": "A refined version that stays closely grounded to the original prompt while resolving necessary ambiguities. The result should read as a natural clarification of the original, maintaining its core essence.",
                "reasoning": "Explain how the refinement preserves the original prompt's intent while addressing ambiguities."
            }"""
            
            # Construct system prompt with guidance BEFORE the system prompt
            system_prompt = guidance_text + "\n\n" + base_system_prompt
            
            human_prompt = f"""Original prompt: "{original_prompt}"
            Analysis: {json.dumps(analysis, indent=2)}
            Creativity level: {creativity_level.value}"""
        
        # === STEP 3: PROMPT REFINEMENT ===
        # Always log refinement as it's a core workflow step
        self.logger.info(f"=== STEP 3: PROMPT REFINEMENT ===")
        self.logger.info(f"System Prompt Length: {len(system_prompt)} characters")
        self.logger.info(f"System Prompt:\n{system_prompt}")
        self.logger.info(f"Human Prompt Length: {len(human_prompt)} characters")
        self.logger.info(f"Human Prompt:\n{human_prompt}")
        
        response = track_llm_call(self.llm_json.invoke, "refine_prompt", [
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        try:
            if isinstance(response.content, str):
                parsed_response = json.loads(response.content)
            elif isinstance(response.content, dict):
                parsed_response = response.content
            elif isinstance(response.content, json):
                parsed_response = response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
            refined_prompt = parsed_response.get('refined_prompt', 'No refined_prompt key')
            
            # === STEP 3 OUTPUT ===
            self.logger.info(f"=== STEP 3 OUTPUT: REFINED PROMPT ===")
            self.logger.info(f"REFINED: {refined_prompt}")
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in prompt refinement: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in prompt refinement: {str(e)}")
            raise


class NegativePromptGenerator:
    """Helper class for generating negative prompts"""
    def __init__(self, llm):
        self.llm = llm
        self.llm_json = llm.bind(response_format={"type": "json_object"})
        self.logger = logger

    def generate_negative_prompt(self, positive_prompt: str, analysis: Dict[str, Any] = None, workflow_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a negative prompt based on the positive prompt and analysis."""
        
        guidance_text = ""
        # Use workflow guidance as the primary guidance source (includes processed RAG guidance)
        if workflow_context:
            workflow_guidance = format_workflow_guidance_text(workflow_context, include_overview=False)
            guidance_text += f"\n\n{workflow_guidance}"
        
        base_negative_prompt_system = """You are an expert at generating negative prompts for image generation models like Qwen-Image and Qwen-Image-Edit.

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

        # Construct system prompt with guidance BEFORE the system prompt
        negative_prompt_system = guidance_text + "\n\n" + base_negative_prompt_system

        if analysis:
            analysis_text = f"\nPrompt analysis context: {json.dumps(analysis, indent=2)}"
        else:
            analysis_text = ""

        negative_prompt_user = f"""Generate a negative prompt for this image generation request:
Positive prompt: "{positive_prompt}"{analysis_text}

Consider the scene type, style, and content to determine what should be avoided."""

        # Log the complete negative prompt system prompt with guidance
        self.logger.info(f"=== NEGATIVE PROMPT - COMPLETE SYSTEM PROMPT (with guidance) ===")
        self.logger.info(f"System Prompt Length: {len(negative_prompt_system)} characters")
        self.logger.info(f"System Prompt:\n{negative_prompt_system}")
        self.logger.info(f"=== END NEGATIVE PROMPT SYSTEM PROMPT ===")

        try:
            response = track_llm_call(
                self.llm_json.invoke, 
                "negative_prompt_generation", 
                [
                    ("system", negative_prompt_system),
                    ("human", negative_prompt_user)
                ]
            )
            
            if isinstance(response.content, str):
                parsed_response = json.loads(response.content)
            elif isinstance(response.content, dict):
                parsed_response = response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response.content)}")
                
            negative_prompt = parsed_response.get('negative_prompt', 'No negative_prompt key')
            self.logger.info(f"Generated negative prompt - COMPLETE PROMPT:")
            self.logger.info(f"NEGATIVE: {negative_prompt}")
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Failed to generate negative prompt: {str(e)}")
            # Fallback to generic negative prompt
            return {
                "negative_prompt": "low quality, blurry, distorted, watermark, text, bad anatomy",
                "reasoning": "Fallback generic negative prompt due to generation error"
            }


class MemoryManager:
    """Manages global memory for model performance analysis using SQLite and RAG DB with FAISS."""
    
    def __init__(self, db_path: str = "model_memory.db", pattern_extraction_frequency: int = 10, 
                 min_entries_for_guidance: dict = None, logger=None):
        self.db_path = db_path
        self.pattern_extraction_frequency = pattern_extraction_frequency
        
        # Minimum entries required in each model's DB before guidance is used
        if min_entries_for_guidance is None:
            min_entries_for_guidance = {
                "qwen_image": 5,      # Minimum entries for Qwen-Image guidance
                "qwen_image_edit": 5  # Minimum entries for Qwen-Image-Edit guidance
            }
        self.min_entries_for_guidance = min_entries_for_guidance
        self.logger = logger
        
        # RAG DB components
        self.embedding_model = None
        self.rag_indices = {}  # Separate FAISS indices for each model
        self.rag_data = {}     # Store metadata for each model
        self.rag_db_paths = {
            "qwen_image": "rag_qwen_image.index",
            "qwen_image_edit": "rag_qwen_image_edit.index"
        }
        
        # Fallback storage paths for multi-GPU persistence
        self.fallback_db_paths = {
            "qwen_image": "fallback_qwen_image.pkl",
            "qwen_image_edit": "fallback_qwen_image_edit.pkl"
        }
        
        if self.logger is None:
            # Create a simple logger if none provided
            import logging
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        self._init_database()
        self._init_rag_system()
        self._init_fallback_storage()
    
    def _init_fallback_storage(self):
        """Initialize or load existing fallback storage from disk."""
        try:
            self.fallback_embeddings = {}
            
            # Load existing fallback data from disk for each model
            for model_key, fallback_path in self.fallback_db_paths.items():
                if os.path.exists(fallback_path):
                    try:
                        with open(fallback_path, 'rb') as f:
                            self.fallback_embeddings[model_key] = pickle.load(f)
                        self.logger.info(f"Loaded {len(self.fallback_embeddings[model_key])} fallback entries for {model_key}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load fallback storage for {model_key}: {str(e)}")
                        self.fallback_embeddings[model_key] = []
                else:
                    self.fallback_embeddings[model_key] = []
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback storage: {str(e)}")
            self.fallback_embeddings = {"qwen_image": [], "qwen_image_edit": []}
    
    def _save_fallback_storage(self, model_key: str):
        """Save fallback storage to disk for persistence across processes."""
        try:
            if model_key in self.fallback_embeddings:
                fallback_path = self.fallback_db_paths[model_key]
                
                # Use file locking to prevent conflicts between processes
                lock_path = f"{fallback_path}.lock"
                
                # Simple file-based locking mechanism
                max_wait_time = 10  # seconds
                wait_time = 0
                
                while os.path.exists(lock_path) and wait_time < max_wait_time:
                    time.sleep(0.1)
                    wait_time += 0.1
                
                try:
                    # Create lock file
                    with open(lock_path, 'w') as lock_file:
                        lock_file.write(str(os.getpid()))
                    
                    # Load existing data from disk to merge with current data
                    existing_data = []
                    if os.path.exists(fallback_path):
                        try:
                            with open(fallback_path, 'rb') as f:
                                existing_data = pickle.load(f)
                        except Exception:
                            existing_data = []
                    
                    # Merge data - avoid duplicates by checking timestamps and prompts
                    merged_data = existing_data.copy()
                    current_data = self.fallback_embeddings[model_key]
                    
                    for new_entry in current_data:
                        # Check if this entry already exists
                        is_duplicate = False
                        for existing_entry in existing_data:
                            if (existing_entry['prompt'] == new_entry['prompt'] and 
                                existing_entry['timestamp'] == new_entry['timestamp']):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            merged_data.append(new_entry)
                    
                    # Save merged data
                    with open(fallback_path, 'wb') as f:
                        pickle.dump(merged_data, f)
                    
                    # Update our in-memory copy
                    self.fallback_embeddings[model_key] = merged_data
                    
                    self.logger.debug(f"Saved {len(merged_data)} fallback entries for {model_key}")
                    
                finally:
                    # Remove lock file
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                        
        except Exception as e:
            self.logger.error(f"Failed to save fallback storage for {model_key}: {str(e)}")
    
    def _load_latest_fallback_storage(self, model_key: str):
        """Load the latest fallback storage from disk before searching."""
        try:
            fallback_path = self.fallback_db_paths[model_key]
            if os.path.exists(fallback_path):
                with open(fallback_path, 'rb') as f:
                    disk_data = pickle.load(f)
                
                # Only update if disk has more recent data
                current_count = len(self.fallback_embeddings.get(model_key, []))
                disk_count = len(disk_data)
                
                if disk_count > current_count:
                    self.fallback_embeddings[model_key] = disk_data
                    self.logger.debug(f"Loaded {disk_count} fallback entries for {model_key} from disk")
                    
        except Exception as e:
            self.logger.debug(f"Could not load latest fallback storage for {model_key}: {str(e)}")
    
    def _init_database(self):
        """Initialize SQLite database with tables for each model."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table for Qwen-Image memory
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qwen_image_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_index TEXT,
                    original_prompt TEXT,
                    refined_prompt TEXT,
                    evaluation_score REAL,
                    confidence_score REAL,
                    regeneration_count INTEGER,
                    trajectory_reasoning TEXT,
                    step_scores TEXT,
                    good_things TEXT,
                    bad_things TEXT,
                    overall_rating REAL,
                    config_data TEXT,
                    process_summary TEXT
                )
            ''')
            
            # Table for Qwen-Image-Edit memory
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qwen_image_edit_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_index TEXT,
                    original_prompt TEXT,
                    refined_prompt TEXT,
                    evaluation_score REAL,
                    confidence_score REAL,
                    regeneration_count INTEGER,
                    reference_image TEXT,
                    trajectory_reasoning TEXT,
                    step_scores TEXT,
                    good_things TEXT,
                    bad_things TEXT,
                    overall_rating REAL,
                    config_data TEXT,
                    process_summary TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.debug(f"Initialized memory database at: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory database: {str(e)}")
    
    def _init_rag_system(self):
        """Initialize RAG system with FAISS indices and Qwen embeddings."""
        global embedding_model
        try:
            # Use the global embedding model if available
            if embedding_model is not None:
                self.embedding_model = embedding_model
                self.logger.info("Using global Qwen embedding model")
            else:
                # Fallback: Initialize embedding model locally
                self.embedding_model = pipeline("feature-extraction", model="Qwen/Qwen3-Embedding-0.6B", device="cuda")
                self.logger.info("Initialized local Qwen embedding model (fallback)")
            
            # Initialize/load FAISS indices for each model
            for model_key, index_path in self.rag_db_paths.items():
                expected_dimension = 1024  # Qwen3-Embedding-0.6B embedding dimension
                
                if os.path.exists(index_path) and os.path.exists(f"{index_path}.metadata"):
                    try:
                        # Load existing index
                        existing_index = faiss.read_index(index_path)
                        
                        # Check if dimensions match
                        if existing_index.d == expected_dimension:
                            self.rag_indices[model_key] = existing_index
                            with open(f"{index_path}.metadata", 'rb') as f:
                                self.rag_data[model_key] = pickle.load(f)
                            self.logger.info(f"Loaded existing RAG index for {model_key} (dimension: {existing_index.d})")
                        else:
                            self.logger.warning(f"Dimension mismatch for {model_key}: existing={existing_index.d}, expected={expected_dimension}. Creating new index.")
                            # Create new index with correct dimension
                            self.rag_indices[model_key] = faiss.IndexFlatIP(expected_dimension)
                            self.rag_data[model_key] = []
                            self.logger.info(f"Created new RAG index for {model_key} (dimension: {expected_dimension})")
                    except Exception as e:
                        self.logger.warning(f"Failed to load existing index for {model_key}: {str(e)}. Creating new index.")
                        # Create new index
                        self.rag_indices[model_key] = faiss.IndexFlatIP(expected_dimension)
                        self.rag_data[model_key] = []
                        self.logger.info(f"Created new RAG index for {model_key} (dimension: {expected_dimension})")
                else:
                    # Create new index
                    self.rag_indices[model_key] = faiss.IndexFlatIP(expected_dimension)  # Inner product for cosine similarity
                    self.rag_data[model_key] = []
                    self.logger.info(f"Created new RAG index for {model_key} (dimension: {expected_dimension})")
            
            # Test embedding model to ensure it works
            if self.embedding_model is not None:
                try:
                    test_embedding = self.embedding_model("test")
                    # For Qwen embeddings pipeline, the output is different format
                    if isinstance(test_embedding, list) and len(test_embedding) > 0:
                        embedding_dim = len(test_embedding[0][0]) if isinstance(test_embedding[0], list) else len(test_embedding[0])
                        self.logger.info(f"Qwen embedding model test successful. Embedding dimension: {embedding_dim}")
                    else:
                        self.logger.info(f"Qwen embedding model test successful. Output: {type(test_embedding)}")
                except Exception as e:
                    self.logger.error(f"Embedding model test failed: {str(e)}")
                    self.embedding_model = None
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {str(e)}")
            # Fallback - create empty indices
            self.rag_indices = {"qwen_image": None, "qwen_image_edit": None}
            self.rag_data = {"qwen_image": [], "qwen_image_edit": []}
    
    def _save_rag_index(self, model_key: str):
        """Save FAISS index and metadata to disk."""
        try:
            if model_key in self.rag_indices and self.rag_indices[model_key] is not None:
                index_path = self.rag_db_paths[model_key]
                faiss.write_index(self.rag_indices[model_key], index_path)
                with open(f"{index_path}.metadata", 'wb') as f:
                    pickle.dump(self.rag_data[model_key], f)
                self.logger.debug(f"Saved RAG index for {model_key}")
        except Exception as e:
            self.logger.error(f"Failed to save RAG index for {model_key}: {str(e)}")
    
    def _update_rag_index(self, model_key: str, prompt: str, analysis_data: Dict[str, Any]):
        """Add new entry to RAG index."""
        try:
            if (self.embedding_model is None or 
                model_key not in self.rag_indices or 
                self.rag_indices[model_key] is None):
                self.logger.warning(f"Skipping RAG index update for {model_key}: embedding model or index not available")
                return
                
            self.logger.debug(f"Updating RAG index for {model_key} with prompt: {prompt[:50]}...")
                
            # Create embedding for the prompt
            try:
                # Use Qwen embedding pipeline - returns different format than previous .encode() method
                embedding_result = self.embedding_model(prompt)
                self.logger.debug(f"Raw embedding result type: {type(embedding_result)}")
                
                # Extract embedding from pipeline result format: [[[...embedding values...]]]
                if isinstance(embedding_result, list) and len(embedding_result) > 0:
                    if isinstance(embedding_result[0], list) and len(embedding_result[0]) > 0:
                        embedding = np.array(embedding_result[0][0])  # Get the actual embedding vector
                    else:
                        embedding = np.array(embedding_result[0])
                else:
                    self.logger.error(f"Unexpected embedding format: {type(embedding_result)}")
                    return
                
                self.logger.debug(f"Extracted embedding shape: {embedding.shape}")
                
                # Ensure it's a numpy array
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                self.logger.debug(f"After numpy conversion shape: {embedding.shape}")
                
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm  # Normalize for cosine similarity
                    self.logger.debug(f"Normalized embedding norm: {np.linalg.norm(embedding)}")
                else:
                    self.logger.warning(f"Zero-norm embedding for prompt: {prompt[:50]}...")
                    return  # Skip adding zero embeddings
                    
            except Exception as e:
                self.logger.error(f"Failed to create embedding for prompt '{prompt[:50]}...': {str(e)}")
                return
            
            # Ensure embedding is 2D for FAISS (reshape if needed)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
                self.logger.debug(f"Reshaped embedding for FAISS: {embedding.shape}")
            
            # Add to FAISS index - using fallback approach due to FAISS compatibility issues
            try:
                # Convert to float32 and ensure proper shape
                if embedding.dtype != np.float32:
                    embedding = embedding.astype(np.float32)
                
                # Ensure 2D shape for FAISS
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                
                # Check FAISS index compatibility first
                expected_dim = self.rag_indices[model_key].d
                if embedding.shape[1] != expected_dim:
                    raise ValueError(f"Embedding dimension {embedding.shape[1]} doesn't match FAISS index dimension {expected_dim}")
                
                self.logger.debug(f"All verification checks passed. Adding to FAISS index with dimension {expected_dim}")
                
                # Try to use FAISS, but fall back to simple storage if it fails
                faiss_success = False
                
                # Try the most basic FAISS approach with extensive debugging
                try:
                    # Create the most basic numpy array possible
                    simple_embedding = embedding.flatten().astype(np.float32)
                    simple_embedding = simple_embedding.reshape(1, -1)
                    simple_embedding = np.ascontiguousarray(simple_embedding)
                    
                    # Additional debugging
                    self.logger.debug(f"Simple embedding details:")
                    self.logger.debug(f"  Type: {type(simple_embedding)}")
                    self.logger.debug(f"  Shape: {simple_embedding.shape}")
                    self.logger.debug(f"  Dtype: {simple_embedding.dtype}")
                    self.logger.debug(f"  Contiguous: {simple_embedding.flags['C_CONTIGUOUS']}")
                    self.logger.debug(f"  Data pointer: {simple_embedding.data}")
                    
                    # Verify the FAISS index is valid
                    self.logger.debug(f"FAISS index ntotal before add: {self.rag_indices[model_key].ntotal}")
                    
                    # Try the actual FAISS add operation
                    self.rag_indices[model_key].add(simple_embedding)
                    
                    # Check if it actually worked
                    new_total = self.rag_indices[model_key].ntotal
                    self.logger.debug(f"FAISS index ntotal after add: {new_total}")
                    
                    if new_total > 0:
                        faiss_success = True
                        self.logger.debug(f"Successfully added embedding to FAISS index")
                    else:
                        self.logger.warning("FAISS add appeared to succeed but ntotal is still 0")
                        
                except Exception as faiss_error:
                    self.logger.warning(f"FAISS add failed: {str(faiss_error)}. Using fallback storage method.")
                    faiss_success = False
                
                # If FAISS failed, use a simple list-based storage as backup
                if not faiss_success:
                    self.logger.info("Using fallback embedding storage method instead of FAISS")
                    
                    # Ensure fallback storage exists
                    if not hasattr(self, 'fallback_embeddings'):
                        self.fallback_embeddings = {}
                    if model_key not in self.fallback_embeddings:
                        self.fallback_embeddings[model_key] = []
                    
                    # Store the embedding with metadata for manual similarity search
                    fallback_entry = {
                        'embedding': embedding.flatten().tolist(),  # Store as list for JSON serialization
                        'prompt': prompt,
                        'analysis': analysis_data,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.fallback_embeddings[model_key].append(fallback_entry)
                    self.logger.debug(f"Stored embedding in fallback storage. Total fallback entries: {len(self.fallback_embeddings[model_key])}")
                    
                    # Save to disk for persistence across processes
                    self._save_fallback_storage(model_key)
                    
            except Exception as e:
                self.logger.error(f"Failed to add embedding (both FAISS and fallback): {str(e)}")
                self.logger.debug(f"Original embedding details - type: {type(embedding)}, shape: {embedding.shape}, dtype: {embedding.dtype}")
                self.logger.debug(f"FAISS index details - type: {type(self.rag_indices[model_key])}, dimension: {self.rag_indices[model_key].d}")
                return
            
            # Store metadata
            try:
                self.rag_data[model_key].append({
                    'prompt': prompt,
                    'analysis': analysis_data,
                    'timestamp': datetime.now().isoformat()
                })
                self.logger.debug(f"Successfully stored metadata. Total entries: {len(self.rag_data[model_key])}")
            except Exception as e:
                self.logger.error(f"Failed to store metadata: {str(e)}")
                return
            
            # Save to disk periodically
            if len(self.rag_data[model_key]) % 10 == 0:
                self._save_rag_index(model_key)
                
        except Exception as e:
            self.logger.error(f"Failed to update RAG index for {model_key}: {str(e)}")
    
    def _create_zeros_copy(self, embedding, expected_dim):
        """Create a fresh array by copying to a zeros array."""
        zeros_array = np.zeros((1, expected_dim), dtype=np.float32)
        zeros_array[0, :] = embedding.flatten()[:expected_dim]
        return np.ascontiguousarray(zeros_array)
    
    def _create_manual_array(self, embedding, expected_dim):
        """Manually create array from list conversion."""
        # Convert to list and back to create the cleanest possible array
        flat_list = embedding.flatten().tolist()[:expected_dim]
        manual_array = np.array([flat_list], dtype=np.float32)
        return np.ascontiguousarray(manual_array)
    
    def search_similar_prompts(self, prompt: str, model_type: str = "qwen_image", top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar prompts in RAG database and return guidance."""
        try:
            model_key = model_type  # Already in correct format from main workflow
            
            # Check both FAISS storage and fallback storage
            has_faiss_data = (self.embedding_model is not None and 
                            model_key in self.rag_indices and 
                            self.rag_indices[model_key] is not None and
                            len(self.rag_data[model_key]) > 0)
            
            # Load latest fallback data from disk (for multi-GPU processes)
            self._load_latest_fallback_storage(model_key)
            
            has_fallback_data = (hasattr(self, 'fallback_embeddings') and 
                                model_key in self.fallback_embeddings and 
                                len(self.fallback_embeddings[model_key]) > 0)
            
            if not has_faiss_data and not has_fallback_data:
                self.logger.debug(f"No data available for similarity search in {model_key}")
                return []
            
            # Create embedding for search prompt
            try:
                # Use Qwen embedding pipeline
                query_result = self.embedding_model(prompt)
                
                # Extract embedding from pipeline result format
                if isinstance(query_result, list) and len(query_result) > 0:
                    if isinstance(query_result[0], list) and len(query_result[0]) > 0:
                        query_embedding = np.array(query_result[0][0])
                    else:
                        query_embedding = np.array(query_result[0])
                else:
                    self.logger.error(f"Unexpected query embedding format: {type(query_result)}")
                    return []
                
                # Ensure it's a numpy array and normalize
                if not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding)
                
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
            except Exception as e:
                self.logger.error(f"Failed to create query embedding for prompt '{prompt[:50]}...': {str(e)}")
                return []
            
            similar_entries = []
            
            # Try FAISS search first if available
            if has_faiss_data:
                try:
                    # Ensure query embedding is 2D for FAISS
                    if query_embedding.ndim == 1:
                        query_embedding_2d = query_embedding.reshape(1, -1)
                    else:
                        query_embedding_2d = query_embedding
                    
                    # Search similar entries using FAISS
                    k = min(top_k, len(self.rag_data[model_key]))
                    if k > 0:
                        # Create FAISS-compatible query embedding
                        query_clean = np.ascontiguousarray(query_embedding_2d.astype(np.float32))
                        
                        scores, indices = self.rag_indices[model_key].search(query_clean, k)
                        
                        # Return similar entries with scores
                        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                            if idx != -1 and idx < len(self.rag_data[model_key]):
                                entry = self.rag_data[model_key][idx].copy()
                                entry['similarity_score'] = float(score)
                                similar_entries.append(entry)
                        
                        self.logger.debug(f"Found {len(similar_entries)} similar entries using FAISS")
                        
                except Exception as faiss_error:
                    self.logger.warning(f"FAISS search failed: {str(faiss_error)}. Falling back to manual search.")
                    similar_entries = []
            
            # Use fallback search if FAISS failed or not available
            if not similar_entries and has_fallback_data:
                try:
                    self.logger.debug("Using fallback manual similarity search")
                    fallback_data = self.fallback_embeddings[model_key]
                    
                    # Calculate similarities manually
                    similarities = []
                    for i, entry in enumerate(fallback_data):
                        stored_embedding = entry['embedding']
                        if isinstance(stored_embedding, list):
                            stored_embedding = np.array(stored_embedding)
                        
                        # Normalize stored embedding
                        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
                        
                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding.flatten(), stored_embedding.flatten())
                        similarities.append((similarity, i))
                    
                    # Sort by similarity and get top_k
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    top_similarities = similarities[:min(top_k, len(similarities))]
                    
                    # Build result entries
                    for similarity, idx in top_similarities:
                        entry = {
                            'prompt': fallback_data[idx]['prompt'],
                            'analysis': fallback_data[idx]['analysis'],
                            'timestamp': fallback_data[idx]['timestamp'],
                            'similarity_score': float(similarity)
                        }
                        similar_entries.append(entry)
                    
                    self.logger.debug(f"Found {len(similar_entries)} similar entries using fallback search")
                    
                except Exception as fallback_error:
                    self.logger.error(f"Fallback search also failed: {str(fallback_error)}")
                    return []
            
            return similar_entries
            
        except Exception as e:
            self.logger.error(f"Failed to search similar prompts: {str(e)}")
            return []
    
    def get_workflow_guidance(self, prompt: str, model_type: str = "qwen_image") -> Dict[str, Dict[str, str]]:
        """Get workflow guidance based on similar prompts from RAG DB."""
        try:
            # Check if we have enough entries in the specific model's database
            min_required = self.min_entries_for_guidance.get(model_type, 5)
            
            # Count from both FAISS storage and fallback storage
            faiss_count = len(self.rag_data.get(model_type, []))
            fallback_count = 0
            if hasattr(self, 'fallback_embeddings') and model_type in self.fallback_embeddings:
                fallback_count = len(self.fallback_embeddings[model_type])
            
            # Use the maximum of both counts (they should be the same, but just in case)
            current_entries = max(faiss_count, fallback_count)
            
            if current_entries < min_required:
                self.logger.info(f"Not enough entries in {model_type} database ({current_entries}/{min_required}). Skipping guidance.")
                # Return appropriate empty structure based on model type
                if model_type == "qwen_image_edit":
                    return {
                        "positive_guidance": {
                            "negative_model_selection": "",
                            "prompt_polishing": "",
                            "image_editing": "",
                            "quality_evaluation": ""
                        },
                        "unsuccessful_patterns": {
                            "negative_model_selection": "",
                            "prompt_polishing": "",
                            "image_editing": "",
                            "quality_evaluation": ""
                        }
                    }
                else:
                    return {
                        "positive_guidance": {
                            "creativity_level_determination": "",
                            "intention_analysis": "",
                            "prompt_refinement": "",
                            "negative_model_selection": "",
                            "prompt_polishing": "",
                            "image_generation": "",
                            "quality_evaluation": "",
                            "regeneration_decision": ""
                        },
                        "unsuccessful_patterns": {
                            "creativity_level_determination": "",
                            "intention_analysis": "",
                            "prompt_refinement": "",
                            "negative_model_selection": "",
                            "prompt_polishing": "",
                            "image_generation": "",
                            "quality_evaluation": "",
                            "regeneration_decision": ""
                        }
                    }
            
            similar_prompts = self.search_similar_prompts(prompt, model_type, top_k=5)
            
            if not similar_prompts:
                self.logger.info(f"No similar prompts found for: {prompt[:50]}...")
                # Return appropriate empty structure based on model type
                if model_type == "qwen_image_edit":
                    return {
                        "positive_guidance": {
                            "negative_model_selection": "",
                            "prompt_polishing": "",
                            "image_editing": "",
                            "quality_evaluation": ""
                        },
                        "unsuccessful_patterns": {
                            "negative_model_selection": "",
                            "prompt_polishing": "",
                            "image_editing": "",
                            "quality_evaluation": ""
                        }
                    }
                else:
                    return {
                        "positive_guidance": {
                            "creativity_level_determination": "",
                            "intention_analysis": "",
                            "prompt_refinement": "",
                            "negative_model_selection": "",
                            "prompt_polishing": "",
                            "image_generation": "",
                            "quality_evaluation": "",
                            "regeneration_decision": ""
                        },
                        "unsuccessful_patterns": {
                            "creativity_level_determination": "",
                            "intention_analysis": "",
                            "prompt_refinement": "",
                            "negative_model_selection": "",
                            "prompt_polishing": "",
                            "image_generation": "",
                            "quality_evaluation": "",
                            "regeneration_decision": ""
                        }
                    }
            
            # Log extracted similar prompts for transparency
            print("=" * 80)
            print(f"MEMORY MANAGER - EXTRACTED SIMILAR PROMPTS ({model_type.upper()})")
            print("=" * 80)
            print(f"Query Prompt: {prompt}")
            print(f"Found {len(similar_prompts)} similar prompts:")
            print("-" * 80)
            
            for i, entry in enumerate(similar_prompts, 1):
                sim_prompt = entry.get('prompt', 'N/A')
                score = entry.get('similarity_score', 0.0)
                # Show complete similar prompts for better understanding
                print(f"#{i} (Score: {score:.3f}):")
                print(f"   PROMPT: {sim_prompt}")
                print()
                
            print("-" * 80)
            print(f"→ Generating guidance from {len(similar_prompts)} similar examples...")
            print("=" * 80)
            self.logger.info(f"Found {len(similar_prompts)} similar prompts for guidance generation")
            
            # Extract insights from similar prompts
            guidance = self._analyze_similar_prompts_for_guidance(similar_prompts)
            
            return guidance
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow guidance: {str(e)}")
            return {
                "positive_guidance": {
                    "creativity_level": "",
                    "intention_analysis": "",
                    "prompt_refinement": "",
                    "negative_prompt": "",
                    "prompt_polishing": "",
                    "generation": "",
                    "evaluation": ""
                },
                "unsuccessful_patterns": {
                    "creativity_level": "",
                    "intention_analysis": "",
                    "prompt_refinement": "",
                    "negative_prompt": "",
                    "prompt_polishing": "",
                    "generation": ""
                }
            }
    
    def get_entry_counts(self) -> Dict[str, int]:
        """Get the current number of entries in each model's database."""
        counts = {}
        for model_type in ["qwen_image", "qwen_image_edit"]:
            # Load latest fallback data from disk (for multi-GPU processes)
            self._load_latest_fallback_storage(model_type)
            
            # Count from both FAISS storage and fallback storage
            faiss_count = len(self.rag_data.get(model_type, []))
            fallback_count = 0
            if hasattr(self, 'fallback_embeddings') and model_type in self.fallback_embeddings:
                fallback_count = len(self.fallback_embeddings[model_type])
            
            # Use the maximum of both counts (they should be the same, but just in case)
            counts[model_type] = max(faiss_count, fallback_count)
            
        return counts
    
    def is_guidance_available(self, model_type: str = "qwen_image") -> bool:
        """Check if guidance is available for the specified model type."""
        min_required = self.min_entries_for_guidance.get(model_type, 5)
        
        # Count from both FAISS storage and fallback storage
        faiss_count = len(self.rag_data.get(model_type, []))
        fallback_count = 0
        if hasattr(self, 'fallback_embeddings') and model_type in self.fallback_embeddings:
            fallback_count = len(self.fallback_embeddings[model_type])
        
        # Use the maximum of both counts (they should be the same, but just in case)
        current_entries = max(faiss_count, fallback_count)
        return current_entries >= min_required
    
    def log_guidance_status(self):
        """Log the current guidance availability status for all model types."""
        entry_counts = self.get_entry_counts()
        self.logger.info("=== GUIDANCE AVAILABILITY STATUS ===")
        for model_type in ["qwen_image", "qwen_image_edit"]:
            min_required = self.min_entries_for_guidance.get(model_type, 5)
            current_entries = entry_counts.get(model_type, 0)
            available = current_entries >= min_required
            status = "AVAILABLE" if available else "NOT AVAILABLE"
            self.logger.info(f"{model_type}: {current_entries}/{min_required} entries - {status}")
    
    def _analyze_similar_prompts_for_guidance(self, similar_prompts: List[Dict[str, Any]], model_type: str = "qwen_image") -> Dict[str, Dict[str, str]]:
        """Analyze similar prompts to extract workflow guidance using LLM."""
        global llm_json
        
        try:
            # Prepare data from similar prompts
            analysis_data = []
            for entry in similar_prompts:
                prompt = entry.get('prompt', '')
                analysis = entry.get('analysis', {})
                similarity = entry.get('similarity_score', 0)
                
                # Parse analysis if it's a JSON string
                if isinstance(analysis, str):
                    try:
                        analysis = json.loads(analysis)
                    except:
                        continue
                
                if isinstance(analysis, dict):
                    trajectory_reasoning = analysis.get('trajectory_reasoning', '')
                    step_scores = analysis.get('step_scores', {})
                    good_things = analysis.get('good_things', {})
                    bad_things = analysis.get('bad_things', {})
                    overall_rating = analysis.get('overall_rating', 0)
                    
                    analysis_data.append({
                        'prompt': prompt,
                        'similarity': similarity,
                        'trajectory_reasoning': trajectory_reasoning,
                        'step_scores': step_scores,
                        'good_things': good_things,
                        'bad_things': bad_things,
                        'overall_rating': overall_rating
                    })
            
            if not analysis_data:
                return {"positive_guidance": {}, "unsuccessful_patterns": {}}
            
            # Create model-specific prompt for LLM analysis with improved structure for better pattern recognition
            similar_data_text = ""
            
            # Group entries by content similarity
            # First, sort by similarity score (highest first)
            sorted_entries = sorted(analysis_data, key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Determine if this is for edit or regular generation
            is_edit_model = model_type == "qwen_image_edit"
            
            # Create edit-specific or generation-specific analysis header
            if is_edit_model:
                similar_data_text += "=== IMAGE EDITING WORKFLOW ANALYSIS ===\n"
                similar_data_text += "Focus: Image modification, object manipulation, and seamless blending\n\n"
            else:
                similar_data_text += "=== IMAGE GENERATION WORKFLOW ANALYSIS ===\n"
                similar_data_text += "Focus: New image creation from text descriptions\n\n"
            
            # Format the data with clear sections
            for i, entry in enumerate(sorted_entries):
                similarity = entry.get('similarity', 0)
                prompt = entry.get('prompt', '')
                good_things = entry.get('good_things', {})
                bad_things = entry.get('bad_things', {})
                
                # Add entry header with similarity score
                similar_data_text += f"\n===== EXAMPLE {i+1} (SIMILARITY: {similarity:.3f}) =====\n"
                similar_data_text += f"PROMPT: '{prompt}'\n"
                
                # Add trajectory reasoning section
                trajectory_reasoning = entry.get('trajectory_reasoning', '')
                if trajectory_reasoning:
                    similar_data_text += f"TRAJECTORY REASONING: {trajectory_reasoning}\n"
                
                # Add step scores section
                step_scores = entry.get('step_scores', {})
                overall_rating = entry.get('overall_rating', 0)
                
                similar_data_text += "STEP SCORES (1-10):\n"
                if isinstance(step_scores, dict):
                    for step, score in step_scores.items():
                        similar_data_text += f"- [{step}]: {score}\n"
                else:
                    similar_data_text += f"- Step scores data format issue\n"
                
                similar_data_text += f"- [overall_rating]: {overall_rating}/10\n"
                
                # Add successful strategies section
                similar_data_text += "SUCCESSFUL STRATEGIES:\n"
                for step, success in good_things.items():
                    similar_data_text += f"- [{step}]: {success}\n"
                
                # Add problems/issues section
                similar_data_text += "PROBLEMS/ISSUES:\n"
                for step, issue in bad_things.items():
                    similar_data_text += f"- [{step}]: {issue}\n"
                
                similar_data_text += "="*50 + "\n"
                
            # Add a model-specific prompt to look for domain-specific patterns
            if is_edit_model:
                similar_data_text += "\nIDENTIFY EDIT-SPECIFIC PATTERNS: What specific image editing techniques, reference image utilization strategies, blending approaches, and object manipulation methods worked or failed for this particular type of edit operation?"
            else:
                similar_data_text += "\nIDENTIFY GENERATION-SPECIFIC PATTERNS: What specific creation techniques, style choices, and compositional approaches worked or failed for this particular content domain or subject matter?"
            
            # Create model-specific guidance prompt
            if is_edit_model:
                workflow_description = """
EDIT WORKFLOW STRUCTURE:
Step 1: NEGATIVE_MODEL_SELECTION → Step 2: PROMPT_POLISHING → Step 3: IMAGE_EDITING → Step 4: EVALUATION

Note: Edit workflows skip creativity level, intention analysis, and prompt refinement steps since they start with an existing reference image and edit instruction.

EDIT-SPECIFIC FOCUS AREAS:
- Reference image analysis and utilization
- Edit instruction clarity and precision
- Blending and seamless integration techniques
- Object removal/addition strategies
- Texture matching and consistency
- Edit boundary handling
- Natural transition creation"""

                task_focus = """EDIT-SPECIFIC INSTRUCTIONS:
1. Analyze SPECIFIC image editing techniques, blending methods, and reference image utilization that led to success or failure
2. Extract CONCRETE advice for edit operations: object manipulation, texture blending, seamless transitions
3. Focus on edit precision, reference image analysis, and modification strategies
4. Identify EDIT-CONTEXTUAL patterns: what works for object removal vs addition, texture editing vs composition changes
5. AVOID generic editing advice like "blend better" or "match textures"
6. Instead, provide detailed examples like "For object removal in backgrounds, preserve lighting direction and shadow consistency"""
            else:
                workflow_description = """
GENERATION WORKFLOW STRUCTURE:
Step 1: CREATIVITY_LEVEL → Step 2: INTENTION_ANALYSIS → Step 3: PROMPT_REFINEMENT → 
Step 4: NEGATIVE_MODEL_SELECTION → Step 5: PROMPT_POLISHING → Step 6: IMAGE_GENERATION → Step 7: EVALUATION

GENERATION-SPECIFIC FOCUS AREAS:
- Prompt interpretation and creative enhancement
- Style and composition guidance
- Scene construction and element placement
- Atmospheric and lighting control
- Detail specification and enhancement
- Creative interpretation of abstract concepts"""

                task_focus = """GENERATION-SPECIFIC INSTRUCTIONS:
1. Analyze SPECIFIC creation techniques, parameters, and approaches that led to success or failure in similar prompts
2. Extract CONCRETE advice based on actual generation outcomes, not general best practices
3. Focus on the particular subject matter, style choices, and technical details that matter for this type of prompt
4. Identify CONTEXTUAL patterns relevant to the specific image generation domain
5. AVOID generic advice like "be more specific" or "add more details"
6. Instead, provide detailed examples like "For nature scenes, explicitly specify time of day and weather conditions"""

            # Create model-specific guidance prompt with different step structures
            if is_edit_model:
                step_analysis_structure = """
        "negative_model_selection": {{
            "success_patterns": "SPECIFIC edit model selection and negative prompt strategies that were effective for this type of edit",
            "failure_patterns": "SPECIFIC edit model selection and negative prompt issues observed in similar cases",
            "impact_on_next": "CONCRETE impact on prompt polishing for editing",
            "preventive_guidance": "DETAILED advice with SPECIFIC negative prompt terms and edit model selection criteria",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "prompt_polishing": {{
            "success_patterns": "SPECIFIC edit prompt polishing techniques that improved similar edit operations",
            "failure_patterns": "SPECIFIC edit prompt polishing issues observed with this type of edit content",
            "impact_on_next": "CONCRETE impact on edit execution quality",
            "preventive_guidance": "DETAILED advice with SPECIFIC edit prompt optimization approaches",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "image_editing": {{
            "success_patterns": "SPECIFIC editing parameters and techniques that worked for similar content and edit types",
            "failure_patterns": "SPECIFIC editing issues observed with this edit operation type",
            "impact_on_next": "CONCRETE impact on evaluation accuracy",
            "preventive_guidance": "DETAILED advice with SPECIFIC editing settings, reference image usage, and blending techniques",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "quality_evaluation": {{
            "success_patterns": "SPECIFIC evaluation criteria effective for this edit content type",
            "failure_patterns": "SPECIFIC evaluation pitfalls observed with similar edit outputs",
            "impact_on_next": "CONCRETE impact on re-edit decision",
            "preventive_guidance": "DETAILED advice with SPECIFIC edit quality indicators and success metrics",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }}"""
            else:
                step_analysis_structure = """
        "creativity_level_determination": {{
            "success_patterns": "SPECIFIC creativity level patterns that worked for this type of prompt",
            "failure_patterns": "SPECIFIC creativity level mistakes observed with similar prompts",
            "impact_on_next": "CONCRETE impact on intention analysis quality",
            "preventive_guidance": "DETAILED advice with SPECIFIC examples of what to look for",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "intention_analysis": {{
            "success_patterns": "SPECIFIC intention analysis techniques that worked for this subject matter",
            "failure_patterns": "SPECIFIC intention analysis pitfalls observed with similar prompts", 
            "impact_on_next": "CONCRETE impact on prompt refinement",
            "preventive_guidance": "DETAILED advice with SPECIFIC elements to identify",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "prompt_refinement": {{
            "success_patterns": "SPECIFIC refinement strategies that enhanced similar prompts",
            "failure_patterns": "SPECIFIC refinement mistakes observed in similar cases",
            "impact_on_next": "CONCRETE impact on negative prompt generation",
            "preventive_guidance": "DETAILED advice with SPECIFIC refinement techniques",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "negative_model_selection": {{
            "success_patterns": "SPECIFIC negative prompt and model selection strategies that were effective for this subject",
            "failure_patterns": "SPECIFIC negative prompt and model selection issues observed in similar cases",
            "impact_on_next": "CONCRETE impact on prompt polishing",
            "preventive_guidance": "DETAILED advice with SPECIFIC negative prompt terms and model selection criteria",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "prompt_polishing": {{
            "success_patterns": "SPECIFIC polishing techniques that improved similar prompts",
            "failure_patterns": "SPECIFIC polishing issues observed with this type of content",
            "impact_on_next": "CONCRETE impact on generation quality",
            "preventive_guidance": "DETAILED advice with SPECIFIC polishing approaches",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "image_generation": {{
            "success_patterns": "SPECIFIC generation parameters that worked for similar content",
            "failure_patterns": "SPECIFIC generation issues observed with this prompt type",
            "impact_on_next": "CONCRETE impact on evaluation accuracy",
            "preventive_guidance": "DETAILED advice with SPECIFIC generation settings",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "quality_evaluation": {{
            "success_patterns": "SPECIFIC evaluation criteria effective for this content type",
            "failure_patterns": "SPECIFIC evaluation pitfalls observed with similar outputs",
            "impact_on_next": "CONCRETE impact on regeneration decision",
            "preventive_guidance": "DETAILED advice with SPECIFIC quality indicators",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }},
        "regeneration_decision": {{
            "success_patterns": "SPECIFIC decision criteria that led to successful outcomes",
            "failure_patterns": "SPECIFIC regeneration mistakes observed with similar content",
            "impact_on_next": "N/A - final step",
            "preventive_guidance": "DETAILED advice with SPECIFIC decision factors",
            "recommended_score": "Recommend target score (1-10) based on similar examples"
        }}"""

            guidance_prompt = f"""Based on the analysis of similar prompts, extract CONCRETE and SPECIFIC workflow guidance for the new prompt.

SIMILAR PROMPTS DATA:
{similar_data_text}

{workflow_description}

{task_focus}

Return JSON with this EXACT structure:
{{
    "step_analysis": {{
{step_analysis_structure}
    }},
    "workflow_insights": {{
        "critical_dependencies": "SPECIFIC step dependencies most relevant to this prompt type",
        "common_failure_chains": "CONCRETE failure patterns observed in similar cases",
        "success_combinations": "SPECIFIC combinations of choices that worked well for this content",
        "overall_rating_prediction": "Predict the likely overall success rating (1-10) for this prompt type"
    }}
}}

Your guidance must be SPECIFIC to the prompt type and content domain, not generic best practices.
For example, instead of 'Use detailed prompts', say 'For architectural images, specify architectural style, materials, lighting conditions, and surrounding environment'.
Use concrete, actionable advice derived from the data, not theoretical recommendations.

IMPORTANT: Ensure the guidance provides DOMAIN-SPECIFIC advice for the content type in the prompts, not just generic image generation tips."""

            response = track_llm_call(llm_json.invoke, "structured_workflow_guidance", [
                ("system", """You are an expert at analyzing patterns in image generation workflows to provide SPECIFIC, CONCRETE, and ACTIONABLE guidance.

Your specialty is identifying context-specific techniques that work for particular types of images, not general best practices.

Focus on content-specific insights like:
- For portrait images: specific lighting techniques, expression guidance, compositional elements
- For landscape scenes: time of day effects, weather condition impacts, foreground element placement
- For abstract concepts: style reference importance, compositional balance techniques, color palette guidance

Avoid general advice like "add more detail" or "be more specific".
Instead provide domain-specific, technical recommendations based on the actual examples you analyze."""),
                ("human", guidance_prompt)
            ])
            
            if isinstance(response.content, str):
                result = json.loads(response.content)
            else:
                result = response.content
                
            self.logger.info(f"Successfully extracted structured workflow guidance with {len(result.get('step_analysis', {}))} step analyses")
            
            # Convert the detailed step_analysis to the expected positive_guidance/unsuccessful_patterns format
            step_analysis = result.get('step_analysis', {})
            positive_guidance = {}
            unsuccessful_patterns = {}
            
            for step_name, step_data in step_analysis.items():
                if isinstance(step_data, dict):
                    success_patterns = step_data.get('success_patterns', '')
                    preventive_guidance = step_data.get('preventive_guidance', '')
                    
                    # Combine success patterns and preventive guidance for positive guidance
                    positive_text = f"{success_patterns} {preventive_guidance}".strip()
                    positive_guidance[step_name] = positive_text
                    
                    # Use failure patterns for unsuccessful patterns
                    failure_patterns = step_data.get('failure_patterns', '')
                    unsuccessful_patterns[step_name] = failure_patterns
            
            return {
                "positive_guidance": positive_guidance,
                "unsuccessful_patterns": unsuccessful_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze similar prompts for guidance: {str(e)}")
            # Return appropriate fallback structure based on model type
            if model_type == "qwen_image_edit":
                return {
                    "positive_guidance": {
                        "negative_model_selection": "",
                        "prompt_polishing": "",
                        "image_editing": "",
                        "quality_evaluation": ""
                    },
                    "unsuccessful_patterns": {
                        "negative_model_selection": "",
                        "prompt_polishing": "",
                        "image_editing": "",
                        "quality_evaluation": ""
                    }
                }
            else:
                return {
                    "positive_guidance": {
                        "creativity_level_determination": "",
                        "intention_analysis": "",
                        "prompt_refinement": "",
                        "negative_model_selection": "",
                        "prompt_polishing": "",
                        "image_generation": "",
                        "quality_evaluation": "",
                        "regeneration_decision": ""
                    },
                    "unsuccessful_patterns": {
                        "creativity_level_determination": "",
                        "intention_analysis": "",
                        "prompt_refinement": "",
                        "negative_model_selection": "",
                        "prompt_polishing": "",
                        "image_generation": "",
                        "quality_evaluation": "",
                        "regeneration_decision": ""
                    }
                }
    
    def _generate_process_summary(self, config_data: Dict[str, Any]) -> str:
        """Generate an explanation of how each step in the image generation workflow works."""
        try:
            # Extract key information for workflow explanation
            prompt_understanding = config_data.get("prompt_understanding", {})
            original_prompt = prompt_understanding.get('original_prompt', 'N/A')
            creativity_level = prompt_understanding.get('creativity_level', 'medium')
            regen_count = config_data.get("regeneration_count", 0)
            
            # Parse prompt analysis if available
            prompt_analysis = prompt_understanding.get('prompt_analysis', {})
            if isinstance(prompt_analysis, str):
                try:
                    prompt_analysis = json.loads(prompt_analysis)
                except:
                    prompt_analysis = {}
            
            # Build workflow step explanations in execution order
            summary = f"""IMAGE GENERATION WORKFLOW EXECUTION EXPLANATION:

1. CREATIVITY LEVEL DETERMINATION:
The system analyzed the user prompt "{original_prompt}" and determined a {creativity_level} creativity level. This controls how the system handles ambiguous or missing details - {creativity_level} level means the system {'asks for clarification on most unclear elements before proceeding' if creativity_level == 'low' else 'balances autonomous interpretation with strategic user clarification for important details' if creativity_level == 'medium' else 'autonomously fills in most missing details using contextual reasoning'}.

2. INTENTION ANALYSIS:
The system performed deep semantic analysis to understand the user's intent behind the prompt. It identified {len(prompt_analysis.get('identified_elements', {}))} specific visual elements (objects, styles, composition details) and detected {len(prompt_analysis.get('ambiguous_elements', []))} ambiguous aspects that needed interpretation. This analysis determines what questions to ask and how to structure the image generation approach.

3. PROMPT REFINEMENT:
Based on the intention analysis, the system {'refined the original prompt by clarifying ambiguous elements and adding contextual details' if prompt_understanding.get('refined_prompt') != original_prompt else 'determined the original prompt was sufficiently clear and required minimal refinement'}. The refined prompt: "{prompt_understanding.get('refined_prompt', 'N/A')}" serves as the foundation for all subsequent generation steps.

4. NEGATIVE PROMPT CREATION:
For each generation attempt, the system analyzes potential unwanted artifacts and creates targeted negative prompts to avoid common issues like blurriness, distortion, low quality, or inappropriate content. This step identifies what NOT to generate based on the prompt requirements and model characteristics.

5. PROMPT POLISHING:
The system then optimizes the generating prompt specifically for the target model's architecture and training patterns. This involves adjusting language, adding technical keywords, and structuring the prompt to maximize the model's understanding and output quality.

6. GENERATION:
The system selects the most appropriate model based on prompt requirements and generates the image using the polished prompts and negative constraints. Model selection considers factors like artistic style needs, technical requirements, and reference image usage.

7. EVALUATION:
Each generated image undergoes automated evaluation analyzing prompt adherence, visual quality, technical execution, and artistic merit. The evaluation produces quantitative scores and qualitative feedback to guide potential improvements.

8. REGENERATION PROCESS:
{'The workflow completed in a single generation cycle with satisfactory results.' if regen_count == 0 else f'The system performed {regen_count} additional generation cycles, each time analyzing the previous results and adjusting the approach based on evaluation feedback and user input. Each regeneration cycle refines model selection, prompt optimization, and generation parameters.'}

This workflow ensures systematic optimization from initial prompt understanding through final image delivery."""

            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate process summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _analyze_model_performance(self, config_data: Dict[str, Any], process_summary: str, model_name: str, regen_config: Dict[str, Any]) -> Dict[str, str]:
        """Use LLM to analyze model performance and extract good/bad things, including visual analysis of generated images."""
        global llm_json
        
        # Extract relevant config information in structured format
        prompt_understanding = config_data.get("prompt_understanding", {})
        
        # Create structured config context (not full JSON dump)
        config_context = {
            "prompt_details": {
                "original": prompt_understanding.get('original_prompt', 'N/A'),
                "refined": prompt_understanding.get('refined_prompt', 'N/A'),
                "creativity_level": prompt_understanding.get('creativity_level', 'N/A')
            },
            "model_selection": {
                "chosen_model": regen_config.get('selected_model', 'N/A'),
                "selection_reasoning": regen_config.get('reasoning', 'N/A'),
                "confidence_score": regen_config.get('confidence_score', 'N/A')
            },
            "generation_params": {
                "generating_prompt": regen_config.get('generating_prompt', 'N/A'),
                "negative_prompt": regen_config.get('negative_prompt', 'N/A'),
                "seed": config_data.get('seed', 'N/A'),
                "has_reference_image": bool(regen_config.get('reference_content_image'))
            },
            "evaluation_metrics": {
                "evaluation_score": regen_config.get('evaluation_score', 'N/A'),
                "user_feedback": regen_config.get('user_feedback', 'N/A'),
                "improvement_suggestions": regen_config.get('improvement_suggestions', 'N/A')
            },
            "system_context": {
                "human_in_loop": config_data.get('is_human_in_loop', False),
                "total_attempts": config_data.get('regeneration_count', 0) + 1,
                "current_attempt": regen_config.get('attempt_number', 1)
            }
        }
        
        # Prepare the message content for the LLM
        message_content = []
        
        # Add the text analysis part with structured information
        analysis_text = f"""You are an expert AI model performance analyst. Analyze the {model_name} model's performance in this image generation task.

WORKFLOW EXECUTION EXPLANATION:
{process_summary}

EXECUTION-SPECIFIC DATA (in workflow sequence):

CREATIVITY LEVEL SETTING:
• Level: {config_context['prompt_details']['creativity_level']}
• Impact: Determined how autonomously the system handled ambiguous prompt elements

INTENTION ANALYSIS RESULTS:
• Original prompt: "{config_context['prompt_details']['original']}"
• Analysis findings: System identified visual elements and ambiguous aspects requiring interpretation

PROMPT REFINEMENT OUTPUT:
• Refined prompt: "{config_context['prompt_details']['refined']}"
• Refinement quality: {'Significant refinement applied' if config_context['prompt_details']['original'] != config_context['prompt_details']['refined'] else 'Minimal refinement needed'}

NEGATIVE PROMPT CREATION:
• Negative prompt applied: "{config_context['generation_params']['negative_prompt']}"
• Purpose: Targeted prevention of unwanted artifacts and quality issues

PROMPT POLISHING:
• Final generating prompt: "{config_context['generation_params']['generating_prompt']}"
• Optimization: Model-specific tuning for {config_context['model_selection']['chosen_model']}

GENERATION EXECUTION:
• Selected model: {config_context['model_selection']['chosen_model']}
• Selection reasoning: {config_context['model_selection']['selection_reasoning']}
• System confidence: {config_context['model_selection']['confidence_score']}/10
• Reference image used: {'Yes' if config_context['generation_params']['has_reference_image'] else 'No'}
• Generation seed: {config_context['generation_params']['seed']}

EVALUATION RESULTS:
• Automated score: {config_context['evaluation_metrics']['evaluation_score']}/10
• User feedback: {config_context['evaluation_metrics']['user_feedback'] or 'None provided'}

REGENERATION STATUS:
• Current attempt: #{config_context['system_context']['current_attempt']} of {config_context['system_context']['total_attempts']} total
• Improvement suggestions: {config_context['evaluation_metrics']['improvement_suggestions'] or 'None from previous cycles'}
• Human oversight: {'Enabled' if config_context['system_context']['human_in_loop'] else 'Autonomous operation'}

ANALYSIS REQUIREMENTS:
Analyze this model's performance considering the execution flow:
1. How well did the model respond to the creativity level and prompt refinement quality?
2. Effectiveness of negative prompt in preventing unwanted artifacts
3. Quality of prompt polishing for this specific model's characteristics
4. Model selection appropriateness based on the refined prompt requirements
5. Technical execution quality visible in the generated image
6. Prompt adherence and creative interpretation balance
7. Reference image utilization (if applicable)

IMPORTANT: You will see the actual generated image(s) below. Provide specific visual analysis referencing the execution context.

Return a JSON response with detailed breakdown by workflow trajectory:
{{
    "trajectory_reasoning": "Overall analysis of how the workflow execution played out from start to finish, including key decision points, transitions between steps, and how each step influenced the next. Analyze the logical flow and coherence of the entire process.",
    "step_scores": {{
        "creativity_level": "Score from 1-10 how appropriate the creativity level setting was for this specific prompt",
        "intention_analysis": "Score from 1-10 how effective the intention analysis was for this prompt",
        "prompt_refinement": "Score from 1-10 how well the prompt was refined for optimal generation",
        "negative_prompt": "Score from 1-10 how effective the negative prompt was in preventing issues",
        "prompt_polishing": "Score from 1-10 how well the prompt was optimized for this model",
        "generation": "Score from 1-10 the overall quality of the generated image",
        "evaluation": "Score from 1-10 how accurate the system's evaluation was"
    }},
    "good_things": {{
        "creativity_level": "How well the creativity level setting worked for this prompt and model",
        "intention_analysis": "Effectiveness of the intention analysis in guiding the process", 
        "prompt_refinement": "Quality and appropriateness of the prompt refinement",
        "negative_prompt": "How well the negative prompt prevented unwanted artifacts",
        "prompt_polishing": "Effectiveness of model-specific prompt optimization",
        "generation": "Model selection appropriateness and generation quality",
        "evaluation": "Accuracy of evaluation scoring relative to visual quality"
    }},
    "bad_things": {{
        "creativity_level": "Issues with creativity level setting for this prompt type",
        "intention_analysis": "Missed elements or incorrect analysis in intention phase",
        "prompt_refinement": "Problems with refined prompt quality or completeness", 
        "negative_prompt": "Artifacts that negative prompt failed to prevent",
        "prompt_polishing": "Poor model-specific optimization or prompt structure",
        "generation": "Model selection issues or generation quality problems",
        "evaluation": "Evaluation inaccuracies or scoring misalignment"
    }},
    "overall_rating": "Rate from 1-10 the overall effectiveness of the entire workflow for this specific prompt"
}}

Focus on specific observations from the actual generated image and how each workflow step contributed to or detracted from the final result."""

        message_content.append({
            "type": "text",
            "text": analysis_text
        })
        
        # Add the generated image for analysis
        gen_image_path = regen_config.get('gen_image_path', '')
        if gen_image_path and os.path.exists(gen_image_path):
            try:
                with open(gen_image_path, "rb") as image_file:
                    base64_gen_image = base64.b64encode(image_file.read()).decode()
                
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_gen_image}",
                        "detail": "high"
                    }
                })
                message_content.append({
                    "type": "text", 
                    "text": f"↑ Generated Image by {model_name}"
                })
                self.logger.debug(f"Added generated image to analysis: {gen_image_path}")
            except Exception as e:
                self.logger.warning(f"Could not load generated image {gen_image_path}: {str(e)}")
                message_content.append({
                    "type": "text",
                    "text": f"[Generated image could not be loaded: {gen_image_path}]"
                })
        else:
            message_content.append({
                "type": "text",
                "text": f"[No generated image available for analysis: {gen_image_path}]"
            })
        
        # For Qwen-Image-Edit, also include the reference image for comparison
        if model_name == "Qwen-Image-Edit":
            ref_image_path = regen_config.get('reference_content_image', '')
            if ref_image_path and os.path.exists(ref_image_path):
                try:
                    with open(ref_image_path, "rb") as image_file:
                        base64_ref_image = base64.b64encode(image_file.read()).decode()
                    
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_ref_image}",
                            "detail": "high"
                        }
                    })
                    message_content.append({
                        "type": "text",
                        "text": "↑ Reference Image (input for editing)"
                    })
                    message_content.append({
                        "type": "text",
                        "text": "Please compare the reference image with the generated image to analyze how well the editing was performed."
                    })
                    self.logger.debug(f"Added reference image to analysis: {ref_image_path}")
                except Exception as e:
                    self.logger.warning(f"Could not load reference image {ref_image_path}: {str(e)}")
                    message_content.append({
                        "type": "text",
                        "text": f"[Reference image could not be loaded: {ref_image_path}]"
                    })

        # Make the LLM call with visual content
        response = track_llm_call(llm_json.invoke, "model_performance_analysis", [
            ("system", "You are an expert AI model performance analyst with strong visual analysis capabilities."),
            ("human", message_content)
        ])
        
        try:
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, dict):
                result = response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response.content)}")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse model performance analysis JSON: {str(e)}")
            self.logger.error(f"Raw response: {response.content}")
            # Fallback analysis data
            result = {
                "good_things": "Unable to parse analysis response - manual review recommended",
                "bad_things": "JSON parsing failed for model performance analysis"
            }
        
        self.logger.debug(f"Model analysis for {model_name}: {result}")
        return result
    
    def save_model_memory(self, config_data: Dict[str, Any]):
        """Save model performance analysis to memory database and update RAG indices."""
        print(f"DEBUG: save_model_memory called with config_data keys: {list(config_data.keys())}")
        
        # Initialize RAG system if needed
        if self.embedding_model is None:
            self._init_rag_system()
        
        # Generate process summary
        process_summary = self._generate_process_summary(config_data)
        print(f"DEBUG: Generated process summary: {len(process_summary)} characters")
        
        # Get regeneration configs to determine which models were used
        regen_configs = config_data.get("regeneration_configs", {})
        regen_count = config_data.get("regeneration_count", 0)
        
        print(f"DEBUG: regen_configs keys: {list(regen_configs.keys())}")
        print(f"DEBUG: regen_count: {regen_count}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        # Process each regeneration attempt
        for i in range(regen_count + 1):
            config_key = f"count_{i}"
            print(f"DEBUG: Processing attempt {i}, looking for config_key: {config_key}")
            
            if config_key not in regen_configs:
                print(f"DEBUG: Config key {config_key} not found in regen_configs")
                continue
                
            regen_config = regen_configs[config_key]
            model_name = regen_config.get('selected_model', '')
            
            print(f"DEBUG: Found regen_config for attempt {i}, model_name: {model_name}")
            
            if not model_name:
                print(f"DEBUG: No model_name found for attempt {i}")
                continue
            
            print(f"DEBUG: About to analyze model performance for {model_name}")
            # Analyze model performance
            analysis = self._analyze_model_performance(config_data, process_summary, model_name, regen_config)
            print(f"DEBUG: Got analysis result: {analysis}")
            
            # Prepare common data
            common_data = {
                'timestamp': timestamp,
                'image_index': config_data.get('image_index', ''),
                'original_prompt': config_data.get('prompt_understanding', {}).get('original_prompt', ''),
                'refined_prompt': config_data.get('prompt_understanding', {}).get('refined_prompt', ''),
                'evaluation_score': regen_config.get('evaluation_score', 0.0),
                'confidence_score': regen_config.get('confidence_score', 0.0),
                'regeneration_count': i,  # Current attempt number
                'reference_image': regen_config.get('reference_content_image', ''),
                'trajectory_reasoning': analysis.get('trajectory_reasoning', ''),
                'step_scores': json.dumps(analysis.get('step_scores', {})) if isinstance(analysis.get('step_scores'), dict) else str(analysis.get('step_scores', {})),
                'good_things': json.dumps(analysis.get('good_things', {})) if isinstance(analysis.get('good_things'), dict) else str(analysis.get('good_things', '')),
                'bad_things': json.dumps(analysis.get('bad_things', {})) if isinstance(analysis.get('bad_things'), dict) else str(analysis.get('bad_things', '')),
                'overall_rating': float(analysis.get('overall_rating', 0)) if analysis.get('overall_rating') and str(analysis.get('overall_rating')).replace('.','',1).isdigit() else 0.0,
                'config_data': json.dumps(config_data) if isinstance(config_data, dict) else str(config_data),
                'process_summary': process_summary,
            }
            
            print(f"DEBUG: Prepared common_data for {model_name}")
            
            if model_name == "Qwen-Image":
                print(f"DEBUG: Inserting into qwen_image_memory")
                cursor.execute('''
                    INSERT INTO qwen_image_memory 
                    (timestamp, image_index, original_prompt, refined_prompt, evaluation_score, 
                     confidence_score, regeneration_count, trajectory_reasoning, step_scores, 
                     good_things, bad_things, overall_rating, config_data, process_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    common_data['timestamp'], common_data['image_index'], common_data['original_prompt'],
                    common_data['refined_prompt'], common_data['evaluation_score'], common_data['confidence_score'],
                    common_data['regeneration_count'], common_data['trajectory_reasoning'], common_data['step_scores'],
                    common_data['good_things'], common_data['bad_things'], common_data['overall_rating'],
                    common_data['config_data'], common_data['process_summary']
                ))
                self.logger.info(f"Saved Qwen-Image memory for attempt {i+1}")
                print(f"DEBUG: Successfully inserted into qwen_image_memory for attempt {i+1}")
                
                # Update RAG index
                try:
                    self.logger.debug(f"Updating RAG index for qwen_image with prompt: {common_data['original_prompt'][:50]}...")
                    self._update_rag_index("qwen_image", common_data['original_prompt'], analysis)
                    self.logger.debug("RAG index update completed successfully")
                except Exception as rag_error:
                    self.logger.error(f"Failed to update RAG index for qwen_image: {str(rag_error)}")
                    import traceback
                    self.logger.error(f"RAG index error traceback: {traceback.format_exc()}")
                
            elif model_name == "Qwen-Image-Edit":
                # Save Qwen-Image-Edit memory for all attempts (fixed: was previously only saving for i > 0)
                print(f"DEBUG: Inserting into qwen_image_edit_memory for attempt {i+1}")
                cursor.execute('''
                    INSERT INTO qwen_image_edit_memory 
                    (timestamp, image_index, original_prompt, refined_prompt, evaluation_score, 
                     confidence_score, regeneration_count, reference_image, trajectory_reasoning, 
                     step_scores, good_things, bad_things, overall_rating, config_data, process_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    common_data['timestamp'], common_data['image_index'], common_data['original_prompt'],
                    common_data['refined_prompt'], common_data['evaluation_score'], common_data['confidence_score'],
                    common_data['regeneration_count'], common_data['reference_image'],
                    common_data['trajectory_reasoning'], common_data['step_scores'],
                    common_data['good_things'], common_data['bad_things'], common_data['overall_rating'],
                    common_data['config_data'], common_data['process_summary']
                ))
                self.logger.info(f"Saved Qwen-Image-Edit memory for attempt {i+1}")
                print(f"DEBUG: Successfully inserted into qwen_image_edit_memory for attempt {i+1}")
                
                # Update RAG index
                try:
                    self.logger.debug(f"Updating RAG index for qwen_image_edit with prompt: {common_data['original_prompt'][:50]}...")
                    self._update_rag_index("qwen_image_edit", common_data['original_prompt'], analysis)
                    self.logger.debug("RAG index update completed successfully for qwen_image_edit")
                except Exception as rag_error:
                    self.logger.error(f"Failed to update RAG index for qwen_image_edit: {str(rag_error)}")
                    import traceback
                    self.logger.error(f"RAG index error traceback: {traceback.format_exc()}")
        
        print(f"DEBUG: About to commit changes to database")
        conn.commit()
        conn.close()
        self.logger.info(f"Successfully saved model memory to database: {self.db_path}")
        print(f"DEBUG: Database operations completed successfully")
        
        # Save RAG indices to disk
        models_used = set()
        for i in range(regen_count + 1):
            config_key = f"count_{i}"
            if config_key in regen_configs:
                model_name = regen_configs[config_key].get('selected_model', '')
                if model_name:
                    models_used.add(model_name)
        
        print(f"DEBUG: Models used: {models_used}")
        for model_name in models_used:
            if model_name == "Qwen-Image":
                model_key = "qwen_image"
            elif model_name == "Qwen-Image-Edit":
                model_key = "qwen_image_edit"
            else:
                self.logger.warning(f"Unknown model name for RAG index: {model_name}")
                continue
            self._save_rag_index(model_key)

    def get_model_memory_summary(self, model_name: str, limit: int = 10) -> List[Dict]:
        """Retrieve recent memory entries for a specific model."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            table_name = "qwen_image_memory" if model_name == "Qwen-Image" else "qwen_image_edit_memory"
            
            cursor.execute(f'''
                SELECT timestamp, image_index, original_prompt, evaluation_score, 
                       confidence_score, good_things, bad_things
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            columns = ['timestamp', 'image_index', 'original_prompt', 'evaluation_score', 
                      'confidence_score', 'good_things', 'bad_things']
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory for {model_name}: {str(e)}")
            return []


def view_memory_database(db_path: str = "model_memory.db"):
    """Utility function to view contents of the memory database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"\n=== Memory Database Contents: {db_path} ===")
        
        # Check Qwen-Image memory
        cursor.execute("SELECT COUNT(*) FROM qwen_image_memory")
        qwen_image_count = cursor.fetchone()[0]
        print(f"\nQwen-Image Memory Entries: {qwen_image_count}")
        
        if qwen_image_count > 0:
            cursor.execute('''
                SELECT timestamp, image_index, original_prompt, evaluation_score, 
                       confidence_score, regeneration_count, good_things, bad_things
                FROM qwen_image_memory 
                ORDER BY timestamp DESC 
                LIMIT 3
            ''')
            results = cursor.fetchall()
            for i, row in enumerate(results):
                print(f"\n  Entry {i+1}:")
                print(f"    Timestamp: {row[0]}")
                print(f"    Image Index: {row[1]}")
                print(f"    Original Prompt: {row[2][:100]}...")
                print(f"    Evaluation Score: {row[3]}")
                print(f"    Confidence Score: {row[4]}")
                print(f"    Regeneration Count: {row[5]}")
                print(f"    Good Things: {row[6][:100]}...")
                print(f"    Bad Things: {row[7][:100]}...")
        
        # Check Qwen-Image-Edit memory
        cursor.execute("SELECT COUNT(*) FROM qwen_image_edit_memory")
        qwen_edit_count = cursor.fetchone()[0]
        print(f"\nQwen-Image-Edit Memory Entries: {qwen_edit_count}")
        
        if qwen_edit_count > 0:
            cursor.execute('''
                SELECT timestamp, image_index, original_prompt, evaluation_score, 
                       confidence_score, regeneration_count, reference_image, good_things, bad_things
                FROM qwen_image_edit_memory 
                ORDER BY timestamp DESC 
                LIMIT 3
            ''')
            results = cursor.fetchall()
            for i, row in enumerate(results):
                print(f"\n  Entry {i+1}:")
                print(f"    Timestamp: {row[0]}")
                print(f"    Image Index: {row[1]}")
                print(f"    Original Prompt: {row[2][:100]}...")
                print(f"    Evaluation Score: {row[3]}")
                print(f"    Confidence Score: {row[4]}")
                print(f"    Regeneration Count: {row[5]}")
                print(f"    Reference Image: {row[6]}")
                print(f"    Good Things: {row[7][:100]}...")
                print(f"    Bad Things: {row[8][:100]}...")
        
        conn.close()
        print(f"\n=== End Database Contents ===\n")
        
    except Exception as e:
        print(f"Error viewing database: {str(e)}")


def load_models(use_quantization=True):
    """Pre-load models with proper GPU memory management
    
    Args:
        use_quantization (bool): Whether to use quantization and FP16 for reduced memory usage.
            If False, models will be loaded in FP32 without quantization for higher precision.
    """
    global qwen_image_pipe, qwen_edit_pipe, embedding_model

    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Load embedding model first
        print("Loading Qwen embedding model...")
        embedding_model = pipeline("feature-extraction", model="Qwen/Qwen3-Embedding-0.6B", device="cuda")
        print("Successfully loaded Qwen embedding model")
        
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
        print("Seed: ",seed)
        generator = torch.Generator("cuda").manual_seed(seed)
        
        with torch.inference_mode():
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

        # Create category-based folder structure for CSV benchmarks
        if config.category:
            # For CSV benchmarks: create images/{category}/qwen/ structure
            category_dir = os.path.join(config.save_dir, "images", config.category.lower().replace('_', ''))
            model_dir = os.path.join(category_dir, "qwen")
            os.makedirs(model_dir, exist_ok=True)
            
            # Set output path with .webp format in category/model folder
            if config.regeneration_count > 0:
                output_path = os.path.join(model_dir, f"{config.image_index}_regen{config.regeneration_count}.webp")
            else:
                output_path = os.path.join(model_dir, f"{config.image_index}.webp")
        else:
            # For other benchmarks: use original structure
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
    
    # Create category-based folder structure for CSV benchmarks
    if config.category:
        # For CSV benchmarks: create images/{category}/qwen/ structure
        category_dir = os.path.join(config.save_dir, "images", config.category.lower().replace('_', ''))
        model_dir = os.path.join(category_dir, "qwen")
        os.makedirs(model_dir, exist_ok=True)
        
        # Set output path with .webp format in category/model folder
        if config.regeneration_count > 0:
            output_path = os.path.join(model_dir, f"{config.image_index}_regen{config.regeneration_count}.webp")
        else:
            output_path = os.path.join(model_dir, f"{config.image_index}.webp")
    else:
        # For other benchmarks: use original structure
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
        print("Seed: ",seed)
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
    def __init__(self, llm, memory_manager=None):
        self.llm = llm
        self.llm_json = llm.bind(response_format={"type": "json_object"})
        self.logger = logger
        self.negative_prompt_generator = NegativePromptGenerator(llm)  # Add this line
        self.memory_manager = memory_manager
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
        # Get comprehensive workflow context
        current_step = "model_selection"
        current_substep = "negative_model_selection"  # This method handles the overall negative prompt and model selection process
        workflow_context = get_workflow_context(current_step, current_substep, config_obj=config)
        
        # Use only workflow context (contains all necessary guidance)
        workflow_guidance = format_workflow_guidance_text(workflow_context, include_overview=False)
        guidance_text = f"\n\n{workflow_guidance}"
            
        if config.regeneration_count > 0:
            base_system_prompt = """Select the most suitable model for the given task and generate both positive and negative prompts.
            
            Available models:
            1. Qwen-Image: {qwen_image_desc}
            2. Qwen-Image-Edit: {qwen_edit_desc}

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
            
            # Construct system prompt with guidance BEFORE the base prompt
            return guidance_text + "\n\n" + base_system_prompt.format(
                qwen_image_desc=self.available_models['Qwen-Image'],
                qwen_edit_desc=self.available_models['Qwen-Image-Edit']
            )
        else:
            base_else_system_prompt = """Generate the most suitable prompt for the given task using Qwen-Image, including both positive and negative prompts.
            
            # Qwen-Image: {qwen_image_desc}

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
            
            # Construct system prompt with guidance BEFORE the base prompt
            return guidance_text + "\n\n" + base_else_system_prompt.format(
                qwen_image_desc=self.available_models['Qwen-Image']
            )

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
        self.logger.debug("Starting model selection process")
        
        try:
            # Create base system prompt and task-specific prompt
            base_prompt = self._create_system_prompt()
            task_prompt = self._create_task_prompt()
            
            self.logger.debug(f"Base system prompt length: {len(base_prompt)} characters")
            self.logger.debug(f"Task prompt length: {len(task_prompt)} characters")

            # === STEP 4: MODEL SELECTION & NEGATIVE PROMPT ===
            # Always log model selection as it's a core workflow step
            self.logger.info(f"=== STEP 4: MODEL SELECTION & NEGATIVE PROMPT ===")
            self.logger.info(f"System Prompt Length: {len(base_prompt)} characters")
            self.logger.info(f"System Prompt:\n{base_prompt}")
            self.logger.info(f"Task Prompt Length: {len(task_prompt)} characters")  
            self.logger.info(f"Task Prompt:\n{task_prompt}")

            response = track_llm_call(self.llm_json.invoke, "negative_model_selection", [
                ("system", base_prompt),
                ("human", task_prompt)
            ])

            result = self._parse_llm_response(response)            # === STEP 4 OUTPUT ===
            selected_model = result.get('selected_model', 'Unknown')
            generating_prompt = result.get('generating_prompt', '')
            negative_prompt = result.get('negative_prompt', '')
            
            self.logger.info(f"=== STEP 4 OUTPUT: MODEL & PROMPTS ===")
            self.logger.info(f"Selected Model: {selected_model}")
            self.logger.info(f"Generating Prompt: {generating_prompt}")
            self.logger.info(f"Negative Prompt: {negative_prompt}")
            
            self.logger.debug(f"Model selection result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Ensure negative_prompt is present
            if "negative_prompt" not in result or not result["negative_prompt"]:
                self.logger.warning("No negative prompt in model selection result, generating one...")
                # Generate negative prompt based on the positive prompt with guidance
                analysis = None
                if hasattr(config, 'prompt_understanding') and config.prompt_understanding.get('prompt_analysis'):
                    try:
                        if isinstance(config.prompt_understanding['prompt_analysis'], str):
                            analysis = json.loads(config.prompt_understanding['prompt_analysis'])
                        else:
                            analysis = config.prompt_understanding['prompt_analysis']
                    except:
                        analysis = None
                
                # Get RAG guidance for negative prompt generation and model selection
                current_step = "model_selection"
                current_substep = "negative_model_selection"
                workflow_context = get_workflow_context(current_step, current_substep, config_obj=config)
                
                neg_result = self.negative_prompt_generator.generate_negative_prompt(
                    result.get("generating_prompt", ""), 
                    analysis,
                    workflow_context
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
    logger.info(f"Message count: {len(state['messages'])}")
    logger.info(f"Overall messages: {state['messages']}")
    logger.info(f"Current config: {config.to_dict()}")

    # First interaction - analyze the prompt
    if len(state["messages"]) == 1:
        config.prompt_understanding["original_prompt"] = last_message.content
        logger.info("Step 1: Determining creativity level based on prompt")
        
        # Get RAG guidance for creativity level
        # Get workflow context for creativity level determination
        creativity_workflow_context = get_workflow_context("intention", "creativity_level_determination", config_obj=config)
        
        # Use workflow context directly (contains all necessary guidance)
        if creativity_workflow_context:
            workflow_guidance = format_workflow_guidance_text(creativity_workflow_context, include_overview=True)
            
            # Determine creativity level using workflow context (will log complete system prompt)
            determined_creativity_level = analyzer.determine_creativity_level(
                last_message.content, 
                workflow_context=creativity_workflow_context
            )
        config.prompt_understanding["creativity_level"] = determined_creativity_level
        logger.info(f"Determined creativity level: {determined_creativity_level.value}")
        
        logger.info("Step 2: Analyzing prompt")
        try:
            # Get comprehensive workflow context for intention analysis
            current_step = "intention"
            current_substep = "intention_analysis"
            workflow_context = get_workflow_context(current_step, current_substep, config_obj=config)
            
            # Analyze prompt with workflow context
            analysis = analyzer.analyze_prompt(
                last_message.content, 
                config.prompt_understanding["creativity_level"],
                workflow_context=workflow_context
            )
            config.prompt_understanding["prompt_analysis"] = json.dumps(analysis)
            logger.info(f"Analysis result: {config.prompt_understanding['prompt_analysis']}")
            
            # Retrieve references info
            analyzer.retrieve_reference(analysis)
            logger.info(f"Current config for retrieved reference info:\n {config.to_dict()}")

            # Retrieve questions
            questions = analyzer.retrieve_questions(analysis, config.prompt_understanding["creativity_level"])
            logger.info(f"Suggested questions for users:\n {questions}")
            
            if questions == "SUFFICIENT_DETAIL" or questions == "AUTOCOMPLETE" or not config.is_human_in_loop:
                # Get workflow context for prompt refinement  
                refinement_workflow_context = get_workflow_context("intention", "prompt_refinement")
                
                # Refine prompt directly with workflow context
                refinement_result = analyzer.refine_prompt_with_analysis(
                    last_message.content,
                    analysis,
                    creativity_level=config.prompt_understanding["creativity_level"],
                    workflow_context=refinement_workflow_context
                )
                config.prompt_understanding["refined_prompt"] = refinement_result['refined_prompt']
                
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
                    
                    # Get workflow context for prompt refinement
                    refinement_workflow_context = get_workflow_context("intention", "prompt_refinement")
                    
                    # Refine prompt with user input and workflow context
                    refinement_result = analyzer.refine_prompt_with_analysis(
                        config.prompt_understanding['original_prompt'],
                        analysis,
                        user_responses,
                        config.prompt_understanding["creativity_level"],
                        workflow_context=refinement_workflow_context
                    )
                    config.prompt_understanding["refined_prompt"] = refinement_result['refined_prompt']
                    if 'suggested_creativity_level' in refinement_result:
                        if "LOW" in refinement_result['suggested_creativity_level']:
                            config.prompt_understanding["creativity_level"] = CreativityLevel.LOW
                        elif "MEDIUM" in refinement_result['suggested_creativity_level']:
                            config.prompt_understanding["creativity_level"] = CreativityLevel.MEDIUM
                        elif "HIGH" in refinement_result['suggested_creativity_level']:
                            config.prompt_understanding["creativity_level"] = CreativityLevel.HIGH
                        logger.info(f"Update creativity level to {config.prompt_understanding['creativity_level']}")
                    
                    command = Command(
                                    update={"messages": state["messages"] + [AIMessage(content=f" User provides clarification. Refined prompt: {config.prompt_understanding['refined_prompt']}")]},
                                    goto="model_selection"
                                )
                    logger.debug(f"Command: {command}")
                    return command
                else:
                    # If not human_in_the_loop, proceed with auto-refinement
                    # Get workflow context for prompt refinement  
                    refinement_workflow_context = get_workflow_context("intention", "prompt_refinement")
                    
                    refinement_result = analyzer.refine_prompt_with_analysis(
                        last_message.content,
                        analysis,
                        creativity_level=CreativityLevel.HIGH,
                        workflow_context=refinement_workflow_context
                    )
                    config.prompt_understanding["refined_prompt"] = refinement_result['refined_prompt']
                    
                    # Log auto-refinement output
                    logger.info("=== STEP 3 OUTPUT (Auto-refinement): ===")
                    logger.info(f"Refined Prompt: {refinement_result['refined_prompt']}")
                    if 'reasoning' in refinement_result:
                        logger.info(f"Reasoning: {refinement_result['reasoning']}")
                    
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
    logger.info("="*50)
    logger.info("ENTERING MODEL SELECTION NODE")
    logger.info(f"State messages count: {len(state['messages'])}")
    logger.info(f"Current regeneration count: {config.regeneration_count}")
    
    selector = ModelSelector(llm, memory_manager)
    
    current_config = config.get_current_config()
    logger.debug(f"Current config keys: {list(current_config.keys())}")
    if config.regeneration_count != 0:
        prev_regen_config = config.get_prev_config()
        logger.debug(f"Previous config evaluation score: {prev_regen_config.get('evaluation_score', 'N/A')}")
    
    try:
        # Select the most suitable model
        logger.debug("Starting model selection process")
        model_selection = selector.select_model()
        logger.debug(f"Model selection completed with keys: {list(model_selection.keys()) if isinstance(model_selection, dict) else 'Not a dict'}")
        
        # Update current config with model selection
        current_config["selected_model"] = model_selection["selected_model"]
        logger.info(f"Selected model: {model_selection['selected_model']}")
        logger.info(f"Model selection reasoning: {model_selection.get('reasoning', 'No reasoning provided')}")
        
        # Choose which model's guidance should be active for subsequent steps
        if current_config["selected_model"] == "Qwen-Image-Edit":
            if config.workflow_guidance.get("qwen_image_edit"):
                config.active_guidance_model = "qwen_image_edit"
                logger.info("Active workflow guidance switched to Qwen-Image-Edit")
                logger.debug(f"Edit guidance available: {len(config.workflow_guidance['qwen_image_edit'].get('positive_guidance', {}))} step analyses")
            else:
                logger.warning("No edit-specific guidance available, continuing with default guidance")
                config.active_guidance_model = "qwen_image"
        else:
            config.active_guidance_model = "qwen_image"
            logger.info("Active workflow guidance set to Qwen-Image")
            logger.debug(f"Default guidance available: {len(config.workflow_guidance.get('qwen_image', {}).get('positive_guidance', {}))} step analyses")
        
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

        # === STEP 5: PROMPT POLISHING ===
        logger.info(f"=== STEP 5: PROMPT POLISHING ===")
        # Get RAG guidance and workflow context for prompt polishing
        current_step = "model_selection"
        current_substep = "prompt_polishing"
        workflow_context = get_workflow_context(current_step, current_substep)
        
        # Polish the prompt before generation
        polished_prompt = polish_prompt_en(current_config["generating_prompt"], workflow_context)
        current_config["generating_prompt"] = polished_prompt  # Update with polished version
        
        logger.info(f"=== STEP 5 OUTPUT: POLISHED PROMPT ===")
        logger.info(f"Original: {model_selection['generating_prompt']}")
        logger.info(f"Polished: {polished_prompt}")

        # === STEP 6: IMAGE GENERATION ===
        logger.info(f"=== STEP 6: IMAGE GENERATION ===")
        logger.info(f"Model: {current_config['selected_model']}")
        logger.info(f"Generating Prompt: {current_config['generating_prompt']}")
        logger.info(f"Negative Prompt: {current_config['negative_prompt']}")
        if current_config.get('reference_content_image'):
            logger.info(f"Reference Image: {current_config['reference_content_image']}")

        # Execute the selected model with negative prompt (prompt already polished)
        gen_image_path = execute_model(
            model_name=current_config['selected_model'],
            prompt=current_config['generating_prompt'],  # Already polished
            negative_prompt=current_config['negative_prompt'],  # Pass negative prompt
            reference_content_image=current_config['reference_content_image'],
        )
        
        current_config["gen_image_path"] = gen_image_path

        # Log generation output
        logger.info(f"=== STEP 6 OUTPUT: IMAGE GENERATED ===")
        logger.info(f"Generated Image Path: {gen_image_path}")
        logger.info(f"Model Used: {current_config['selected_model']}")
        
        command = Command(
            update={"messages": state["messages"] + [
                AIMessage(content=f"Generated images using {current_config['selected_model']}. "
                         f"Image path saved in: {current_config['gen_image_path']}. "
                         f"Negative prompt: {current_config['negative_prompt']}")  # Include negative prompt in output
            ]},
            goto=END
        )
        logger.debug(f"Command: {command}")
        logger.info("MODEL SELECTION NODE COMPLETED SUCCESSFULLY")
        return command

    except Exception as e:
        logger.error(f"EXCEPTION IN MODEL SELECTION NODE: {str(e)}")
        logger.error(f"Exception type: {type(e)}")

        # Check if this is a regeneration attempt
        if config.regeneration_count > 0:
            logger.warning("Error during regeneration attempt, returning previous image")
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


def polish_prompt_en(original_prompt, workflow_context=None):
    
    # Prepare guidance text - use workflow guidance (includes processed RAG guidance)
    guidance_text = ""
    # Use workflow guidance as the primary guidance source (includes processed RAG guidance)
    if workflow_context:
        workflow_guidance = format_workflow_guidance_text(workflow_context, include_overview=False)
        guidance_text += f"\n\n{workflow_guidance}"
    
    SYSTEM_PROMPT = f'''
{guidance_text}
You are a Prompt optimizer designed to rewrite user inputs into high-quality Prompts that are more complete and expressive while preserving the original meaning.
Task Requirements:
1. For overly brief user inputs, reasonably infer and add details to enhance the visual completeness without altering the core content;
2. Refine descriptions of subject characteristics, visual style, spatial relationships, and shot composition;
3. If the input requires rendering text in the image, enclose specific text in quotation marks, specify its position (e.g., top-left corner, bottom-right corner) and style. This text should remain unaltered and not translated;
4. Match the Prompt to a precise, niche style aligned with the user's intent. If unspecified, choose the most appropriate style (e.g., realistic photography style);
5. Please ensure that the Rewritten Prompt is less than 200 words.
'''
    
    original_prompt = original_prompt.strip()
    magic_prompt = "Ultra HD, 4K, cinematic composition"
    
    # Log the complete polishing prompt with final guidance
    complete_human_prompt = f"User Input: {original_prompt}\n\n Rewritten Prompt:"
    
    # === STEP 5: PROMPT POLISHING ===
    # Always log prompt polishing as it's a core workflow step
    logger.info(f"=== STEP 5: PROMPT POLISHING ===")
    logger.info(f"System Prompt Length: {len(SYSTEM_PROMPT)} characters")
    logger.info(f"System Prompt:\n{SYSTEM_PROMPT}")
    logger.info(f"Human Prompt: User Input: {original_prompt}")
            
    response = track_llm_call(llm.invoke, "polish_prompt", [
                    ("system", SYSTEM_PROMPT),
                    ("human", complete_human_prompt)
                ])
    
    polished_prompt = response.content.strip()
    polished_prompt = polished_prompt.replace("\n", " ")
    
    final_prompt = polished_prompt + " " + magic_prompt
    
    # === STEP 5 OUTPUT ===
    logger.info(f"=== STEP 5 OUTPUT: POLISHED PROMPT ===")
    logger.info(f"FINAL: {final_prompt}")
            
    return final_prompt

def get_active_guidance(config_obj, logger_obj=None) -> Dict[str, Any]:
    """Get guidance from the currently active model only if minimum threshold is met."""
    global memory_manager
    
    active_model_key = getattr(config_obj, "active_guidance_model", "qwen_image")
    if logger_obj:
        logger_obj.debug(f"Getting active guidance from model: {active_model_key}")
    
    # Check if memory manager exists and minimum threshold is met
    if memory_manager and not memory_manager.is_guidance_available(active_model_key):
        if logger_obj:
            logger_obj.debug(f"Minimum threshold not met for {active_model_key}, returning empty guidance")
        return {"positive_guidance": {}, "unsuccessful_patterns": {}}
    
    if isinstance(config_obj.workflow_guidance, dict):
        # Get guidance for the active model
        active_guidance = config_obj.workflow_guidance.get(active_model_key, {})
        if logger_obj:
            logger_obj.debug(f"Active guidance has {len(active_guidance.get('positive_guidance', {}))} step analyses")
        return active_guidance
    
    if logger_obj:
        logger_obj.warning("workflow_guidance is not a dict, returning empty guidance")
    return {"positive_guidance": {}, "unsuccessful_patterns": {}}

def get_workflow_context(current_step: str, current_substep: str = "", include_substep_filter: bool = True, config_obj=None) -> Dict[str, Any]:
    """
    Get simplified workflow context for guidance.
    
    Args:
        current_step: The current main workflow step
        current_substep: The current sub-step within the main step
        include_substep_filter: If True, filter guidance to relevant substeps
        config_obj: Configuration object (will use global config if not provided)
    
    Returns:
        Dict containing current step info and active guidance
    """
    import logging
    global config, memory_manager
    logger = logging.getLogger(__name__)
    logger.debug(f"Getting workflow context for step: {current_step}, substep: {current_substep}")
    
    # Use passed config first, then global config
    config_to_use = config_obj if config_obj else config
    
    # Get guidance from config.workflow_guidance (which contains RAG guidance retrieved at workflow start)
    active_guidance = {"positive_guidance": {}, "unsuccessful_patterns": {}}
    
    if config_to_use and hasattr(config_to_use, 'workflow_guidance') and config_to_use.workflow_guidance:
        # Get the active model guidance
        active_model_key = getattr(config_to_use, "active_guidance_model", "qwen_image")
        active_guidance = config_to_use.workflow_guidance.get(active_model_key, active_guidance)
        logger.debug(f"Using stored workflow guidance for {active_model_key}: {len(active_guidance.get('positive_guidance', {}))} positive patterns")
    else:
        logger.debug("No workflow guidance available in config, using empty guidance")
    
    positive_guidance = active_guidance.get("positive_guidance", {})
    unsuccessful_patterns = active_guidance.get("unsuccessful_patterns", {})
    logger.debug(f"Available positive guidance: {list(positive_guidance.keys())}")

    # Determine workflow structure based on active model type
    active_model_key = getattr(config_to_use, "active_guidance_model", "qwen_image")
    if active_model_key == "qwen_image_edit":
        # Edit workflow: 4 steps
        all_substeps = [
            "negative_model_selection",
            "prompt_polishing", 
            "image_editing",
            "quality_evaluation"
        ]
    else:
        # Generation workflow: 8 steps
        all_substeps = [
            "creativity_level_determination",
            "intention_analysis", 
            "prompt_refinement",
            "negative_model_selection",
            "prompt_polishing",
            "image_generation",
            "quality_evaluation",
            "regeneration_decision"
        ]
    
    # Build relevant substeps list for current and upcoming substeps
    relevant_substeps = []
    if include_substep_filter and current_substep:
        # Find current substep index and include current + upcoming
        try:
            current_index = all_substeps.index(current_substep)
            relevant_substeps = all_substeps[current_index:]  # Current + all remaining substeps
        except ValueError:
            relevant_substeps = [current_substep]
    else:
        # Include all substeps
        relevant_substeps = all_substeps
    
    logger.debug(f"Relevant substeps for guidance: {relevant_substeps}")
    
    # Map substeps to proper step numbers and display names
    substep_to_step = {
        "creativity_level_determination": {"number": 1, "name": "Creativity Level Determination"},
        "intention_analysis": {"number": 2, "name": "Intention Analysis"},
        "prompt_refinement": {"number": 3, "name": "Prompt Refinement"},
        "negative_model_selection": {"number": 4, "name": "Negative Prompt & Model Selection"},
        "prompt_polishing": {"number": 5, "name": "Prompt Polishing"},
        "image_generation": {"number": 6, "name": "Image Generation"},
        "quality_evaluation": {"number": 7, "name": "Quality Evaluation"},
        "regeneration_decision": {"number": 8, "name": "Regeneration Decision"}
    }
    
    # Get step info from substep, with fallback
    if current_substep and current_substep in substep_to_step:
        step_info = substep_to_step[current_substep]
        step_number = step_info["number"]
        step_name = step_info["name"]
    else:
        # Fallback based on current_step
        step_number = 1
        step_name = current_step.replace('_', ' ').title()
    
    # Simple context structure
    context = {
        "current_position": {
            "step": current_step,
            "step_name": step_name,
            "substep": current_substep,
            "step_number": step_number
        },
        "filtered_substeps": relevant_substeps,
        "active_guidance": {
            "positive_guidance": positive_guidance,
            "unsuccessful_patterns": unsuccessful_patterns
        }
    }
    
    return context

def get_substep_explanation(substep: str, is_edit_workflow: bool = False) -> str:
    """Provide brief explanations for each workflow substep, context-aware for edit vs generation."""
    if is_edit_workflow:
        # Edit workflow explanations
        edit_explanations = {
            "negative_model_selection": "Configures Qwen-Image-Edit model and creates negative prompts to avoid unwanted elements in the edit",
            "prompt_polishing": "Optimizes the edit instruction for the Qwen-Image-Edit model's understanding",
            "image_editing": "Performs the actual image editing using the reference image and edit instructions",
            "quality_evaluation": "Assesses edit quality, blending seamlessness, and instruction adherence"
        }
        return edit_explanations.get(substep, "Specialized edit workflow substep")
    else:
        # Generation workflow explanations
        generation_explanations = {
            "creativity_level_determination": "Analyzes prompt specificity to determine how creative the system should be (LOW/MEDIUM/HIGH)",
            "intention_analysis": "Extracts key elements, style preferences, and user intent from the prompt",
            "prompt_refinement": "Clarifies and enhances the prompt while preserving core intent and adding necessary details",
            "negative_model_selection": "Selects Qwen-Image model and creates negative prompts to avoid unwanted elements",
            "prompt_polishing": "Final optimization of the prompt for optimal Qwen-Image model performance",
            "image_generation": "Generates the actual image using the Qwen-Image model and refined prompts",
            "quality_evaluation": "Assesses technical quality, prompt adherence, and artistic merit of generated content",
            "regeneration_decision": "Determines whether to regenerate based on quality thresholds and improvement potential"
        }
        return generation_explanations.get(substep, "Specialized generation workflow substep")

def format_workflow_guidance_text(workflow_context: Dict[str, Any], include_overview: bool = True) -> str:
    """Format workflow context into guidance text for LLM prompts.
    
    Always shows workflow substep explanations. When minimum DB threshold is met,
    adds specific guidance patterns for current and subsequent substeps only.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Formatting workflow guidance text, include_overview: {include_overview}")
    
    # Get active guidance to access positive_guidance and unsuccessful_patterns
    active_guidance = workflow_context.get("active_guidance", {})
    positive_guidance = active_guidance.get("positive_guidance", {})
    unsuccessful_patterns = active_guidance.get("unsuccessful_patterns", {})
    filtered_substeps = workflow_context.get("filtered_substeps", [])
    
    guidance_parts = []
    
    # Determine if this is an edit workflow from the active guidance model
    # Check if workflow_context has model information or derive from available substeps
    available_substeps = list(positive_guidance.keys()) if positive_guidance else []
    is_edit_workflow = ("image_editing" in available_substeps or 
                       len(available_substeps) <= 4 and "creativity_level_determination" not in available_substeps)
    
    # === ALWAYS PRESENT: WORKFLOW SUBSTEP EXPLANATIONS ===
    workflow_type_name = "IMAGE EDITING" if is_edit_workflow else "IMAGE GENERATION"
    guidance_parts.append(f"=== {workflow_type_name} WORKFLOW SUBSTEP EXPLANATIONS ===")
    guidance_parts.append(f"These substeps guide the {workflow_type_name.lower()} process:")
    guidance_parts.append("")
    
    # Use model-specific substeps
    if is_edit_workflow:
        all_substeps = [
            "negative_model_selection",
            "prompt_polishing", 
            "image_editing",
            "quality_evaluation"
        ]
    else:
        all_substeps = [
            "creativity_level_determination",
            "intention_analysis", 
            "prompt_refinement",
            "negative_model_selection",
            "prompt_polishing",
            "image_generation",
            "quality_evaluation",
            "regeneration_decision"
        ]
    
    for substep in all_substeps:
        substep_display = substep.replace('_', ' ').title()
        explanation = get_substep_explanation(substep, is_edit_workflow)
        guidance_parts.append(f"• {substep_display}: {explanation}")
    
    guidance_parts.append("")
    
    # Current position
    current = workflow_context["current_position"]
    guidance_parts.append(f"=== CURRENT STEP: {current['step_number']} - {current['step_name']} ===")
    if current["substep"]:
        substep_display = current["substep"].replace('_', ' ').title()
        guidance_parts.append(f"Current Focus: {substep_display}")
        guidance_parts.append(f"Purpose: {get_substep_explanation(current['substep'], is_edit_workflow)}")
    
    # === CONDITIONAL: SPECIFIC GUIDANCE (only when we have actual guidance patterns) ===
    # Check if we have any actual non-empty guidance patterns available for relevant substeps
    has_meaningful_guidance = False
    
    # Check if we have any non-empty guidance patterns for the filtered substeps
    substeps_to_check = filtered_substeps if filtered_substeps else all_substeps
    for substep in substeps_to_check:
        if (substep in positive_guidance and positive_guidance[substep].strip()) or \
           (substep in unsuccessful_patterns and unsuccessful_patterns[substep].strip()):
            has_meaningful_guidance = True
            break
    
    logger.debug(f"Checking guidance availability: positive={len(positive_guidance)}, unsuccessful={len(unsuccessful_patterns)}, has_meaningful={has_meaningful_guidance}")
    logger.debug(f"Using substeps: {substeps_to_check}")
    
    if has_meaningful_guidance:
        guidance_parts.append(f"\n=== LEARNED GUIDANCE PATTERNS ===")
        guidance_parts.append("Based on previous successful and unsuccessful generations:")
        guidance_parts.append("")
        
        # Show guidance for current and subsequent substeps only (filtered_substeps)
        guidance_shown = False
        logger.debug(f"Available positive guidance keys: {list(positive_guidance.keys())}")
        logger.debug(f"Available unsuccessful patterns keys: {list(unsuccessful_patterns.keys())}")
        
        for substep in substeps_to_check:
            has_positive = substep in positive_guidance and positive_guidance[substep].strip()
            has_negative = substep in unsuccessful_patterns and unsuccessful_patterns[substep].strip()
            
            if has_positive or has_negative:
                substep_display = substep.replace('_', ' ').title()
                guidance_parts.append(f"• {substep_display}:")
                
                if has_positive:
                    guidance_parts.append(f"  ✓ Success Patterns: {positive_guidance[substep]}")
                
                if has_negative:
                    guidance_parts.append(f"  ❌ Patterns to Avoid: {unsuccessful_patterns[substep]}")
                
                guidance_parts.append("")
                guidance_shown = True
        
        # If no meaningful guidance was shown (shouldn't happen due to our check above)
        if not guidance_shown:
            guidance_parts.append("  No specific patterns available.")
            guidance_parts.append("")
    
    # Don't show any guidance section when there's no meaningful guidance
    # This prevents empty "Success Patterns:" and "Patterns to Avoid:" from appearing
    
    return "\n".join(guidance_parts)

def execute_model(model_name: str, prompt: str, negative_prompt: str, reference_content_image: str = None) -> str:
    """Execute the selected model and return paths to generated images."""
    logger.debug(f"Starting model execution: {model_name}")
    logger.debug(f"Input prompt: '{prompt[:100]}...'")
    logger.debug(f"Negative prompt: '{negative_prompt}'")
    logger.debug(f"Reference image: {reference_content_image}")
    
    global qwen_image_pipe, qwen_edit_pipe
    global model_inference_times
    selector = ModelSelector(llm)
    if model_name not in selector.tools:
        logger.error(f"Unknown model requested: {model_name}")
        raise ValueError(f"Unknown model: {model_name}")

    # get regen count
    regen_count = config.regeneration_count
    if regen_count == 0:
        seed = config.seed
        logger.debug(f"Using configured seed: {seed}")
    else:
        # random seed
        seed = random.randint(0, 1000000)
        logger.debug(f"Generated random seed for regeneration: {seed}")
    
    # Note: Prompt polishing is now done before calling this function
    
    logger.info("="*50)
    logger.info(f"MODEL EXECUTION DETAILS:")
    logger.info(f"Model: {model_name}")
    logger.info(f"Final Prompt: {prompt}")
    logger.info(f"Negative Prompt: {negative_prompt}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Regeneration Count: {regen_count}")
    if reference_content_image:
        logger.info(f"Reference Image: {reference_content_image}")
    logger.info("="*50)
    
    # Also print for console visibility
    print("="*50)
    print("Final Prompt:", prompt)
    print("Negative Prompt:", negative_prompt)
    print("="*50)

    if model_name == "Qwen-Image":
        logger.debug("Executing Qwen-Image model")
        t0 = time.time()
        result = selector.tools[model_name].invoke({"prompt": prompt, "negative_prompt": negative_prompt, "seed": seed})
        t1 = time.time()
        execution_time = t1 - t0
        model_inference_times["Qwen-Image"].append(execution_time)
        logger.info(f"Qwen-Image execution completed in {execution_time:.2f} seconds")
        logger.debug(f"Qwen-Image result: {result}")
        return result
    elif model_name == "Qwen-Image-Edit":
        logger.debug("Executing Qwen-Image-Edit model")
        logger.info(f"Reference image path: {reference_content_image}")
        t0 = time.time()
        result = selector.tools[model_name].invoke({"prompt": prompt, "negative_prompt": negative_prompt, "existing_image_dir": reference_content_image, "seed": seed, "guidance_scale": 4.0})
        t1 = time.time()
        execution_time = t1 - t0
        model_inference_times["Qwen-Image-Edit"].append(execution_time)
        logger.info(f"Qwen-Image-Edit execution completed in {execution_time:.2f} seconds")
        logger.debug(f"Qwen-Image-Edit result: {result}")
        return result
    else:
        logger.error(f"Unknown model in execution branch: {model_name}")
        raise ValueError(f"Unknown model: {model_name}")

def evaluation_node(state: MessagesState) -> Command[str]:
    """Evaluate generated images and handle regeneration if needed."""
    last_message = state["messages"][-1]
    
    logger.info("-"*50)
    logger.info("INSIDE EVALUATION NODE")
    logger.info(f"Current config: {config.to_dict()}")
    
    try:
        current_config = config.get_current_config()
        
        # Prepare evaluation prompt with comprehensive workflow context
        current_step = "evaluation"
        current_substep = "quality_evaluation"
        workflow_context = get_workflow_context(current_step, current_substep)
        
        try:
            active_guidance = get_active_guidance(config, logger)
            evaluation_guidance = active_guidance.get("step_analysis", {}).get("evaluation", {}).get("success_patterns", "")
        except (AttributeError, TypeError):
            evaluation_guidance = ""
        
        guidance_text = ""
        if evaluation_guidance:
            guidance_text = f"\n\n=== GUIDANCE ===\n{evaluation_guidance}"
        
        # Add comprehensive workflow context
        workflow_guidance = format_workflow_guidance_text(workflow_context, include_overview=False)
        guidance_text += f"\n\n{workflow_guidance}"
        
        with open(current_config['gen_image_path'], "rb") as image_file:
            base64_gen_image = base64.b64encode(image_file.read()).decode("utf-8")
        evaluation_prompt = [
            (
                "system",
                guidance_text + "\n\n" + make_gen_image_judge_prompt(config)
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
        
        try:
            if isinstance(evaluation_result.content, str):
                evaluation_data = json.loads(evaluation_result.content)
            else:
                evaluation_data = evaluation_result.content
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {str(e)}")
            logger.error(f"Raw evaluation response: {evaluation_result.content}")
            # Fallback evaluation data
            evaluation_data = {
                "overall_score": 5.0,
                "improvement_suggestions": "Unable to parse evaluation response. Manual review recommended."
            }

        # Update config with evaluation score
        current_config["evaluation_score"] = evaluation_data.get("overall_score", 5.0)
        current_config["improvement_suggestions"] = evaluation_data.get("improvement_suggestions", "No specific suggestions available")
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

def check_existing_images(save_dir, image_index, category=None):
    """
    Check if images already exist for the given image_index in the save directory.
    
    Args:
        save_dir: Base save directory
        image_index: Index/ID of the image to check
        category: Category for CSV benchmarks (None for other benchmarks)
        
    Returns:
        dict: {
            'exists': bool,
            'paths': list of existing image paths,
            'has_base': bool (True if base image exists),
            'has_regenerations': bool (True if regeneration images exist)
        }
    """
    existing_paths = []
    has_base = False
    has_regenerations = False
    
    if category:
        # For CSV benchmarks: check images/{category}/qwen/ structure
        category_dir = os.path.join(save_dir, "images", category.lower().replace('_', ''))
        model_dir = os.path.join(category_dir, "qwen")
        
        # Check for base image
        base_path = os.path.join(model_dir, f"{image_index}.webp")
        if os.path.exists(base_path):
            existing_paths.append(base_path)
            has_base = True
            
        # Check for regeneration images
        regen_count = 1
        while True:
            regen_path = os.path.join(model_dir, f"{image_index}_regen{regen_count}.webp")
            if os.path.exists(regen_path):
                existing_paths.append(regen_path)
                has_regenerations = True
                regen_count += 1
            else:
                break
    else:
        # For other benchmarks: check direct save_dir structure
        # Check for base image
        base_path = os.path.join(save_dir, f"{image_index}_Qwen-Image.png")
        if os.path.exists(base_path):
            existing_paths.append(base_path)
            has_base = True
            
        # Also check for Qwen-Image-Edit
        base_edit_path = os.path.join(save_dir, f"{image_index}_Qwen-Image-Edit.png")
        if os.path.exists(base_edit_path):
            existing_paths.append(base_edit_path)
            has_base = True
            
        # Check for regeneration images
        regen_count = 1
        while True:
            regen_path = os.path.join(save_dir, f"{image_index}_regen{regen_count}_Qwen-Image.png")
            regen_edit_path = os.path.join(save_dir, f"{image_index}_regen{regen_count}_Qwen-Image-Edit.png")
            
            found_regen = False
            if os.path.exists(regen_path):
                existing_paths.append(regen_path)
                has_regenerations = True
                found_regen = True
            if os.path.exists(regen_edit_path):
                existing_paths.append(regen_edit_path)
                has_regenerations = True
                found_regen = True
                
            if not found_regen:
                break
            regen_count += 1
    
    return {
        'exists': len(existing_paths) > 0,
        'paths': existing_paths,
        'has_base': has_base,
        'has_regenerations': has_regenerations
    }

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
    
    llm_latencies[llm_type].append(latency)
    llm_token_counts[llm_type].append((prompt_tokens, completion_tokens, total_tokens))
    
    logger.info(f"LLM call {llm_type}: {latency:.2f}s, tokens: {prompt_tokens}+{completion_tokens}={total_tokens}")
    return response

def main(benchmark_name, human_in_the_loop, model_version, use_open_llm=False, open_llm_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503", open_llm_host="0.0.0.0", open_llm_port="8000", calculate_latency=False, use_quantization=True, min_entries_for_guidance=None, gpu_id=0, total_gpus=1):
    """Main CLI entry point."""
    # Check CUDA availability and initialize
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires a GPU to run.")
    
    # Initialize primary CUDA device - always use 0 since CUDA_VISIBLE_DEVICES limits visibility
    torch.cuda.init()
    torch.cuda.set_device(0)
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nUsing GPU {gpu_id}: {gpu_name}")
    print(f"Available GPU memory: {gpu_memory:.2f} GB")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Multi-GPU setup: GPU {gpu_id}/{total_gpus-1}")
    # Declare globals
    global logger, config
    global llm_latencies, llm_token_counts
    global model_inference_times
    global llm, llm_json
    global memory_manager

    # Initialize LLMs based on open_llm flag
    llm, llm_json = initialize_llms(use_open_llm, open_llm_model=open_llm_model, local_host=open_llm_host, local_port=open_llm_port)

    # Initialize memory manager (will create its own logger initially)
    # Set default minimum entries if not provided
    if min_entries_for_guidance is None:
        min_entries_for_guidance = {
            "qwen_image": 5,      # Default minimum entries for Qwen-Image guidance
            "qwen_image_edit": 5  # Default minimum entries for Qwen-Image-Edit guidance
        }
    
    memory_manager = MemoryManager(min_entries_for_guidance=min_entries_for_guidance)

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
            elif benchmark_name.endswith('.csv'):
                import csv
                csv_reader = csv.DictReader(file)
                prompts = {}
                for row in csv_reader:
                    # Create unique key combining category and id to handle duplicate IDs across categories
                    unique_key = f"{row['category']}_{row['id']}"
                    prompts[unique_key] = {
                        'category': row['category'],
                        'id': row['id'],
                        'prompt': row['prompt_en'],
                        'type': row['type'],
                        'prompt_length': row['prompt_length'],
                        'class': row.get('class', ''),
                        'unique_key': unique_key  # Store the unique key for reference
                    }
                prompt_keys = list(prompts.keys())
                bench_result_folder = 'OneIG-Bench'
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

        # Data splitting for multi-GPU processing
        total_items = len(prompt_keys)
        if total_gpus > 1:
            # Calculate data split for this GPU
            items_per_gpu = total_items // total_gpus
            remainder = total_items % total_gpus
            
            # Distribute remainder among first few GPUs
            if gpu_id < remainder:
                start_idx_gpu = gpu_id * (items_per_gpu + 1)
                end_idx_gpu = start_idx_gpu + items_per_gpu + 1
            else:
                start_idx_gpu = gpu_id * items_per_gpu + remainder
                end_idx_gpu = start_idx_gpu + items_per_gpu
            
            # Split the data for this GPU
            prompt_keys = prompt_keys[start_idx_gpu:end_idx_gpu]
            
            print(f"\n=== Multi-GPU Data Split ===")
            print(f"GPU {gpu_id}: Processing items {start_idx_gpu}-{end_idx_gpu-1} ({len(prompt_keys)} items)")
            print(f"Total dataset size: {total_items}")
            print(f"============================\n")
            
            # Also split seeds if they exist
            if "DrawBench" in str(benchmark_name) and "seed" in str(benchmark_name):
                seeds = seeds[start_idx_gpu:end_idx_gpu]
            
            # For GenAI benchmark, we need to handle the dictionary differently
            if "GenAIBenchmark" in str(benchmark_name) or benchmark_name.endswith('.csv'):
                # Filter the prompts dictionary to only include this GPU's keys
                gpu_prompts = {key: prompts[key] for key in prompt_keys}
                prompts = gpu_prompts

        # Create model type suffix for directory
        model_suffix = model_version
        if use_open_llm:
            # Get model name for the suffix - extract just the model name without org prefix
            model_name = open_llm_model.split('/')[-1]
            model_suffix += f"_open_llm_{model_name}"
        
        # For CSV benchmarks, create a different base structure
        if benchmark_name.endswith('.csv'):
            # For CSV: create a simple base directory without the AgentSys prefix
            if human_in_the_loop:
                base_save_dir = os.path.join("results", bench_result_folder, f"{model_suffix}_human_in_loop")
            else:
                base_save_dir = os.path.join("results", bench_result_folder, model_suffix)
        else:
            # For other benchmarks: use original AgentSys structure
            if human_in_the_loop:
                base_save_dir = os.path.join("results", bench_result_folder, f"AgentSys_{model_suffix}_human_in_loop")
            else:
                base_save_dir = os.path.join("results", bench_result_folder, f"AgentSys_{model_suffix}")
        
        # Add GPU-specific subdirectory for multi-GPU runs
        if total_gpus > 1:
            save_dir = os.path.join(base_save_dir, f"gpu_{gpu_id}")
        else:
            save_dir = base_save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
            "negative_model_selection": [], 
            "evaluation": [],
            "polish_prompt": [],
            "creativity_determination": [],
            "model_performance_analysis": [],
            "structured_workflow_guidance": []
        }
        llm_token_counts = {
            "intention_analysis": [], 
            "refine_prompt": [], 
            "negative_model_selection": [], 
            "evaluation": [],
            "polish_prompt": [],
            "creativity_determination": [],
            "model_performance_analysis": [],
            "structured_workflow_guidance": []
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
            

        for idx, key in tqdm(enumerate(prompt_keys[start_idx:]), total=len(prompts) - start_idx):
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
                config.seed = prompts[key].get("random_seed") if "seed" in str(benchmark_name) else torch.initial_seed()
                config.image_index = prompts[key]['id']
                config.category = None  # GenAI doesn't have categories
            elif benchmark_name.endswith('.csv'):
                text_prompt = prompts[key]['prompt']
                config.seed = torch.initial_seed()  # CSV doesn't have seeds, use random
                # Use the unique key as image_index to ensure uniqueness
                config.image_index = key  # key is now the unique "category_id"
                config.category = prompts[key]['category']
            else:  # DrawBench or other benchmarks
                text_prompt = key
                config.seed = seeds[idx + start_idx] if "seed" in str(benchmark_name) else torch.initial_seed()
                config.image_index = f"{(idx+start_idx):03d}"
                config.category = None  # DrawBench doesn't have categories
                print(f"Working on Benchmark name: {benchmark_name}")

            # Check if images already exist for this prompt
            existing_check = check_existing_images(save_dir, config.image_index, config.category)
            
            if existing_check['exists']:
                print(f"⏭️  SKIPPING {config.image_index}: Images already exist")
                print(f"   Found {len(existing_check['paths'])} existing images:")
                for path in existing_check['paths']:
                    print(f"   - {os.path.basename(path)}")
                
                # Update statistics as if this was a single-turn generation (0 time)
                single_turn_count += 1
                single_turn_times.append(0.0)
                end2end_times.append(0.0)
                inference_times.append(0.0)
                max_gpu_memories.append(0.0)
                
                # Update progress without logging or processing
                progress_data = {
                    "total_time": total_time,
                    "current_idx": idx + start_idx + 1,
                    "inference_times": inference_times
                }
                with open(progress_file, "w") as file:
                    json.dump(progress_data, file)
                
                # Update stats.json
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
                    "gpu_id": gpu_id,
                    "total_gpus": total_gpus,
                    "total_items_for_gpu": len(prompt_keys),
                    "completed_items": idx + start_idx + 1
                }
                with open(stats_file, "w") as f:
                    json.dump(stats, f)
                
                continue  # Skip to next iteration

            # Setup logging for this iteration
            logger = setup_logging(save_dir, filename=f"{config.image_index}.log", console_output=False)
            config.logger = logger

            # Use the global memory manager and update its logger for this iteration
            memory_manager.logger = logger  # Update logger to current iteration's logger
            
            # Log current database status
            memory_manager.log_guidance_status()
            
            # Set the original prompt for RAG guidance retrieval
            config.prompt_understanding["original_prompt"] = text_prompt
            
            # Retrieve RAG guidance for both models at the beginning of the workflow
            try:
                logger.info("="*80)
                logger.info("RETRIEVING RAG WORKFLOW GUIDANCE")
                logger.info("="*80)
                logger.info(f"Current prompt: '{text_prompt}'")
                
                # Load guidance for both model types
                qwen_image_guidance = memory_manager.get_workflow_guidance(text_prompt, "qwen_image")
                qwen_edit_guidance = memory_manager.get_workflow_guidance(text_prompt, "qwen_image_edit")
                
                # Log COMPLETE guidance content for transparency
                logger.info(f"\n" + "="*80)
                logger.info(f"COMPLETE EXTRACTED GUIDANCE CONTENT")
                logger.info(f"="*80)
                
                logger.info(f"\n--- QWEN-IMAGE COMPLETE GUIDANCE ---")
                if qwen_image_guidance.get('positive_guidance'):
                    logger.info(f"✓ Available steps: {list(qwen_image_guidance['positive_guidance'].keys())}")
                    for step_name, guidance_text in qwen_image_guidance['positive_guidance'].items():
                        logger.info(f"\n📋 {step_name.upper()}:")
                        logger.info(f"   GUIDANCE: {guidance_text}")
                        
                    # Show unsuccessful patterns if available
                    if qwen_image_guidance.get('unsuccessful_patterns'):
                        logger.info(f"\n❌ UNSUCCESSFUL PATTERNS:")
                        for step_name, warning_text in qwen_image_guidance['unsuccessful_patterns'].items():
                            if warning_text.strip():
                                logger.info(f"   {step_name}: {warning_text}")
                else:
                    logger.info("  ❌ No step guidance available")
                
                logger.info(f"\n--- QWEN-IMAGE-EDIT COMPLETE GUIDANCE ---")
                if qwen_edit_guidance.get('positive_guidance'):
                    logger.info(f"✓ Available steps: {list(qwen_edit_guidance['positive_guidance'].keys())}")
                    for step_name, guidance_text in qwen_edit_guidance['positive_guidance'].items():
                        logger.info(f"\n📋 {step_name.upper()}:")
                        logger.info(f"   GUIDANCE: {guidance_text}")
                        
                    # Show unsuccessful patterns if available
                    if qwen_edit_guidance.get('unsuccessful_patterns'):
                        logger.info(f"\n❌ UNSUCCESSFUL PATTERNS:")
                        for step_name, warning_text in qwen_edit_guidance['unsuccessful_patterns'].items():
                            if warning_text.strip():
                                logger.info(f"   {step_name}: {warning_text}")
                else:
                    logger.info("  ❌ No step guidance available")
                
                logger.info(f"\n" + "="*80)
                
                # Store both guidances in config
                config.workflow_guidance = {
                    "qwen_image": qwen_image_guidance,
                    "qwen_image_edit": qwen_edit_guidance
                }
                # Default active guidance source
                config.active_guidance_model = "qwen_image"
                config.rag_guidance_retrieved = True
                
                logger.info(f"\n--- GUIDANCE SUMMARY ---")
                logger.info(f"Qwen-Image guidance: {len(qwen_image_guidance.get('positive_guidance', {}))} step analyses")
                logger.info(f"Qwen-Image-Edit guidance: {len(qwen_edit_guidance.get('positive_guidance', {}))} step analyses")
                logger.info(f"Active guidance model: {config.active_guidance_model}")
                logger.info(f"="*80)
                logger.info("="*80)
                
            except Exception as e:
                logger.error(f"Failed to retrieve RAG guidance: {str(e)}")
                config.workflow_guidance = {
                    "qwen_image": {
                        "step_analysis": {},
                        "workflow_insights": {}
                    },
                    "qwen_image_edit": {
                        "step_analysis": {},
                        "workflow_insights": {}
                    }
                }

            # start timing
            torch.cuda.reset_peak_memory_stats(0)
            # torch.cuda.reset_peak_memory_stats(1)
            start_time = time.time()

            logger.info("\n" + "="*83)
            logger.info(f"New Session Started for index {config.image_index}: {datetime.now()}")
            logger.info("="*50)

            logger.info(f"Starting workflow with prompt: {text_prompt}, seed: {config.seed}")
            result = run_workflow(workflow, text_prompt)

            # Save config state after generation
            config_save_path = os.path.join(save_dir, f"{config.image_index}_config.json")
            config.save_to_file(config_save_path)
            logger.info(f"Saved config state to: {config_save_path}")
            
            # Analyze and save model performance to memory database
            try:
                logger.info("Analyzing model performance and saving to memory database...")
                memory_manager.save_model_memory(config.to_dict())
                logger.info("Successfully saved model performance analysis to memory database")
            except Exception as e:
                logger.error(f"Failed to save model performance analysis: {str(e)}")
            
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
            
            # Safely collect GPU memory stats for current GPU
            try:
                current_gpu_memory = torch.cuda.max_memory_allocated(0) / 1024**3
                max_gpu_memories.append(current_gpu_memory)
            except RuntimeError as e:
                logger.warning(f"Could not collect GPU memory stats for GPU {gpu_id}: {e}")
                max_gpu_memories.append(0)
                
            logger.info(f"Inference time for prompt {key}: {inference_time:.4f} seconds")

            logger.info("Workflow completed")    

            # Save progress
            progress_data = {
                "total_time": total_time,
                "current_idx": idx + start_idx + 1,
                "inference_times": inference_times
            }
            with open(progress_file, "w") as file:
                json.dump(progress_data, file)  
                
            # Save stats.json
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
                "gpu_id": gpu_id,
                "total_gpus": total_gpus,
                "total_items_for_gpu": len(prompt_keys),
                "completed_items": idx + start_idx + 1 if 'idx' in locals() else len(prompt_keys)
            }
            with open(stats_file, "w") as f:
                json.dump(stats, f)
        
        # Calculate and print average time
        # avg_time = total_time / progress_data["current_idx"]
        # print(f"\nAverage inference time per image: {avg_time:.4f} seconds")
        # print(f"Total time for {progress_data['current_idx']} images: {total_time:.4f} seconds")
        
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
                print(f"GPU {gpu_id} max memory usage: {max(max_gpu_memories):.2f} GB / {total_gpu_memory:.2f} GB")
            else:
                print(f"GPU {gpu_id} max memory usage: 0.00 GB (fail to log)")


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
    parser.add_argument('--view_memory', action='store_true', help='View contents of the model memory database')
    parser.add_argument('--min_qwen_image_entries', default=5, type=int, help='Minimum entries in qwen_image DB before guidance is used')
    parser.add_argument('--min_qwen_edit_entries', default=5, type=int, help='Minimum entries in qwen_image_edit DB before guidance is used')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID to use for this process (0-based)')
    parser.add_argument('--total_gpus', default=1, type=int, help='Total number of GPUs being used for parallel processing')

    args = parser.parse_args()
    
    # Handle memory viewing separately
    if args.view_memory:
        view_memory_database()
    else:
        # Create min_entries_for_guidance dictionary from arguments
        min_entries_for_guidance = {
            "qwen_image": args.min_qwen_image_entries,
            "qwen_image_edit": args.min_qwen_edit_entries
        }
        
        main(args.benchmark_name, args.human_in_the_loop, args.model_version, 
             args.use_open_llm, args.open_llm_model, args.open_llm_host, 
             args.open_llm_port, args.calculate_latency, True, min_entries_for_guidance,
             args.gpu_id, args.total_gpus)