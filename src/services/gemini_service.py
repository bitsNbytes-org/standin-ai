import json
import os
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
from config import get_settings
from models import Summary

class GeminiClient:
    
    def __init__(self):
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        
    def generate_narration_and_prompts(self, summary: Summary) -> dict:
        prompt = self._get_narration_prompt(summary)
        
        try:
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "title": types.Schema(type=types.Type.STRING),
                        "slides": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    "slide_number": types.Schema(type=types.Type.INTEGER),
                                    "narration_text": types.Schema(type=types.Type.STRING),
                                },
                                required=["slide_number", "narration_text"]
                            )
                        )
                    },
                    required=["title", "slides"]
                )
            )

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]

            response = self.client.models.generate_content(
                model=self.settings.text_model,
                contents=contents,
                config=generate_content_config
            )

            result = json.loads(response.text)  
            return result

        except Exception as e:
            raise RuntimeError(f"Failed to generate narration and prompts: {e}") from e

    def generate_slide_json(self, narration_text: str) -> dict:
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "heading": types.Schema(type=types.Type.STRING),
                    "bullets": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))
                },
                required=["heading", "bullets"]
            )
        )
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self._get_slide_json_prompt(narration_text))]
            )
        ]
        response = self.client.models.generate_content(
            model=self.settings.text_model,
            contents=contents,
            config=generate_content_config
        )
        return json.loads(response.text)

    def _get_narration_prompt(self, summary: Summary) -> str:
        return f"""Create a structured presentation for: {summary.title}

        Duration: {str(summary.estimated_duration)} minutes
        Content: {summary.content}
        Attendee: {summary.attendee}
        Create knowledge transfer presentation narration based on the provided content. Structure into logical slides (maximum 10-15 slides). Make narration conversational and engaging. Use the attendee's name in the narration to make it more engaging. The conversation should be in natural language and it should not feel like a script. Do not hallucinate any extra content. Only use the content provided in the summary.

        Output format:
        title: Presentation Title  
        slides:
            slide_number: 1
            narration_text: "Welcome to our lesson on..."
        """

    def _get_slide_json_prompt(self, narration_text: str) -> str:
        return f"""Generate a single JSON object representing a slide. Follow these strict rules:

        1. **Content Extraction**
        - Use ONLY the narration text provided: "{narration_text}"
        - Do NOT add, infer, summarize, or hallucinate any extra content
        - Split the narration into:
        - `"heading"`: the main title of the slide (short phrase or sentence)
        - `"bullets"`: an array of 3–6 bullet points, directly derived from the narration

        2. **Output Format**
        - Return only a valid JSON object that SHOULD NOT BE EMPTY OR NULL
        - The JSON must have exactly two keys:
        - `"heading"`: string
        - `"bullets"`: array of strings
        - Example:
        {{
            "heading": "Introduction",
            "bullets": [
            "Welcome to our lesson on...",
            "This is a test slide"
            ]
        }}

        Generate the JSON output strictly following these rules."""


