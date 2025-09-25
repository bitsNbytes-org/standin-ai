from typing import List
from models import Summary, Slide, NarrationResult
from .gemini_service import GeminiClient
import json
import os

class NarrationGenerator:
    
    def __init__(self):
        self.client = GeminiClient()
    
    def generate(self, summary: Summary, generate_slides: bool = True) -> NarrationResult:
        structured_data = self.client.generate_narration_and_prompts(summary)
        
        slides = []
        
        for slide_data in structured_data["slides"]:
            slide = Slide(
                slide_number=slide_data["slide_number"],
                narration_text=slide_data["narration_text"],
            )
            slides.append(slide)                                
        
        if generate_slides:
            self._generate_slides(slides)
        
        result = NarrationResult(
            title=structured_data.get("title", summary.title),
            slides=slides
        )
        
        return result

    def _generate_slides(self, slides: List[Slide]) -> None:
        for slide in slides:
            try:
                slide_json = self.client.generate_slide_json(slide.narration_text)
                slide.slide_json = slide_json
            except Exception as e:
                slide.slide_json = {}

    def export_results(self, result: NarrationResult, output_dir: str = "output") -> dict:
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, "presentation_data.json")
        

        metadata = result.model_dump()
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return {
            "metadata_path": metadata_path
        } 