#!/usr/bin/env python3

from src.models import Summary
from src.generator import NarrationGenerator


def generate_employee_handbook():
    summary = Summary(
        title="Employee onboarding",
        content="""
        ### Comprehensive Summary of Employee Handbook and Key Policies at KeyValue Software Systems
**High-Level Overview**
The Employee Handbook at KeyValue Software Systems serves as a vital resource, detailing essential employment terms, policies, and benefits. It promotes a culture of responsibility, flexibility, and well-being, ensuring employees are well-informed about their roles and the support available to them.
---
### Key Sections
#### 1. **Terms of Employment**
- Employment terms are outlined in the offer letter.
- KeyValue retains the right to amend these terms.
- Confidentiality of contract details is mandatory.

""",
        estimated_duration=1,
        attendee="John"
    )
    
    generator = NarrationGenerator()
    result = generator.generate(summary, generate_slides=True)
    generator.export_results(result)
    
    return result


def main():
    try:
        generate_employee_handbook()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 