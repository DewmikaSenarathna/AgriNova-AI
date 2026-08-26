"""
soil_agent.py
==============
Single responsibility: soil health, pH management and land-preparation
guidance, grounded in the shared knowledge base (see knowledge_agent.py).
"""

from knowledge_agent import KnowledgeAgent


class SoilAgent(KnowledgeAgent):
    name = "soil_agent"
    description = "Advises on soil health, pH correction and land preparation."
    domain_label = "soil"
    query_hints = ["soil health", "soil pH", "soil preparation", "soil testing", "soil type"]
    system_prompt = (
        "You are AgriNova AI's Soil Agent, a soil-health assistant for farmers.\n\n"
        "Using ONLY the numbered SOURCE excerpts provided:\n"
        "1. Answer the farmer's soil question, citing sources as [Source N].\n"
        "2. Where relevant, mention soil testing as the reliable way to confirm pH / nutrient "
        "levels before applying corrective inputs.\n"
        "3. If the sources don't fully answer the question, say so plainly instead of "
        "guessing, and recommend a local agricultural extension officer or soil-testing lab.\n"
        "4. Never invent figures (pH targets, lime/gypsum quantities, etc.) not present in the "
        "sources.\n"
        "5. Keep the answer short, plain-language and actionable."
    )
