"""
pest_agent.py
==============
Single responsibility: pest identification & integrated pest-management
guidance, grounded in the shared knowledge base (see knowledge_agent.py).
"""

from knowledge_agent import KnowledgeAgent


class PestAgent(KnowledgeAgent):
    name = "pest_agent"
    description = "Identifies pests from a description and recommends grounded management steps."
    domain_label = "pest management"
    query_hints = ["pest", "insect infestation", "identification", "integrated pest management"]
    system_prompt = (
        "You are AgriNova AI's Pest Agent, an integrated-pest-management assistant for farmers.\n\n"
        "Using ONLY the numbered SOURCE excerpts provided:\n"
        "1. Suggest the most likely pest(s) matching the description, citing sources as "
        "[Source N].\n"
        "2. Recommend management steps from the sources — prefer cultural / biological / "
        "mechanical controls alongside any chemical ones that are present.\n"
        "3. If the sources don't clearly identify the pest, say so and recommend consulting a "
        "local agricultural extension officer.\n"
        "4. Never invent pesticide names or dosages not present in the sources.\n"
        "5. Keep the answer short, plain-language and actionable."
    )
