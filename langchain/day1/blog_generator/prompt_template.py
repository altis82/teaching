from langchain_core.prompts import PromptTemplate
# Create prompt template
def create_template():
    """Create the prompt template for blog title generation"""
    
    template = """You are an expert SEO content strategist and blog writer.
Your task is to generate {num_titles} creative, engaging, and SEO-optimized blog titles.

Blog Topic: {topic}

Requirements for the titles:
1. Make them attention-catching and compelling
2. Include relevant keywords naturally
3. Keep them between 5-10 words
4. Vary the style (how-to, listicle, question, thought-leadership)
5. Make them suitable for the target audience

Generate exactly {num_titles} titles and provide a brief strategy explanation.

Format your response as JSON with this exact structure:
{{
    "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
    "summary": "Brief explanation of the title strategy used"
}}

Remember: The titles should be clickable, memorable, and optimized for search engines."""

    return PromptTemplate(
        input_variables=["topic", "num_titles"],
        template=template
    )
