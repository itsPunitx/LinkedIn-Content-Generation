import json
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


def process_posts(raw_file_path, processed_file_path=None):
    with open(raw_file_path, encoding='utf-8') as file:
        posts = json.load(file)
        enriched_posts = []
        for post in posts:
            metadata = extract_metadata(post['text'])
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)
    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags[tag] for tag in current_tags}
        post['tags'] = list(new_tags)

    with open(processed_file_path, encoding='utf-8', mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)


def extract_metadata(post):
    def clean_text(text):
        return text.encode("utf-8", "ignore").decode("utf-8")

    template = '''
You are given a LinkedIn post. Extract metadata in **raw JSON only**, no explanation or extra text.

Requirements:
1. Output must be valid JSON — no preamble or follow-up text.
2. Return exactly three keys: `line_count`, `language`, `tags`
3. `tags` must be an array with max two text tags
4. Language must be either "English" or "Hinglish"

Post:
{post}
'''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": clean_text(post)})

    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res


import re

def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])

    unique_tags_list = ', '.join(unique_tags)

    template = '''You will be given a list of tags. You need to unify them with the following rules:
1. Merge similar tags to create a concise list. Examples:
   - "Jobseekers", "Job Hunting" → "Job Search"
   - "Motivation", "Inspiration" → "Motivation"
2. Use Title Case for each tag.
3. Return **valid JSON only**, no preamble or extra text.
4. JSON format: {{"original_tag": "unified_tag", ...}}

Tags:
{tags}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": unique_tags_list})
    
    # Extract JSON from the response content
    match = re.search(r'\{[\s\S]*\}', response.content)
    if not match:
        raise OutputParserException("No valid JSON found in model output.")

    clean_json = match.group()
    
    try:
        res = json.loads(clean_json)
    except json.JSONDecodeError as e:
        raise OutputParserException(f"Failed to parse JSON: {e}")

    return res



if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")