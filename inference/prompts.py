"""
Centralized Prompts Module for Op-Schemes.

This module contains all prompts used across different features:
- DSPy Extraction Pipeline
- Image Company Recognition
- Product Mapping

Organized by feature/section for easy maintenance.
"""

import json


# =============================================================================
# DSPY EXTRACTION PROMPTS
# =============================================================================

# ----------------------------------------------------------------------------
# Step 1: Metadata Extraction (VLM)
# ----------------------------------------------------------------------------

METADATA_EXTRACTION_PROMPT = """
Extract metadata for ALL schemes visible in this image.

You are working with scheme docuements of paint industry in India.

For EACH scheme, extract:
1. scheme_name: Official name/title of the scheme
2. timeline: Overall scheme duration/validity period in "DD/MM to DD/MM" format (e.g., "01/08 to 31/10"). No year. For "By DDth Month" format, use "01/MM to DD/MM" (e.g., "By 19th Sept" → "01/09 to 19/09").
3. products: Array of ALL UNIQUE products covered by this scheme. As you are working with paints indeustry products, iphone, fridge, luggage etc are NOT products.

Product extraction guidelines:
1. Examine all tables, headers, and sections in the document
2. If product names are combined with separators (e.g., "+", "&", "/"), split them into individual products
3. Extract the exact product names as they appear
4. List each product only once, even if it appears in multiple locations
5. Be very careful about scheme name, every big bold word is not necessarily the scheme name. Understand the context.

CRITICAL:
Deduplicate products by meaning, not wording.
Treat plural/singular, abbreviations, and formatting variations as the same product.
Merge such variants into ONE entry and keep the expanded clear name.

VERY IMPORTANT:
DO NOT make up data. Only extract what is explicitly present in the table. SKIP any entires where value of reward or volume is not clearly present, such as: "3gm Gold", "tv+20k voucher", "gift item", "special discount", etc.

Format: [{"scheme_name": "...", "timeline": "...", "products": ["...", "..."]}]
"""


def get_dspy_table_markers_prompt(table_count: int, table_previews_str: str, products: list, is_multi_scheme: bool = False, scheme_name: str = None) -> str:
    """Generate prompt for table marker identification."""
    products_str = ', '.join(products)

    prompt = f"""
Analyze this scheme image from paints industry in India and identify ALL structured tables containing reward, discount, or incentive data.

IMPORTANT CONTEXT:
1. OCR has detected {table_count} table(s) in this document
2. You MUST identify markers for ALL {table_count} table(s)
3. Below are previews of what OCR extracted from each table

OCR TABLE PREVIEWS:
{table_previews_str}

Known products in this scheme: {products_str}

SCHEME CONTEXT FOR PRODUCT IDENTIFICATION:
- Analyze the ENTIRE scheme context to identify which products each table belongs to
- Tables may not explicitly mention product names but can be identified through:
  * Table content patterns (e.g., specific reward structures, volume thresholds, or product codes)
  * Table positioning within the scheme (which products typically appear together)
  * Reward/incentive types that are specific to certain products
  * Generic table structures that are standard for particular product categories
  * Scheme-wide patterns that indicate which products are covered

For EACH of the {table_count} TABLES, extract:
1. table_id: A short descriptive identifier based on the table's content
2. section_type: Always "table"
3. marker: A UNIQUE text combination that appears ONLY in this specific table
4. period: Time period in "DD/MM to DD/MM" format (e.g., "01/08 to 31/10"). No year. For "By DDth Month" format, use "01/MM to DD/MM". If not visible, use empty string.
5. products_in_section: Which products from the known product list appear in this table

CRITICAL - You must identify exactly {table_count} tables to match the OCR output.

CRITICAL - Marker Selection Rules:
1. The marker must be UNIQUE - if the same text appears in multiple tables, it's not a good marker
2. Combine multiple pieces of text from the table to create uniqueness (use | as separator)
3. Include enough context to distinguish this table from all others
4. Prefer combining: table title + column header, OR product name + column header
5. Verify the marker would only match this ONE specific table in the entire document

CRITICAL - Product Mapping Rules:
1. For products_in_section, ONLY use products from the known product list provided above
2. Match product names flexibly - handle acronyms, abbreviations, short forms, and variations in formatting
3. Consider singular/plural variations, spacing differences, and punctuation
4. Return the EXACT product names as they appear in the known product list
5. If products are NOT explicitly mentioned in the table, use SCHEME CONTEXT to identify them
6. When products aren't explicit, look for:
   - Generic tables that apply to ALL products (use entire known product list)
   - Tables with specific patterns that match certain product categories
   - Tables positioned in sections that clearly indicate product coverage

NEVER skip a table - you must find markers for ALL {table_count} tables

Return ONLY valid JSON object with table_id as keys. No markdown, no explanations.
Format: {{"table_1": {{"section_type": "table", "marker": "...", "period": "...", "products_in_section": ["...", "..."]}}}}
"""

    if is_multi_scheme and scheme_name:
        prompt += f"\n\nCRITICAL - MULTI-SCHEME FOCUS: Focus only on the section for scheme: '{scheme_name}'. Ignore other schemes in the image."

    return prompt


INBILL_EXTRACTION_INSTRUCTIONS = """
Task:
- For each product that appears in the table, extract per-unit rebate values across all time periods
- Only extract data for products actually present in the table (return empty array for missing products)
- Extract numeric values only (strip currency symbols, units, and other text)
- Do not extract volume or quantity numbers—only per-unit pricing/rebate amounts
- CRITICAL: Format all periods in "DD/MM to DD/MM" format (e.g., "01/08 to 31/10"). No year. For "By DDth Month" format, use "01/MM to DD/MM".
"""


def get_dspy_credit_note_instructions(product: str, period: str) -> str:
    """Generate instructions for credit note data extraction for a specific product."""
    return f"""
You are extracting volume-based reward data for product: "{product}"

Step 1: Check if "{product}" appears in the table (in column headers, row labels, or anywhere)

Step 2: Based on Step 1, follow the appropriate extraction rule:

IF product name NOT found in table:
1. The table is GENERIC (applies to all products)
2. Extract ALL volume-reward pairs
3. Every slab in this table applies to "{product}"

IF product name FOUND in table:
1. The table is PRODUCT-SPECIFIC
2. Extract ONLY pairs from the column/section labeled "{product}"
3. Ignore slabs for other products

Data extraction rules:
1. Each entry must have both volume (threshold) and reward_value (amount)
2. Return numeric values only (strip currency symbols, units like "ltr", separators). Return only when explicit numeric values are present. Ignore slabs where either volume or reward_value is missing or non-numeric.
3. It may happen that there is a numeric reward value but it is not sufficient to give value for that slab. For eg. "tv+20k voucher" - here 20k is numeric but not sufficient to give value for that slab. So ignore such slabs.
4. Handle combined product names flexibly (e.g., "DSI+DSE" matches "DSI")
5. Be precise about the period - it must be accurate. Period presision is very important. Format: "DD/MM to DD/MM" (e.g., "01/08 to 31/10"). No year. For "By DDth Month" format, use "01/MM to DD/MM".

VERY IMPORTANT:
DO NOT make up data. Only extract what is explicitly present in the table. SKIP any entires where value of reward or volume is not clearly present, such as: "3gm Gold", "tv+20k voucher", "gift item", "special discount", etc.
CRITICAL:
Return empty array if:
1. Product-specific table but "{product}" has no data
2.Table has no volume-reward structure
"""


def get_dspy_nontabular_rewards_prompt(product: str, scheme_timeline: str, markdown_without_tables: str) -> str:
    """Generate prompt for non-tabular reward extraction."""
    return f"""
You are analyzing a scheme document from paint industry in India to extract non-tabular reward information.

Product: {product}
Scheme Timeline: {scheme_timeline}

Below is the markdown content with ALL TABLES REMOVED. Extract ONLY reward entries that EXPLICITLY mention "{product}" or very close variants in the reward text itself.

STRICT PRODUCT MATCHING RULES:
- The reward description MUST contain the exact product name "{product}" or clear synonyms/abbreviations
- DO NOT extract rewards based on contextual clues, related terms, or indirect associations
- Only extract if the reward text specifically refers to this product

Look for:
1. Bonus rewards
2. Gift items
3. Additional incentives
4. Special offers

For each entry, extract:
1. target: Target volume/sale amount/percentage (if mentioned)
2. reward_text: Full reward description
3. reward_value: Numeric value (if explicitly mentioned)
4. period: Time period in "DD/MM to DD/MM" format (e.g., "01/08 to 31/10"). No year. For "By DDth Month" format, use "01/MM to DD/MM". If not mentioned, use scheme timeline.
5. SKIP entries without clear numeric reward values

CRITICAL: The period must be accurate. Don't miss any rewards.

Return ONLY valid JSON array:
[
  {{"target": "55%", "reward_text": "Rs 4 per liter bonus on Luxury Emulsions", "reward_value": 4, "period": "01/09 to 19/09"}},
]

Return empty array [] if no reward entries found that explicitly mention this product.
No markdown code blocks, no explanations.

MARKDOWN CONTENT (tables removed):
{markdown_without_tables}
"""


def get_dspy_product_matching_prompt(table_products: list, scheme_products: list) -> str:
    """Generate prompt for matching product names from table to scheme product list."""
    return f"""
Match product names from a data table to a reference product list.

TABLE PRODUCTS:
{json.dumps(table_products, indent=2)}

REFERENCE PRODUCTS:
{json.dumps(scheme_products, indent=2)}

Task: For each table product, find the best matching reference product.

Match ONLY if the core product identity is the same. Accept variations in:
- Abbreviations and full forms
- Singular/plural forms
- Spacing, punctuation, formatting

DO NOT match different products even if they share similar words. Core distinguishing terms must match exactly.
If no reasonable match exists, use null.

Return ONLY valid JSON mapping table_product -> reference_product.
Format: {{"table_product": "reference_product", ...}}

No explanations, only JSON.
"""


# =============================================================================
# IMAGE COMPANY RECOGNITION PROMPTS
# =============================================================================

COMPANY_RECOGNITION_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that extracts and transcribes text content from images. "
    "Provide accurate extraction and mapping results when product names and mappings are supplied."
)

COMPANY_IDENTIFICATION_PROMPT = """
You are analyzing paint industry scheme documents from India.

First, determine if this is a scheme document:
1. A scheme document typically contains: tables, discount/reward structures, product names, volume targets, or rebate information
2. If it lacks these elements, it is likely NOT a scheme document. Be logical, like of "scheme name" is menioned then it is a scheme document.
3. If the word "scheme" is mentioned anywhere and has tables, consider it a scheme document.
4. If NOT a scheme document, respond with exactly: {"company_name": "Not a scheme document"}

If it IS a scheme document, identify the company:
1. Look for company logos, brand names, product names, letterhead, or header text
2. You can ONLY identify: Asian Paints or Berger Paints
3. ONLY identify Asian Paints if you see: "Asian Paints" logo, name, or Asian Paints product names (Apex, Ultima, Royal Play, Royale, etc.)
4. ONLY identify Berger Paints if you see: "Berger Paints" logo, name, or Berger product names (Berger Silk, Berger Recess, etc.)
5. If you see a logo but cannot confirm if it's Asian Paints or Berger Paints, respond: {"company_name": "Uncertain"}
6. If there is NO visible logo, NO company name, and NO identifiable product names in the image, respond: {"company_name": "Uncertain"}
7. DO NOT guess. DO NOT make assumptions. If uncertain, always choose "Uncertain" over guessing.

Respond ONLY with valid JSON. Do NOT include markdown or extra text.
"""

COMPANY_RECOGNITION_DEFAULT_USER_PROMPT = "Please extract and transcribe all visible text from this image. Maintain the original structure and formatting."

COMPANY_RECOGNITION_SHORT_USER_PROMPT = "Please identify product names and the corresponding company using the system prompt mapping. Return JSON only."


# =============================================================================
# PRODUCT MAPPING PROMPTS
# =============================================================================

def get_product_mapping_prompt(product_name: str, category_pairs_text: str) -> str:
    """Generate prompt for matching product name against category pairs."""
    return f"""
You are a paint product expert from India. Given a product name, identify the most relevant product category pairs from the list below.

Product name: "{product_name}"

Available category pairs (Category: Type):
{category_pairs_text}

Instructions:
1. Analyze the product name to understand what type of paint product it is
2. Match it against the most relevant category pairs
3. Return ONLY the relevant pairs that best match this product
4. Consider factors like:
   - Product type (emulsion, primer, etc.)
   - Quality level (luxury, premium, standard)
   - Use case (interior, exterior, etc.)

CRITICAL: Return only the relevant category pairs in JSON array format as shown below. ONLY output when you are certain of the match. If unsure, return an empty array.

Return your answer as a JSON array of objects with "category" and "type" fields.
Example: [{{"category": "", "type": ""}}, {{"category": "", "type": ""}}]

If no good match is found, return an [] JSON array.

IMPORTANT: Return ONLY valid JSON. No additional text or explanation.
"""

PRODUCT_MAPPING_LLM_SYSTEM_PROMPT = "You are a helpful paint product expert."


# =============================================================================
# DotsOCR PROMPTS
# =============================================================================

DOTSOCR_PROMPT_LAYOUT_ALL_EN = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:

    - Picture: For the 'Picture' category, the text field should be omitted.

    - Formula: Format its text as LaTeX.

    - Table: Format its text as HTML.

    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:

    - The output text must be the original text from the image, with no translation.

    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.

"""

DOTSOCR_PROMPT_LAYOUT_ONLY_EN = """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format."""

DOTSOCR_PROMPT_OCR = """Extract the text content from this image."""

DOTSOCR_PROMPT_GROUNDING_OCR = """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).
Bounding Box:
"""

# DotsOCR prompt modes dictionary
DOTSOCR_PROMPT_MODES = {
    "prompt_layout_all_en": DOTSOCR_PROMPT_LAYOUT_ALL_EN,
    "prompt_layout_only_en": DOTSOCR_PROMPT_LAYOUT_ONLY_EN,
    "prompt_ocr": DOTSOCR_PROMPT_OCR,
    "prompt_grounding_ocr": DOTSOCR_PROMPT_GROUNDING_OCR,
}
