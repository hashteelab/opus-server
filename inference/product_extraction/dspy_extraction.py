"""
Hybrid VLM+LLM Product Extraction Pipeline (TABLES + NON-TABULAR REWARDS)

Clean Strategy:
1. INPUT: Image + Markdown (with HTML table tags)
2. VLM: Acts as "eyes" - identifies table markers/keywords for locating tables
3. REGEX: Cuts out specific tables from markdown using VLM's markers
4. LLM: Extracts structured data from isolated table (no cross-contamination)
5. PYTHON: Finds max slabs, deduplicates, assembles final output

Focus: 
- PRIMARY: Extracts data from structured HTML tables (IN Bill, Power Slabs, Full Month)
- FOR REVIEW: Also extracts non-tabular rewards (text-based, not in tables)

Non-Tabular Rewards (FOR REVIEW):
- VLM identifies text-based reward sections (not in tables)
- Markdown with all tables removed is sent to LLM
- LLM extracts reward data from pure text content
- Kept separate for review purposes

Benefits:
- VLM only does spatial localization (what it's best at)
- Regex cleanly isolates tables (no VLM hallucination)
- LLM sees ONLY relevant content (table-only or text-only)
- Non-tabular extraction uses markdown without tables to avoid confusion
- Much more reliable and maintainable
"""

import json
import re
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
import dspy
from logger import (
    logger, log_header, log_step, log_subsection, log_item,
    log_success, log_error, log_warning, log_info, log_processing,
    log_data_point, log_list_item, log_separator, log_empty_line
)
from inference.product_extraction.vlm_client import VLMClient
from inference import (
    METADATA_EXTRACTION_PROMPT,
    get_dspy_table_markers_prompt,
    INBILL_EXTRACTION_INSTRUCTIONS,
    get_dspy_credit_note_instructions,
    get_dspy_nontabular_rewards_prompt,
    get_dspy_product_matching_prompt,
)
from inference import create_openai_client, create_dspy_lm, LLMProviderEnum, get_llm_config
from openai import OpenAI


# ============================================================================
# Pydantic Models
# ============================================================================

class RewardSlab(BaseModel):
    """Individual reward slab entry"""
    volume: Optional[Union[int, float, str]] = Field(None, description="Target volume")
    sale_amount: Optional[Union[int, float, str]] = Field(None, description="Target sales amount")
    period: Optional[str] = Field(None, description="Time period for this slab")
    reward_text: str = Field(..., description="Original reward description")
    reward_value: Optional[float] = Field(None, description="Monetary value of reward")
    reward_type: str = Field(..., description="Type: credit_note, in_bill_rebate, bonus, etc.")


class ProductOffer(BaseModel):
    """Structured product offer"""
    scheme_name: Optional[str] = Field(None, description="Scheme title")
    timeline: Optional[str] = Field(None, description="Scheme duration")
    product: Optional[str] = Field(None, description="Product name/code")
    reward_slabs: Optional[List[RewardSlab]] = Field(None, description="All reward slabs")


# ============================================================================
# DSPy Models for Structured Extraction
# ============================================================================

class VolumeRewardPair(BaseModel):
    """Single volume-reward pair from table"""
    volume: Union[int, float] = Field(..., description="Volume threshold")
    reward_value: Union[int, float] = Field(..., description="Reward amount")
    period: Optional[str] = Field(None, description="Time period for this slab")


class ExtractCreditNoteData(dspy.Signature):
    """Extract volume-based reward data for a product from HTML table."""
    
    table_html: str = dspy.InputField()
    product_name: str = dspy.InputField()
    period: str = dspy.InputField()
    extraction_instructions: str = dspy.InputField()
    
    volume_reward_pairs: List[VolumeRewardPair] = dspy.OutputField()


class INBillPeriodData(BaseModel):
    """Single period's in-bill rebate data"""
    period: str = Field(..., description="Time period")
    rebate_value: Union[int, float] = Field(..., description="Rebate amount per unit")


class ExtractINBillData(dspy.Signature):
    """Extract in-bill rebate data for products from HTML table."""
    
    table_html: str = dspy.InputField()
    product_names: List[str] = dspy.InputField()
    extraction_instructions: str = dspy.InputField()
    
    product_rebates: Dict[str, List[INBillPeriodData]] = dspy.OutputField()


# ============================================================================
# LLM Client Setup
# ============================================================================

def get_llm_client() -> OpenAI:
    """Get configured LLM client (RunPod)"""
    config = get_llm_config(LLMProviderEnum.runpod)
    client = create_openai_client(config["url"])
    log_success(f"LLM client initialized: {config['url']}", indent=0)
    return client


def get_dspy_lm() -> dspy.LM:
    """Get configured DSPy LM (RunPod) without calling dspy.configure()"""
    config = get_llm_config(LLMProviderEnum.runpod)
    lm = create_dspy_lm(
        api_base=config["url"],
        model_name=config["model"],
        temperature=0.0,  # Use 0 for maximum consistency
        max_tokens=2048
    )
    # NOTE: Don't call dspy.configure() here - it can only be called once per async task
    # Instead, we'll use dspy.context() in the extraction function
    log_success(f"DSPy LM created: {config['url']}, model={config['model']}", indent=0)
    return lm


def call_llm(client: OpenAI, prompt: str, model: str = None) -> str:
    """
    Call LLM with a prompt and return response.
    
    Args:
        client: OpenAI client instance
        prompt: The prompt to send
        model: Model name (default: gets from RunPod config)
        
    Returns:
        LLM response text
    """
    if model is None:
        config = get_llm_config(LLMProviderEnum.runpod)
        model = config["model"]
    
    try:
        logger.debug("Calling LLM...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        log_error(f"LLM call failed: {e}", indent=0)
        raise


# ============================================================================
# Step 1: Metadata Extraction (VLM)
# ============================================================================

def _split_and_deduplicate_products(products: List[str]) -> List[str]:
    """
    Split composite products if at least one of the split components exists as
    a standalone product elsewhere in the input.

    Example:
    ["IWC", "DSI", "DSI+DSE", "Premium Emulsions (Interior + Exterior)"]
    -> ["IWC", "DSI", "DSE", "Premium Emulsions (Interior + Exterior)"]
    """

    def normalize(p: str) -> str:
        return re.sub(r'\s+', ' ', p.strip())

    # 1️⃣ Collect atomic products (those without separators)
    atomic_products = set()
    for product in products:
        if not re.search(r'[+&/]', product):
            atomic_products.add(normalize(product))

    unique_products = []
    seen = set()

    # 2️⃣ Process each product
    for product in products:
        normalized_product = normalize(product)

        # Candidate split
        parts = [normalize(p) for p in re.split(r'[+&/]', product) if p.strip()]

        # Decide whether to split - if at least one part exists as standalone
        should_split = (
            len(parts) > 1 and
            any(part in atomic_products for part in parts)
        )

        final_parts = parts if should_split else [normalized_product]

        for p in final_parts:
            if p not in seen:
                seen.add(p)
                unique_products.append(p)

    return unique_products



def extract_scheme_metadata(vlm_client: VLMClient, image_path: str) -> List[Dict[str, Any]]:
    """
    Extract scheme metadata using VLM.

    Returns list of schemes with name, timeline, and products.
    """
    log_step(1, "EXTRACTING SCHEME METADATA (VLM)")

    prompt = METADATA_EXTRACTION_PROMPT
    
    # List of words to filter out (lowercase)
    filter_words = ["slab", "slabs", "cn", "lt", "lts", "ltr", "litre", "litres"]

    def _filter_products(products):
        return [p for p in products if p.lower() not in filter_words]

    try:
        response = vlm_client.run(prompt, image_path)
        response = _clean_json_response(response)
        schemes = json.loads(response)

        if not isinstance(schemes, list):
            schemes = [schemes]

        # Post-process: filter, then split and deduplicate products
        for scheme in schemes:
            if 'products' in scheme and scheme['products']:
                # First filter unwanted products
                scheme['products'] = _filter_products(scheme['products'])
                # Then split and deduplicate
                scheme['products'] = _split_and_deduplicate_products(scheme['products'])

        log_success(f"Found {len(schemes)} scheme(s)", indent=0)
        log_empty_line()

        for idx, scheme in enumerate(schemes, 1):
            log_data_point(f"Scheme {idx}", scheme.get('scheme_name'), indent=2)
            log_data_point("Timeline", scheme.get('timeline'), indent=4)
            products = scheme.get('products', [])
            log_data_point("Products", f"({len(products)}) {', '.join(products)}", indent=4)
            log_empty_line()

            if not products or len(products) == 0:
                error_msg = f"CRITICAL ERROR: No products found in scheme '{scheme.get('scheme_name')}'."
                log_error(error_msg, indent=2)
                raise ValueError(error_msg)

        # Save output to file
        _save_output_to_file("step1_metadata", schemes)

        return schemes
        
    except Exception as e:
        log_error(f"Metadata extraction failed: {e}", indent=0)
        raise


# ============================================================================
# Step 2: Table Localization (VLM identifies markers)
# ============================================================================

def expand_combined_products_in_markers(table_markers: Dict[str, Dict[str, str]], known_products: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Post-process table markers to expand combined products (e.g., "DSI+DSE" → ["DSI", "DSE"]).

    This catches cases where the VLM's marker contains combined products but products_in_section
    only lists some of them. We scan the marker for combined patterns and expand them.

    Args:
        table_markers: Raw markers from VLM
        known_products: List of all known products from Step 1

    Returns:
        Updated table_markers with expanded products_in_section
    """
    for table_id, info in table_markers.items():
        marker = info.get('marker', '')
        products_in_section = set(info.get('products_in_section', []))

        # Find all combined product patterns in marker (e.g., "DSI+DSE", "IWC&DB2K", "DSE/DSI")
        # Pattern: uppercase letters/numbers separated by +, &, or /
        combined_pattern = r'\b([A-Z][A-Z0-9]*(?:[+&/][A-Z][A-Z0-9]*)+)\b'
        matches = re.findall(combined_pattern, marker)

        for match in matches:
            # Split by separators
            parts = re.split(r'[+&/]', match)

            for part in parts:
                part = part.strip()
                # Check if part matches any known product (case-insensitive)
                for known_prod in known_products:
                    if part.upper() == known_prod.upper():
                        products_in_section.add(known_prod)
                        break

        # Update products_in_section with expanded list
        info['products_in_section'] = list(products_in_section)

    return table_markers


def identify_table_markers(vlm_client: VLMClient, b64_image: str, markdown: str, scheme_name: str, products: List[str], is_multi_scheme: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Use VLM to identify unique keywords/markers for each table.
    Now includes information about tables found in markdown OCR.

    Args:
        vlm_client: VLM client instance
        b64_image: Base64 encoded image
        markdown: Full markdown with HTML tables (for table counting)
        scheme_name: Name of the scheme
        products: List of products from Step 1 metadata extraction

    Returns: Dict with dynamic table IDs as keys
    """
    log_step(2, "IDENTIFYING TABLE MARKERS (VLM)")

    # Extract table count and previews from markdown
    table_info = _extract_table_info_from_markdown(markdown)
    table_count = table_info['count']
    table_previews = table_info['previews']

    log_data_point("Tables found in OCR markdown", table_count, indent=2)
    log_empty_line()

    prompt = get_dspy_table_markers_prompt(
        table_count=table_count,
        table_previews_str=_format_table_previews(table_previews),
        products=products,
        is_multi_scheme=is_multi_scheme,
        scheme_name=scheme_name
    )
    
    try:
        payload = vlm_client._create_payload(prompt, b64_image)
        response_data = vlm_client._send_request(payload)
        response = response_data["choices"][0]["message"]["content"]
        response = _clean_json_response(response)
        markers = json.loads(response)
        
        log_success(f"Identified {len(markers)} table marker(s)", indent=0)
        
        # Validation: Check if marker count matches OCR table count
        if len(markers) < table_count:
            log_warning(f"VLM found {len(markers)} markers but OCR detected {table_count} tables - some tables may be missed", indent=0)
        elif len(markers) > table_count:
            log_warning(f"VLM found {len(markers)} markers but OCR detected only {table_count} tables - VLM may be over-identifying", indent=0)
        
        log_empty_line()
        
        for table_id, info in markers.items():
            log_data_point(f"[{table_id}]", f"Marker: '{info.get('marker')}'", indent=2)
            log_data_point("Period", info.get('period'), indent=4)
            log_data_point("Products", ', '.join(info.get('products_in_section', [])), indent=4)
            log_empty_line()
        
        # Save output to file
        _save_output_to_file("step2_table_markers", markers)
        
        return markers
        
    except Exception as e:
        log_error(f"Table marker identification failed: {e}", indent=0)
        return {}


def _extract_table_info_from_markdown(markdown: str) -> Dict[str, Any]:
    """
    Extract table count and previews from markdown HTML tables.
    
    Returns:
        {
            'count': int,
            'previews': [{'index': 1, 'preview': '...'}, ...]
        }
    """
    import re
    
    # Find all tables
    tables = re.findall(r'<table>.*?</table>', markdown, re.DOTALL | re.IGNORECASE)
    
    previews = []
    for idx, table_html in enumerate(tables, 1):
        # Extract first few rows as preview
        preview = _create_table_preview(table_html, max_rows=3)
        previews.append({
            'index': idx,
            'preview': preview
        })
    
    return {
        'count': len(tables),
        'previews': previews
    }


def _create_table_preview(table_html: str, max_rows: int = 3) -> str:
    """
    Create a text preview of table content (first few rows).
    
    Args:
        table_html: HTML table string
        max_rows: Maximum rows to include in preview
        
    Returns:
        Text preview of table
    """
    import re
    
    # Extract all table rows
    rows = re.findall(r'<tr>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
    
    preview_rows = []
    for row in rows[:max_rows]:
        # Extract cells (th or td)
        cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row, re.DOTALL | re.IGNORECASE)
        # Clean HTML tags and whitespace
        cleaned_cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
        # Join cells with | separator
        preview_rows.append(' | '.join(cleaned_cells))
    
    if len(rows) > max_rows:
        preview_rows.append(f"... ({len(rows) - max_rows} more rows)")
    
    return '\n'.join(preview_rows)


def _format_table_previews(previews: List[Dict]) -> str:
    """
    Format table previews for prompt.
    
    Args:
        previews: List of preview dicts with 'index' and 'preview'
        
    Returns:
        Formatted string for prompt
    """
    formatted = []
    for p in previews:
        formatted.append(f"TABLE {p['index']}:")
        formatted.append(p['preview'])
        formatted.append("")  # Empty line between tables
    
    return '\n'.join(formatted)

# ============================================================================
# Step 2b: Non-Tabular Text Extraction (Remove Tables from Markdown)
# ============================================================================

def remove_tables_from_markdown(markdown: str) -> str:
    """
    Remove all <table>...</table> elements from markdown content.
    
    This is used to isolate non-tabular text content for reward extraction.
    
    Args:
        markdown: Full markdown content
        
    Returns:
        Markdown content with all table elements removed
    """
    # Remove all table elements including content
    markdown_without_tables = re.sub(r'<table>.*?</table>', '', markdown, flags=re.DOTALL | re.IGNORECASE)
    
    table_count_before = markdown.count('<table>')
    table_count_after = markdown_without_tables.count('<table>')
    
    log_step("2b", "REMOVING TABLES FROM MARKDOWN (CONTEXT ENGINEERING)")
    
    log_data_point("Tables removed", f"{table_count_before - table_count_after}", indent=2)
    log_data_point("Remaining text length", f"{len(markdown_without_tables)} characters", indent=2)
    log_empty_line()
    
    # Save the cleaned markdown for debugging
    _save_output_to_file("step2b_markdown_without_tables", {"cleaned_text": markdown_without_tables})
    
    return markdown_without_tables


# ============================================================================
# Step 3: Extract Tables from Markdown (Regex)
# ============================================================================

def extract_table_from_markdown(markdown: str, marker: str) -> Optional[str]:
    """
    Extract an HTML table from markdown using a marker string.
    Handles pipe-separated markers by requiring ALL parts to be present in the table.

    Args:
        markdown: Full markdown content with HTML tables
        marker: Unique text marker (may contain | separators for combined markers)

    Returns:
        Extracted HTML table as string, or None if not found
    """
    # Normalize dashes in both marker and markdown to handle en-dash (–) vs hyphen (-) issues
    def normalize_dashes(text: str) -> str:
        """Replace all types of dashes with regular hyphen for matching"""
        return text.replace('–', '-').replace('—', '-').replace('−', '-')

    # Normalize both strings for comparison
    normalized_markdown = normalize_dashes(markdown)
    normalized_marker = normalize_dashes(marker)

    # Split marker by pipe to get individual marker components
    marker_parts = [part.strip() for part in normalized_marker.split('|')]

    # Strategy 1: Find table containing ALL marker parts
    # This handles the new pipe-separated markers from VLM
    all_tables = re.findall(r'<table>.*?</table>', normalized_markdown, re.DOTALL | re.IGNORECASE)

    for table_html in all_tables:
        table_lower = table_html.lower()
        # Check if ALL marker parts are in this table
        if all(part.lower() in table_lower for part in marker_parts):
            # Return the original (non-normalized) version
            start_pos = markdown.lower().find(table_html.lower())
            if start_pos >= 0:
                end_pos = start_pos + len(table_html)
                result = markdown[start_pos:end_pos]
                logger.debug(f"  ✓ Found table containing ALL marker parts [Strategy 1: all parts match]")
                return result

    # Strategy 2: Exact marker match (for single non-pipe markers)
    if '|' not in marker:
        marker_pattern = re.escape(normalized_marker)
        pattern = rf'{marker_pattern}.*?(<table>.*?</table>)'
        match = re.search(pattern, normalized_markdown, re.DOTALL | re.IGNORECASE)

        if match:
            start_pos = match.start(1)
            end_pos = match.end(1)
            table_html = markdown[start_pos:end_pos]
            logger.debug(f"  ✓ Found table after marker [Strategy 2: exact match]")
            return table_html

    # Strategy 3: Partial match - find table with MOST marker parts (fallback)
    best_table = None
    best_match_count = 0

    for table_html in all_tables:
        table_lower = table_html.lower()
        match_count = sum(1 for part in marker_parts if part.lower() in table_lower)

        if match_count > best_match_count:
            best_match_count = match_count
            best_table = table_html

    if best_table and best_match_count >= len(marker_parts) * 0.5:  # At least 50% of parts match
        start_pos = markdown.lower().find(best_table.lower())
        if start_pos >= 0:
            end_pos = start_pos + len(best_table)
            result = markdown[start_pos:end_pos]
            logger.debug(f"  ✓ Found table with {best_match_count}/{len(marker_parts)} marker parts [Strategy 3: partial match]")
            return result

    log_warning(f"Could not find table with marker '{marker}' (tried 3 strategies)", indent=2)
    return None


def extract_all_tables(markdown: str, table_markers: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Extract all identified tables from markdown.
    
    Returns: {table_id: html_table_string}
    """
    log_step(3, "EXTRACTING TABLES FROM MARKDOWN (REGEX)")
    
    # Count total tables in markdown for debugging
    table_count = markdown.count('<table>')
    log_data_point("Total <table> elements found", table_count, indent=2)
    log_empty_line()
    
    extracted_tables = {}
    
    for table_id, info in table_markers.items():
        marker = info.get('marker')
        log_processing(f"Extracting [{table_id}] using marker: '{marker}'", indent=2)
        
        table_html = extract_table_from_markdown(markdown, marker)
        
        if table_html:
            extracted_tables[table_id] = table_html
            log_success("Extracted successfully", indent=4)
        else:
            log_warning("Failed to extract - marker may not be in a table structure", indent=4)
    
    log_empty_line()
    log_success(f"Extracted {len(extracted_tables)}/{len(table_markers)} table(s)", indent=0)
    
    if len(extracted_tables) < len(table_markers):
        failed = len(table_markers) - len(extracted_tables)
        log_warning(f"{failed} marker(s) did not match any tables (likely text-based content, not table structures)", indent=0)
    
    log_empty_line()
    
    # Save output to file
    _save_output_to_file("step3_extracted_tables", extracted_tables)
    
    return extracted_tables


# ============================================================================
# Step 4: Data Extraction from Tables (LLM)
# ============================================================================

def extract_inbill_data_llm(llm_client: OpenAI, table_html: str, products: List[str]) -> Dict[str, List[Dict]]:
    """
    Extract IN Bill data from HTML table using DSPy.

    Uses DSPy with Pydantic models to ensure structured JSON without extra text.

    Returns: {product_name: [{"period": "...", "rebate_per_unit": ...}]}
    """
    log_processing("Using DSPy to extract IN Bill data...", indent=4)

    instructions = INBILL_EXTRACTION_INSTRUCTIONS
    
    try:
        # Create DSPy predictor
        extract = dspy.Predict(ExtractINBillData)
        
        # Execute extraction
        result = extract(
            table_html=table_html,
            product_names=products,
            extraction_instructions=instructions
        )
        
        # Convert Pydantic models to dicts
        data = {}
        if hasattr(result, 'product_rebates') and result.product_rebates:
            for product, rebate_list in result.product_rebates.items():
                data[product] = []
                for rebate in rebate_list:
                    if isinstance(rebate, INBillPeriodData):
                        data[product].append({
                            "period": rebate.period,
                            "rebate_per_unit": rebate.rebate_value
                        })
                    elif isinstance(rebate, dict):
                        # Handle case where DSPy returns dict
                        data[product].append({
                            "period": rebate.get("period"),
                            "rebate_per_unit": rebate.get("rebate_value")
                        })
        
        # Log what was extracted
        for product, entries in data.items():
            if entries:
                log_data_point(product, f"{len(entries)} period(s)", indent=6)
            else:
                log_data_point(product, "Not in table", indent=6)
        
        return data
        
    except Exception as e:
        log_error(f"IN Bill extraction failed: {e}", indent=4)
        return {product: [] for product in products}


def extract_credit_note_data_llm(llm_client: OpenAI, table_html: str, product: str, table_id: str, period: str) -> List[Dict]:
    """
    Extract Credit Note data for a SINGLE product from HTML table using DSPy.

    This is called once per product per table.
    Uses DSPy with Pydantic models to ensure structured JSON output.

    Returns: [{"volume": ..., "reward_value": ...}, ...]
    """
    log_processing(f"Using DSPy to extract CN data for {product} from [{table_id}]...", indent=4)

    instructions = get_dspy_credit_note_instructions(product, period)
    
    try:
        # Create DSPy predictor
        extract = dspy.Predict(ExtractCreditNoteData)
        
        # Execute extraction
        result = extract(
            table_html=table_html,
            product_name=product,
            period=period,
            extraction_instructions=instructions
        )
        
        # Convert Pydantic models to dicts
        data = []
        if hasattr(result, 'volume_reward_pairs') and result.volume_reward_pairs:
            for pair in result.volume_reward_pairs:
                if isinstance(pair, VolumeRewardPair):
                    data.append({
                        "volume": pair.volume,
                        "reward_value": pair.reward_value,
                        "period": pair.period or period
                    })
                elif isinstance(pair, dict):
                    # Handle case where DSPy returns dict
                    data.append({
                        "volume": pair.get("volume"),
                        "reward_value": pair.get("reward_value"),
                        "period": pair.get("period", period)
                    })
        
        log_success(f"Extracted {len(data)} row(s) for {product}", indent=6)
        
        # Log the volumes extracted for debugging
        if data:
            volumes = [row.get('volume') for row in data if row.get('volume') is not None]
            logger.debug(f"        Volumes extracted: {volumes}")
        
        return data
        
    except Exception as e:
        log_error(f"CN extraction failed for {product}: {e}", indent=4)
        logger.debug(f"        Error details: {type(e).__name__}: {str(e)}")
        return []


# ============================================================================
# NON-TABULAR REWARDS EXTRACTION - LLM Direct Approach
# ============================================================================

def extract_nontabular_rewards_llm(llm_client: OpenAI, markdown_without_tables: str, product: str, scheme_timeline: str) -> List[Dict]:
    """
    Extract non-tabular reward data for a SINGLE product from markdown (with tables removed) using LLM.

    This uses CONTEXT ENGINEERING: tables are removed, LLM sees only remaining text.
    No VLM involvement - LLM directly processes the cleaned markdown.

    Args:
        llm_client: LLM client instance
        markdown_without_tables: Markdown content with all <table> elements removed
        product: Product name to extract rewards for
        scheme_timeline: Overall scheme timeline for context

    Returns: [{"target": "55%", "reward_text": "...", "reward_value": 4, "period": "..."}]
    """
    log_processing(f"Using LLM to extract non-tabular rewards for {product}...", indent=4)

    prompt = get_dspy_nontabular_rewards_prompt(product, scheme_timeline, markdown_without_tables)
    
    try:
        response = call_llm(llm_client, prompt)
        response = _clean_json_response(response)
        data = json.loads(response)
        
        if not isinstance(data, list):
            data = []
        
        log_success(f"Extracted {len(data)} non-tabular reward entry(ies) for {product}", indent=6)
        
        return data
        
    except Exception as e:
        log_error(f"Non-tabular reward extraction failed for {product}: {e}", indent=4)
        return []


# ============================================================================
# Step 5: Process and Assemble
# ============================================================================

def process_inbill_data(rows: List[Dict[str, Any]], period: str) -> List[RewardSlab]:
    """
    Convert IN Bill rows to RewardSlab objects.
    """
    slabs = []
    
    for row in rows:
        rebate = row.get('rebate_per_unit')
        row_period = row.get('period', period)
        
        if rebate is not None and rebate > 0:
            slab = RewardSlab(
                volume=None,
                sale_amount=None,
                period=row_period,
                reward_text=f"In-bill rebate: Rs {rebate}/unit",
                reward_value=float(rebate),
                reward_type="in_bill_rebate"
            )
            slabs.append(slab)
    
    return slabs


def process_credit_note_data(rows: List[Dict[str, Any]], period: str) -> List[RewardSlab]:
    """
    Process credit note rows and create RewardSlab objects.
    
    Returns list of all RewardSlabs (finding max is commented out for now).
    """
    if not rows:
        log_warning("No CN rows provided", indent=8)
        return []
    
    # Filter valid numeric rows
    valid_rows = []
    invalid_count = 0
    for row in rows:
        vol = row.get('volume')
        reward = row.get('reward_value')
        row_period = row.get('period', period)
        
        # Convert to float, skip if not numeric
        try:
            vol_num = float(vol) if vol is not None else None
            reward_num = float(reward) if reward is not None else None
            
            if vol_num is not None and reward_num is not None and vol_num > 0 and reward_num > 0:
                valid_rows.append({
                    'volume': vol_num,
                    'reward_value': reward_num,
                    'period': row_period
                })
            else:
                invalid_count += 1
                if vol_num is None:
                    log_warning(f"Skipping row with missing volume (reward: {reward_num})", indent=8)
        except (ValueError, TypeError):
            invalid_count += 1
            continue
    
    if not valid_rows:
        log_warning(f"No valid CN rows (skipped {invalid_count} invalid rows)", indent=8)
        return []
    
    log_success(f"Found {len(valid_rows)} valid volume→reward pairs", indent=8)
    
    slabs = []
    for row in valid_rows:
        slab = RewardSlab(
            volume=row['volume'],
            sale_amount=None,
            period=row['period'],
            reward_text=f"Credit Note: Rs {row['reward_value']}",
            reward_value=row['reward_value'],
            reward_type="credit_note"
        )
        slabs.append(slab)
    
    log_info(f"→ Returning all {len(slabs)} slab(s)", indent=8)
    
    return slabs


# ============================================================================
# NON-TABULAR REWARDS PROCESSING - FOR REVIEW
# ============================================================================

def process_nontabular_rewards(rows: List[Dict[str, Any]], period: str) -> List[RewardSlab]:
    """
    Convert non-tabular reward rows to RewardSlab objects.
    
    Returns list of RewardSlab objects for all non-tabular reward entries.
    """
    slabs = []
    
    for row in rows:
        target = row.get('target')
        reward_text = row.get('reward_text', 'Non-tabular reward')
        reward_value = row.get('reward_value')
        row_period = row.get('period', period)
        
        # Parse target to volume or sale_amount
        volume = None
        sale_amount = None
        if target:
            # If target contains percentage, it's likely a sales amount condition
            if '%' in str(target):
                sale_amount = target
            else:
                # Try to extract numeric value
                match = re.search(r'(\d+(?:\.\d+)?)', str(target))
                if match:
                    volume = float(match.group(1))
        
        slab = RewardSlab(
            volume=volume,
            sale_amount=sale_amount,
            period=row_period,
            reward_text=reward_text,
            reward_value=float(reward_value) if reward_value else None,
            reward_type="nontabular_reward"
        )
        slabs.append(slab)
    
    return slabs


def deduplicate_slabs(slabs: List[RewardSlab]) -> List[RewardSlab]:
    """
    Remove duplicate slabs.
    
    For IN Bill: Keep unique (period, reward_value) combinations
    For CN: Keep unique (volume, reward_value) combinations, prefer longer periods
    For Non-tabular: Keep all (they are unique by design)
    """
    # Separate by type
    inbill_slabs = [s for s in slabs if s.reward_type == "in_bill_rebate"]
    cn_slabs = [s for s in slabs if s.reward_type == "credit_note"]
    nontabular_slabs = [s for s in slabs if s.reward_type == "nontabular_reward"]
    
    # Deduplicate IN Bill
    inbill_seen = set()
    deduped_inbill = []
    for slab in inbill_slabs:
        key = (slab.period, slab.reward_value)
        if key not in inbill_seen:
            inbill_seen.add(key)
            deduped_inbill.append(slab)
    
    if len(inbill_slabs) > len(deduped_inbill):
        log_info(f"→ Deduped IN Bill: Removed {len(inbill_slabs) - len(deduped_inbill)} duplicate(s)", indent=6)
    
    # Deduplicate CN
    cn_groups = {}
    for slab in cn_slabs:
        key = (slab.volume, slab.reward_value)
        if key not in cn_groups:
            cn_groups[key] = []
        cn_groups[key].append(slab)
    
    deduped_cn = []
    for key, group_slabs in cn_groups.items():
        if len(group_slabs) == 1:
            deduped_cn.append(group_slabs[0])
        else:
            # Keep the one with longest period
            longest_slab = max(group_slabs, key=lambda s: len(s.period or ""))
            deduped_cn.append(longest_slab)
            
            if len(group_slabs) > 1:
                log_info(f"→ Deduped CN: Removed {len(group_slabs)-1} duplicate(s)", indent=6)
    
    # Non-tabular rewards - keep all (no deduplication needed)
    return deduped_inbill + deduped_cn + nontabular_slabs


def match_table_products_to_scheme_products_llm(
    llm_client: OpenAI,
    table_products: List[str],
    scheme_products: List[str]
) -> Dict[str, str]:
    """
    Use LLM to intelligently match product names from table to scheme product list.

    Args:
        table_products: Product names as they appear in the table
        scheme_products: Product names from scheme metadata (VLM extraction)

    Returns:
        Dictionary mapping table_product -> scheme_product
        Example: {"Premium Emulsion (Int + Ext)": "Premium Emulsions (Interior + Exterior)"}
    """
    if not table_products or not scheme_products:
        return {}

    prompt = get_dspy_product_matching_prompt(table_products, scheme_products)
    
    try:
        response = call_llm(llm_client, prompt)
        response = _clean_json_response(response)
        mapping = json.loads(response)
        
        # Filter out null mappings
        return {k: v for k, v in mapping.items() if v is not None}
        
    except Exception as e:
        log_warning(f"Product name matching failed: {e}. Using fallback matching.", indent=0)
        # Fallback: simple case-insensitive substring matching
        mapping = {}
        for table_prod in table_products:
            for scheme_prod in scheme_products:
                if (table_prod.lower() in scheme_prod.lower() or 
                    scheme_prod.lower() in table_prod.lower()):
                    mapping[table_prod] = scheme_prod
                    break
        return mapping


def assemble_product_offers(
    vlm_client: VLMClient,
    llm_client: OpenAI,
    b64_image: str,
    scheme_metadata: Dict[str, Any],
    table_markers: Dict[str, Dict[str, str]],
    extracted_tables: Dict[str, str],
    markdown_without_tables: str
) -> List[ProductOffer]:
    """
    Assemble product offers by extracting data for each product from relevant tables and text.
    
    Args:
        markdown_without_tables: Markdown content with all tables removed (for non-tabular extraction)
    """
    log_step(4, "ASSEMBLING PRODUCT OFFERS (LLM + PYTHON)")
    
    scheme_name = scheme_metadata.get('scheme_name')
    timeline = scheme_metadata.get('timeline')
    products = scheme_metadata.get('products', [])
    
    log_data_point("Scheme", scheme_name, indent=2)
    log_data_point("Products", ', '.join(products), indent=2)
    log_empty_line()
    
    offers = []
    
    # Separate tables by type: IN Bill tables vs other reward tables
    inbill_tables = {}
    reward_tables = {}
    
    for table_id, table_html in extracted_tables.items():
        # Detect if it's an IN Bill table by checking content
        marker_lower = table_markers.get(table_id, {}).get("marker", "").lower()
        table_id_lower = table_id.lower()
        table_lower = table_html.lower()
        
        # IN Bill indicators:
        # 1. Marker/table_id contains "in bill", "in-bill", "inbill", "rebate"
        # 2. Table HTML contains "in bill" text (search within table content)
        # 3. Table HTML contains per-unit pricing like "rs/ltr", "rs/lit", "rs/kg"
        has_inbill_name = (
            any(x in marker_lower or x in table_id_lower for x in ["in bill", "in-bill", "inbill", "rebate"]) or
            # Search within table content for "in bill" patterns
            any(x in table_lower for x in ["in bill", "in-bill", "inbill"])
        )
        has_per_unit_pricing = any(x in table_lower for x in ["rs/ltr", "rs/lit", "rs/kg", "rs/unit", "/ltr", "/lit", "/kg"])
        
        # Credit Note indicators: "vol", "volume", "cn", "credit note"
        has_volume_cn = any(x in marker_lower or x in table_id_lower or x in table_lower for x in [" vol", "volume", " cn", "credit note", "power slab"])
        
        # It's an IN Bill table if it has IN Bill naming OR per-unit pricing, AND NOT volume/CN indicators
        is_inbill = (has_inbill_name or has_per_unit_pricing) and not has_volume_cn
        
        if is_inbill:
            inbill_tables[table_id] = table_html
            log_separator("-")
            log_item(f"Detected IN Bill table: {table_id}", indent=2)
            logger.debug(f"    Marker: '{marker_lower}', has_inbill_name={has_inbill_name}, has_volume_cn={has_volume_cn}")
        else:
            reward_tables[table_id] = table_html
            log_separator("-")
            log_item(f"Detected reward table (CN/Volume): {table_id}", indent=2)
            logger.debug(f"    Marker: '{marker_lower}', has_inbill_name={has_inbill_name}, has_volume_cn={has_volume_cn}")
    
    # Extract IN Bill data from all IN Bill tables
    inbill_data = {}
    for table_id, table_html in inbill_tables.items():
        log_separator("-")
        log_subsection(f"Processing IN Bill table: {table_id}")
        table_inbill = extract_inbill_data_llm(llm_client, table_html, products)
        
        # Check if ALL products have empty data (meaning no specific products mentioned)
        all_empty = all(not data for data in table_inbill.values())
        
        if all_empty:
            # No specific products mentioned - rebate applies to ALL products
            log_warning("No specific products found in table - applying to ALL products", indent=4)
            
            # Extract rebate data without product filtering (using empty product name)
            log_processing("Re-extracting data for all products...", indent=4)
            generic_table_inbill = extract_inbill_data_llm(llm_client, table_html, [""])
            
            # If we got data, apply it to all scheme products
            if generic_table_inbill.get("") or generic_table_inbill.get(next(iter(generic_table_inbill), "")):
                rebate_data = generic_table_inbill.get("") or generic_table_inbill.get(next(iter(generic_table_inbill), ""))
                if rebate_data:
                    log_success(f"Applying rebate to all {len(products)} product(s)", indent=4)
                    for scheme_prod in products:
                        if scheme_prod not in inbill_data:
                            inbill_data[scheme_prod] = []
                        inbill_data[scheme_prod].extend(rebate_data)
                        log_data_point("Applied to", scheme_prod, indent=6)
        else:
            # Get table products and create mapping to scheme products
            table_products = list(table_inbill.keys())
            if table_products:
                log_processing("Matching table products to scheme products...", indent=4)
                product_mapping = match_table_products_to_scheme_products_llm(
                    llm_client, table_products, products
                )
                log_success(f"Matched {len(product_mapping)} product(s)", indent=4)
                
                # Remap products using the mapping
                for table_prod, scheme_prod in product_mapping.items():
                    if table_prod in table_inbill and table_inbill[table_prod]:
                        if scheme_prod not in inbill_data:
                            inbill_data[scheme_prod] = []
                        inbill_data[scheme_prod].extend(table_inbill[table_prod])
                        log_data_point("Mapping", f"'{table_prod}' → '{scheme_prod}'", indent=6)
    
    # Process each product
    for product in products:
        log_separator("=")
        log_subsection(f"Product: {product}")
        all_slabs = []
        
        # Add IN Bill slabs
        if product in inbill_data and inbill_data[product]:
            # Try to get period from first IN Bill table
            period = next((table_markers.get(tid, {}).get("period", "") for tid in inbill_tables), "")
            inbill_slabs = process_inbill_data(inbill_data[product], period)
            all_slabs.extend(inbill_slabs)
            log_data_point("IN Bill", f"{len(inbill_slabs)} slab(s)", indent=4)
        
        # Process all reward tables dynamically
        for table_id, table_html in reward_tables.items():
            # Check if product is in this table using LLM-based matching
            products_in_section = table_markers.get(table_id, {}).get("products_in_section", [])
            
            if not products_in_section:
                log_data_point(table_id, "No products in section, skipping", indent=4)
                continue
            
            # Use LLM to match current product to table products
            product_mapping = match_table_products_to_scheme_products_llm(
                llm_client, products_in_section, [product]
            )
            
            # Check if any table product maps to our current product
            matched_table_product = None
            for table_prod, scheme_prod in product_mapping.items():
                if scheme_prod == product:
                    matched_table_product = table_prod
                    break
            
            if not matched_table_product:
                log_data_point(table_id, "Product not in table, skipping", indent=4)
                continue
            
            log_data_point(table_id, f"Matched '{product}' → '{matched_table_product}'", indent=4)
            period = table_markers.get(table_id, {}).get("period", "")
            
            # Use the matched table product name for extraction
            cn_rows = extract_credit_note_data_llm(llm_client, table_html, matched_table_product, table_id, period)
            cn_slabs = process_credit_note_data(cn_rows, period)
            
            if cn_slabs:
                all_slabs.extend(cn_slabs)
                log_data_point(table_id, f"{len(cn_slabs)} slab(s)", indent=4)
            else:
                log_data_point(table_id, "No data", indent=4)
        
        # Process non-tabular rewards using LLM (direct extraction from cleaned markdown)
        log_separator("-")
        log_subsection("Processing non-tabular rewards (LLM direct)")
        reward_rows = extract_nontabular_rewards_llm(
            llm_client, markdown_without_tables, product, timeline
        )
        
        if reward_rows:
            reward_slabs = process_nontabular_rewards(reward_rows, timeline)
            if reward_slabs:
                all_slabs.extend(reward_slabs)
                log_success(f"Extracted {len(reward_slabs)} non-tabular reward(s)", indent=6)
        
        if not all_slabs:
            log_info(f"→ No slabs found, skipping {product}", indent=4)
            continue
        
        # Deduplicate
        all_slabs = deduplicate_slabs(all_slabs)
        
        # Create offer
        offer = ProductOffer(
            scheme_name=scheme_name,
            timeline=timeline,
            product=product,
            reward_slabs=all_slabs
        )
        
        offers.append(offer)
        log_success(f"Created offer with {len(all_slabs)} slab(s)", indent=4)
    
    log_empty_line()
    log_separator("=")
    log_success(f"Assembled {len(offers)} product offer(s)", indent=0)
    log_empty_line()
    
    # Save output to file (convert to dicts for JSON serialization)
    offers_as_dicts = [offer.model_dump() for offer in offers]
    _save_output_to_file("step4_assembled_offers", offers_as_dicts)
    
    return offers


# ============================================================================
# Main Pipeline
# ============================================================================

def extract_products_hybrid(image_path: str, markdown: str) -> Dict[str, Any]:
    """
    Run complete hybrid VLM+LLM extraction pipeline.
    
    Args:
        image_path: Path to scheme image
        markdown: Full markdown content with HTML tables
        
    Returns:
        Dict with product_offers list
    """
    log_empty_line()
    log_header("HYBRID VLM+LLM EXTRACTION PIPELINE", char="=")
    
    try:
        # Initialize clients
        vlm_client = VLMClient()
        llm_client = get_llm_client()
        dspy_lm = get_dspy_lm()  # Initialize DSPy LM (without global configure)
        
        # Use dspy.context() to set the LM for this extraction context only
        # This allows multiple async tasks to use DSPy without conflicts
        with dspy.context(lm=dspy_lm):
            # Step 1: Extract metadata (VLM)
            schemes_metadata = extract_scheme_metadata(vlm_client, image_path)
            
            is_multi_scheme = len(schemes_metadata) > 1
            if is_multi_scheme:
                log_warning(f"Multiple schemes detected ({len(schemes_metadata)}), processing each separately", indent=0)
            
            # Encode image for subsequent VLM calls
            log_separator("-")
            log_processing("Encoding image...", indent=0)
            b64_image = vlm_client._encode_image_to_base64(image_path)
            log_success("Image encoded", indent=0)
            log_empty_line()
            
            all_offers = []
            for scheme_metadata in schemes_metadata:
                log_separator("=")
                log_subsection(f"Processing scheme: {scheme_metadata.get('scheme_name')}")
                
                # Split markdown by scheme name (for multi-scheme support)
                scheme_markdown = markdown
                if is_multi_scheme:
                    scheme_name = scheme_metadata.get('scheme_name')
                    if scheme_name:
                        # Simple splitting: find scheme name positions in order and split between them
                        all_scheme_names = [s.get('scheme_name') for s in schemes_metadata if s.get('scheme_name')]
                        current_idx = all_scheme_names.index(scheme_name)
                        
                        # Debug: log what we're looking for
                        log_info(f"Looking for scheme '{scheme_name}' in markdown (index {current_idx})", indent=4)
                        log_info(f"All schemes in order: {all_scheme_names}", indent=4)
                        
                        # Find positions of all schemes in the markdown
                        scheme_positions = {}
                        for name in all_scheme_names:
                            match = re.search(rf'{re.escape(name)}', markdown, re.IGNORECASE)
                            if match:
                                scheme_positions[name] = match.start()
                                log_info(f"Scheme '{name}' found at position {match.start()}", indent=4)
                            else:
                                log_warning(f"Scheme '{name}' not found in markdown", indent=4)
                        
                        if scheme_name in scheme_positions:
                            start_pos = scheme_positions[scheme_name]
                            
                            # End at next scheme position or document end
                            if current_idx < len(all_scheme_names) - 1:
                                next_scheme = all_scheme_names[current_idx + 1]
                                if next_scheme in scheme_positions:
                                    end_pos = scheme_positions[next_scheme]
                                    log_info(f"Ending at next scheme '{next_scheme}' at position {end_pos}", indent=4)
                                else:
                                    end_pos = len(markdown)
                                    log_warning(f"Next scheme '{next_scheme}' position not found, using document end", indent=4)
                            else:
                                end_pos = len(markdown)
                                log_info("Last scheme, using document end", indent=4)
                            
                            scheme_markdown = markdown[start_pos:end_pos].strip()
                            log_success(f"Split markdown for scheme '{scheme_name}' ({len(scheme_markdown)} chars)", indent=2)
                            
                            # Debug: Save split markdown with generic name
                            _save_output_to_file(f"scheme_{current_idx + 1}_markdown", scheme_markdown)
                        else:
                            log_warning(f"Could not find scheme '{scheme_name}' in markdown, using full markdown", indent=2)
                            scheme_markdown = markdown
                            # Debug: Save full markdown as fallback with generic name
                            _save_output_to_file(f"scheme_{current_idx + 1}_markdown_fallback", scheme_markdown)
                            scheme_markdown = markdown
                
                # Step 2: Identify table markers (VLM) - pass products from Step 1 and multi-scheme flag
                products = scheme_metadata.get('products', [])
                table_markers = identify_table_markers(vlm_client, b64_image, scheme_markdown, scheme_metadata['scheme_name'], products, is_multi_scheme)
                
                if not table_markers:
                    log_warning(f"No table markers for scheme '{scheme_metadata.get('scheme_name')}', skipping", indent=2)
                    continue
                
                # Step 2 Post-processing: Expand combined products in markers
                log_processing("Post-processing: Expanding combined products in markers...", indent=2)
                table_markers = expand_combined_products_in_markers(table_markers, products)
                log_success("Products expanded in markers", indent=2)
                log_empty_line()
                
                # Log updated products for verification
                for table_id, info in table_markers.items():
                    log_data_point(f"[{table_id}] Updated Products", ', '.join(info.get('products_in_section', [])), indent=4)
                log_empty_line()
                
                # Step 2b: Remove tables from markdown (Context Engineering)
                markdown_without_tables = remove_tables_from_markdown(scheme_markdown)
                
                # Step 3: Extract tables from markdown (Regex)
                extracted_tables = extract_all_tables(scheme_markdown, table_markers)
                
                if not extracted_tables:
                    log_warning(f"No tables extracted for scheme '{scheme_metadata.get('scheme_name')}'. VLM may have identified text-based sections instead of tables.", indent=2)
                    log_warning("This is expected if the scheme only has reward text descriptions without structured tables.", indent=2)
                    # Continue to next scheme instead of raising error
                
                # Step 4: Assemble offers (LLM + Python) - including non-tabular rewards
                offers = assemble_product_offers(
                    vlm_client, llm_client, b64_image, scheme_metadata, 
                    table_markers, extracted_tables, markdown_without_tables
                )
                
                all_offers.extend(offers)
        
        log_empty_line()
        log_header("EXTRACTION COMPLETE", char="=")
        
        # Convert to dicts
        offers_as_dicts = [offer.model_dump() for offer in all_offers]
        
        final_output = {
            "product_offers": offers_as_dicts
        }
        
        # Save final output to file
        _save_output_to_file("step5_final_output", final_output)
        
        return final_output
        
    except Exception as e:
        log_error(f"Pipeline failed: {e}", indent=0)
        logger.error(f"Full traceback:", exc_info=True)
        return {
            "product_offers": []
        }


# ============================================================================
# Utilities
# ============================================================================


def _save_output_to_file(stage_name: str, data: Any, output_dir: str = "output_logs") -> None:
    """
    Save stage output to a text file (overrides previous file for same stage).
    
    Args:
        stage_name: Name of the processing stage (e.g., "step1_metadata")
        data: Data to save (will be JSON-formatted if dict/list)
        output_dir: Directory to save outputs (default: "output_logs")
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename without timestamp (so it overrides each time)
    filename = f"{stage_name}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Format data
    if isinstance(data, (dict, list)):
        content = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        content = str(data)
    
    # Write to file (overrides existing file)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== {stage_name.upper()} ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(content)
    
    log_success(f"Saved output to: {filepath}", indent=0)


def _clean_json_response(response: str) -> str:
    """Clean LLM/VLM response to extract pure JSON."""
    response = response.strip()
    
    # Remove markdown code blocks
    if response.startswith("```"):
        lines = response.split("\n")
        # Find closing ``` 
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip() == "```":
                response = "\n".join(lines[1:i])
                break
        response = response.strip()
    
    # Remove json tag if present
    if response.startswith("json"):
        response = response[4:].strip()
    
    return response