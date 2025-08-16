import os
import logging
import csv
import io
from mcp.server.fastmcp import FastMCP
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('google_ads_mcp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastMCP("Google Ads Keyword Planner")

def validate_credentials():
    """Validate that all required environment variables are set"""
    required_vars = [
        "DEVELOPER_TOKEN", 
        "CLIENT_ID", 
        "CLIENT_SECRET", 
        "REFRESH_TOKEN",
        "LOGIN_CUSTOMER_ID"
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        error_msg = f"Missing required environment variables: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("All required credentials are configured")

# Validate credentials on startup
try:
    validate_credentials()
except ValueError as e:
    logger.critical(f"Startup failed: {e}")
    exit(1)

CREDENTIALS = {
    "developer_token": os.getenv("DEVELOPER_TOKEN"),
    "client_id": os.getenv("CLIENT_ID"),
    "client_secret": os.getenv("CLIENT_SECRET"),
    "refresh_token": os.getenv("REFRESH_TOKEN"),
    "login_customer_id": os.getenv("LOGIN_CUSTOMER_ID"),
    "use_proto_plus": True,
    "token_uri": "https://oauth2.googleapis.com/token",
}
# Use LOGIN_CUSTOMER_ID consistently throughout
DEFAULT_CUSTOMER_ID = os.getenv("LOGIN_CUSTOMER_ID")

LOCATION_OPTIONS = {
    # Countries
    "United States": "2840",
    "India": "2356",
    "Canada": "2124",
    "United Kingdom": "2826",
    "Australia": "2036",
    "Germany": "2276",
    "France": "2250",
    "Japan": "2392",
    "Brazil": "2076",
    "Mexico": "2484",
    # US Cities
    "New York, NY": "1023191",
    "Los Angeles, CA": "1012429",
    # UK Cities
    "London, UK": "1006879",
    # Indian Cities
    "Mumbai, India": "1007751",
    "Delhi, India": "1007789",
    "Bangalore, India": "1007785",
    "Chennai, India": "1007786",
    "Kolkata, India": "1007748",
    "Hyderabad, India": "1007790",
    "Pune, India": "1007744",
    # Other Cities
    "Toronto, Canada": "1009714",
    "Sydney, Australia": "1008959"
}

def sanitize_keywords(keywords: str) -> List[str]:
    """Sanitize and validate keyword input"""
    if not keywords or not keywords.strip():
        return []
    
    # Split, strip whitespace, limit length, and remove empty strings
    sanitized = []
    for keyword in keywords.split(","):
        clean_keyword = keyword.strip()[:100]  # Limit to 100 chars
        if clean_keyword and len(clean_keyword) >= 2:  # Minimum 2 chars
            sanitized.append(clean_keyword)
    
    return sanitized[:50]  # Limit to 50 keywords max

def validate_locations(locations: List[str]) -> tuple[List[str], str]:
    """Validate location names and return valid location IDs"""
    if not locations:
        return [], "Error: At least one location must be specified"
    
    valid_location_ids = []
    invalid_locations = []
    
    for loc_name in locations:
        if loc_name in LOCATION_OPTIONS:
            valid_location_ids.append(LOCATION_OPTIONS[loc_name])
        else:
            invalid_locations.append(loc_name)
    
    if invalid_locations:
        available_locations = ", ".join(list(LOCATION_OPTIONS.keys())[:10]) + "..."
        return [], f"Error: Invalid locations: {invalid_locations}. Available locations include: {available_locations}"
    
    return valid_location_ids, ""

def validate_result_limit(result_limit: int) -> tuple[int, str]:
    """Validate result limit parameter"""
    if result_limit < 1 or result_limit > 100:
        return 100, "Warning: Result limit must be between 1 and 100. Using default value of 100."
    return result_limit, ""

def extract_concept_annotation(idea):
    """Extract actual concept names (like brand names, product types) from KEYWORD_CONCEPT annotation"""
    try:
        if hasattr(idea, 'keyword_annotations') and idea.keyword_annotations is not None:
            annotations = idea.keyword_annotations
            
            # The annotation has a 'concepts' attribute with structured data
            if hasattr(annotations, 'concepts') and annotations.concepts:
                concept_names = []
                for concept in annotations.concepts:
                    if hasattr(concept, 'name') and concept.name:
                        concept_names.append(concept.name.strip())
                
                if concept_names:
                    # Return list of concept names instead of joined string
                    return concept_names
        
        return ["general"]
    except Exception as e:
        logger.error(f"Error extracting concept annotation for '{idea.text}': {e}")
        return ["general"]

def extract_concept_annotation_legacy(idea):
    """Extract concept names as joined string for backward compatibility"""
    concepts = extract_concept_annotation(idea)
    if isinstance(concepts, list):
        return " › ".join(concepts)
    return concepts

# Removed classify_keyword_type - redundant since concept groups show actual brand names
# Removed extract_competitor_annotation - not reliable without brand context

def export_to_csv(df: pd.DataFrame, filename_prefix: str = "keyword_data") -> str:
    """Export DataFrame to CSV format and return as string"""
    try:
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_content = output.getvalue()
        output.close()
        
        # Also save to file for logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                f.write(csv_content)
            logger.info(f"Data exported to {filename}")
        except Exception as e:
            logger.warning(f"Could not save to file {filename}: {e}")
        
        return csv_content
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return "Error: Could not export data to CSV format"

def map_locations_ids_to_resource_names(client, location_ids):
    """Convert location IDs to resource names"""
    try:
        build_resource_name = client.get_service("GeoTargetConstantService").geo_target_constant_path
        return [build_resource_name(location_id) for location_id in location_ids]
    except Exception as e:
        logger.error(f"Error mapping location IDs to resource names: {e}")
        raise

def generate_keyword_ideas_core(client, customer_id, location_ids, language_id, keyword_texts, page_url, competitor_url=None):
    """Core function to generate keyword ideas with optional competitor analysis"""
    try:
        keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
        keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
        location_rns = map_locations_ids_to_resource_names(client, location_ids)
        language_rn = client.get_service("GoogleAdsService").language_constant_path(language_id)

        request = client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = customer_id
        request.language = language_rn
        request.geo_target_constants = location_rns
        request.include_adult_keywords = False
        request.keyword_plan_network = keyword_plan_network

        # Add keyword annotations for SEO-focused clustering and classification
        try:
            # Use client.enums approach (cleaner) or nested enum access
            keyword_concept_enum = client.enums.KeywordPlanKeywordAnnotationEnum.KEYWORD_CONCEPT
            request.keyword_annotation.append(keyword_concept_enum)
            logger.debug("Added KEYWORD_CONCEPT annotation for semantic clustering")
        except Exception as e:
            logger.warning(f"Could not add keyword annotations: {e}")

        # Handle different seed types including competitor site analysis
        if competitor_url and not keyword_texts and not page_url:
            request.site_seed.site = competitor_url
            logger.info(f"Generating keywords from competitor site: {competitor_url}")
        elif not keyword_texts and page_url:
            request.url_seed.url = page_url
            logger.info(f"Generating keywords from URL: {page_url}")
        elif keyword_texts and not page_url and not competitor_url:
            request.keyword_seed.keywords.extend(keyword_texts)
            logger.info(f"Generating keywords from seed keywords: {keyword_texts}")
        elif keyword_texts and page_url:
            request.keyword_and_url_seed.url = page_url
            request.keyword_and_url_seed.keywords.extend(keyword_texts)
            logger.info(f"Generating keywords from both URL and seed keywords")
        elif keyword_texts and competitor_url:
            # Combine keywords with competitor site analysis
            request.keyword_and_url_seed.url = competitor_url
            request.keyword_and_url_seed.keywords.extend(keyword_texts)
            logger.info(f"Generating keywords from both competitor site and seed keywords")

        keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
        logger.info("Successfully generated keyword ideas")
        return keyword_ideas
    except GoogleAdsException as ex:
        error_msg = f"Google Ads API Error: {str(ex.error.code)} - {ex.error.message}"
        logger.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Unexpected error in generate_keyword_ideas_core: {str(e)}"
        logger.error(error_msg)
        return None

def generate_historical_metrics(client, customer_id, location_ids, language_id, keywords):
    """Get historical monthly search data for past 24 months with proper date handling"""
    try:
        keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
        location_rns = map_locations_ids_to_resource_names(client, location_ids)
        language_rn = client.get_service("GoogleAdsService").language_constant_path(language_id)

        # Calculate date range more carefully (last 24 months)
        today = datetime.now()
        # Go back 24 months from current month
        start_date = today.replace(day=1) - timedelta(days=730)
        # Ensure we get the first day of the start month
        start_date = start_date.replace(day=1)
        
        # End date should be current month
        end_date = today.replace(day=1)

        request = client.get_type("GenerateKeywordHistoricalMetricsRequest")
        request.customer_id = customer_id
        request.keywords.extend(keywords)
        request.language = language_rn
        request.geo_target_constants.extend(location_rns)

        # Set historical metrics options with explicit month enum values
        historical_metrics_options = client.get_type("HistoricalMetricsOptions")
        historical_metrics_options.year_month_range.start.year = start_date.year
        historical_metrics_options.year_month_range.start.month = start_date.month
        historical_metrics_options.year_month_range.end.year = end_date.year
        historical_metrics_options.year_month_range.end.month = end_date.month
        request.historical_metrics_options = historical_metrics_options

        result = keyword_plan_idea_service.generate_keyword_historical_metrics(request=request)
        logger.info("Successfully generated historical metrics")
        return result
    except Exception as e:
        logger.error(f"Error generating historical metrics: {e}")
        raise

def clean_historical_data(historical_response):
    """Clean and normalize the historical data, fixing month issues"""
    try:
        cleaned_data = []

        for result in historical_response.results:
            for month_data in result.keyword_metrics.monthly_search_volumes:
                # Fix the month issue - Google Ads API sometimes returns month 13 for January
                corrected_month = int(month_data.month)
                corrected_year = int(month_data.year)
                
                # Handle month 13 (should be January of next year)
                if corrected_month == 13:
                    corrected_month = 1
                    corrected_year += 1
                
                # Handle month 0 (should be December of previous year)
                elif corrected_month == 0:
                    corrected_month = 12
                    corrected_year -= 1
                
                # Validate corrected month
                if corrected_month < 1 or corrected_month > 12:
                    continue
                    
                # Validate year
                if corrected_year < 2020 or corrected_year > 2030:
                    continue

                cleaned_data.append({
                    "keyword": result.text,
                    "year": corrected_year,
                    "month": corrected_month,
                    "monthly_searches": int(month_data.monthly_searches) if month_data.monthly_searches else 0
                })

        if not cleaned_data:
            logger.warning("No valid historical data found after cleaning")
            return pd.DataFrame()

        df = pd.DataFrame(cleaned_data)

        # Create proper datetime column for better handling
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        
        # Add month names
        df['month_name'] = df['date'].dt.strftime('%B')  # Full month name
        df['month_abbr'] = df['date'].dt.strftime('%b')   # Abbreviated month name
        df['year_month'] = df['date'].dt.strftime('%Y-%m')

        # Sort by date
        df = df.sort_values(['keyword', 'date']).reset_index(drop=True)

        # Fill missing months for complete time series
        keywords = df['keyword'].unique()
        all_months = pd.date_range(
            start=df['date'].min(),
            end=df['date'].max(),
            freq='MS'  # Month start frequency
        )

        # Create complete dataset with all months
        complete_data = []
        for keyword in keywords:
            for month in all_months:
                complete_data.append({
                    'keyword': keyword,
                    'date': month,
                    'year': month.year,
                    'month': month.month,
                    'month_name': month.strftime('%B'),
                    'month_abbr': month.strftime('%b'),
                    'year_month': month.strftime('%Y-%m')
                })

        complete_df = pd.DataFrame(complete_data)
        
        # Merge with actual data, filling missing values with 0
        final_df = complete_df.merge(
            df[['keyword', 'year', 'month', 'monthly_searches']], 
            on=['keyword', 'year', 'month'], 
            how='left'
        )
        final_df['monthly_searches'] = final_df['monthly_searches'].fillna(0)
        
        logger.info(f"Successfully processed historical data for {len(keywords)} keywords")
        return final_df.drop(columns=['date'])
    except Exception as e:
        logger.error(f"Error cleaning historical data: {e}")
        return pd.DataFrame()

@app.tool()
async def get_keyword_ideas_mcp(
    keywords: Optional[str] = "",
    page_url: Optional[str] = "",
    competitor_url: Optional[str] = "",
    locations: Optional[List[str]] = None,
    customer_id: Optional[str] = None,
    language_id: Optional[str] = "1000",
    result_limit: Optional[int] = 100,
    include_concept_grouping: Optional[bool] = True,
    export_csv: Optional[bool] = False
) -> str:
    """
    Generates SEO-focused keyword ideas from Google Ads Keyword Planner with semantic clustering.
    Uses credentials from environment variables (LOGIN_CUSTOMER_ID, DEVELOPER_TOKEN, etc.).

    Args:
        keywords: Comma-separated seed keywords (up to 50, max 100 chars each).
        page_url: Website URL to analyze for keyword suggestions.
        competitor_url: Competitor website URL for competitive keyword analysis.
        locations: List of location names to target (defaults to ["United States"]).
        customer_id: Override customer ID (defaults to LOGIN_CUSTOMER_ID from environment).
        language_id: Language targeting ID (default "1000" for English).
        result_limit: Maximum keywords to return, 1-100 (default 100).
        include_concept_grouping: Include semantic concept clustering (default True).
        export_csv: Include CSV format in response (default False).

    Returns:
        Markdown-formatted keyword data with search volumes, CPC, and concept groups.
    """
    # Set default location to United States
    if locations is None:
        locations = ["United States"]

    final_customer_id = customer_id if customer_id else DEFAULT_CUSTOMER_ID

    if not final_customer_id:
        logger.error("No customer ID provided")
        return "Error: Customer ID is required."

    # Validate and sanitize inputs
    keyword_texts_list = sanitize_keywords(keywords)
    if not keyword_texts_list and not page_url and not competitor_url:
        return "Error: At least one of keywords, page_url, or competitor_url must be provided."

    # Validate locations
    selected_location_ids, location_error = validate_locations(locations)
    if location_error:
        logger.error(f"Location validation failed: {location_error}")
        return location_error

    # Validate result limit
    result_limit, limit_warning = validate_result_limit(result_limit)
    
    try:
        googleads_client = GoogleAdsClient.load_from_dict(CREDENTIALS)
        googleads_client.login_customer_id = CREDENTIALS["login_customer_id"]

        keyword_ideas = generate_keyword_ideas_core(
            googleads_client,
            final_customer_id,
            selected_location_ids,
            language_id,
            keyword_texts_list,
            page_url,
            competitor_url
        )

        if keyword_ideas:
            results = []
            all_concept_groups = []  # Store concept groups for all keywords
            
            # First pass: collect all data and concept groups
            for idea in keyword_ideas:
                competition_value = str(idea.keyword_idea_metrics.competition)
                
                # Extract SEO-focused data
                concept_groups = extract_concept_annotation(idea) if include_concept_grouping else ["general"]
                all_concept_groups.append(concept_groups)
                
                # Extract CPC data for commercial value analysis
                low_cpc = idea.keyword_idea_metrics.low_top_of_page_bid_micros / 1_000_000 if idea.keyword_idea_metrics.low_top_of_page_bid_micros else 0
                high_cpc = idea.keyword_idea_metrics.high_top_of_page_bid_micros / 1_000_000 if idea.keyword_idea_metrics.high_top_of_page_bid_micros else 0
                
                # Build base result (without concept columns yet)
                result = {
                    "keyword": idea.text,
                    "avg_monthly_searches": idea.keyword_idea_metrics.avg_monthly_searches,
                    "low_cpc": round(low_cpc, 2),
                    "high_cpc": round(high_cpc, 2),
                    "concept_group": " › ".join(concept_groups),  # Keep legacy format for summary
                    "concept_groups_list": concept_groups  # Store the list for dynamic column creation
                }
                
                results.append(result)

            # Find the maximum number of concepts across all keywords
            max_concepts = max(len(concepts) for concepts in all_concept_groups) if all_concept_groups else 1
            
            # Second pass: add dynamic concept columns based on the maximum found
            for result in results:
                concept_groups = result["concept_groups_list"]
                
                # Add dynamic concept columns
                for j in range(max_concepts):
                    result[f"concept_{j+1}"] = concept_groups[j] if j < len(concept_groups) else ""
                
                # Remove the temporary concept_groups_list
                del result["concept_groups_list"]

            df = pd.DataFrame(results)
            df_sorted = df.sort_values(by="avg_monthly_searches", ascending=False).head(result_limit)
            
            # Reorganize columns for better display - concept columns first, then metrics
            base_columns = ["keyword", "avg_monthly_searches", "low_cpc", "high_cpc"]
            concept_columns = [col for col in df_sorted.columns if col.startswith("concept_") and col != "concept_group"]
            other_columns = [col for col in df_sorted.columns if col not in base_columns + concept_columns + ["concept_group"]]
            
            # Reorder columns: keyword, concept columns, then metrics
            display_columns = ["keyword"] + concept_columns + ["avg_monthly_searches", "low_cpc", "high_cpc"] + other_columns
            df_display = df_sorted[display_columns].copy()
            
            # Fill NaN values in concept columns with empty strings for cleaner display
            for col in concept_columns:
                df_display[col] = df_display[col].fillna("")
            
            response = f"## SEO Keyword Ideas Results\n\n"
            if limit_warning:
                response += f"**{limit_warning}**\n\n"
            
            response += f"**Locations:** {', '.join(locations)}\n"
            if competitor_url:
                response += f"**Competitor Analysis:** {competitor_url}\n"
            response += f"**Total Results:** {len(df_sorted)}\n"
            response += f"**Note:** Individual concept groups displayed in separate columns\n"
            
            response += "\n"
            response += df_display.to_markdown(index=False)
            
            # Add concept grouping summary with commercial value analysis
            if include_concept_grouping and len(df_sorted) > 0:
                # Calculate concept-level CPC insights first
                concept_cpc_analysis = df_sorted.groupby('concept_group').agg({
                    'high_cpc': ['mean', 'max'],
                    'low_cpc': 'mean',
                    'avg_monthly_searches': 'sum'
                }).round(2)
                concept_cpc_analysis.columns = ['avg_high_cpc', 'max_high_cpc', 'avg_low_cpc', 'total_search_volume']
                
                # Calculate Traffic Value (total volume * avg cpc)
                concept_cpc_analysis['traffic_value'] = (concept_cpc_analysis['total_search_volume'] * concept_cpc_analysis['avg_high_cpc']).round(2)
                
                # Sort by total search volume descending
                concept_cpc_analysis = concept_cpc_analysis.sort_values('total_search_volume', ascending=False)
                
                # Get keyword counts for each concept
                concept_summary = df_sorted['concept_group'].value_counts()
                
                response += f"\n\n### Semantic Concept Groups & Commercial Value\n"
                for concept in concept_cpc_analysis.index:
                    count = concept_summary[concept]
                    cpc_data = concept_cpc_analysis.loc[concept]
                    response += f"- **{concept}:** {count} keywords | Avg CPC: ${cpc_data['avg_high_cpc']:.2f} | Max CPC: ${cpc_data['max_high_cpc']:.2f} | Total Volume: {int(cpc_data['total_search_volume']):,} | Traffic Value: ${cpc_data['traffic_value']:,.2f}\n"
                
                # Identify highest value concept groups
                if len(concept_cpc_analysis) > 1:
                    highest_value_concept = concept_cpc_analysis['traffic_value'].idxmax()
                    highest_volume_concept = concept_cpc_analysis['total_search_volume'].idxmax()
                    highest_cpc_concept = concept_cpc_analysis['avg_high_cpc'].idxmax()
                    response += f"\n**Highest Commercial Value:** {highest_value_concept} (${concept_cpc_analysis.loc[highest_value_concept]['traffic_value']:,.2f} traffic value)\n"
                    response += f"**Highest Search Volume:** {highest_volume_concept} ({int(concept_cpc_analysis.loc[highest_volume_concept]['total_search_volume']):,} total searches)\n"
                    response += f"**Highest CPC:** {highest_cpc_concept} (${concept_cpc_analysis.loc[highest_cpc_concept]['avg_high_cpc']:.2f} avg CPC)\n"
            
            if export_csv:
                response += "\n\n## CSV Export\n\n```csv\n"
                response += export_to_csv(df_sorted, "seo_keyword_ideas")
                response += "\n```"
            
            logger.info(f"Successfully returned {len(df_sorted)} SEO-focused keyword ideas")
            return response
        else:
            logger.warning("No keyword ideas found")
            return "No keyword ideas found."

    except GoogleAdsException as ex:
        error_msg = f"Google Ads API Error: {str(ex.error.code)} - {ex.error.message}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return error_msg

@app.tool()
async def get_historical_keyword_data(
    keywords: str,
    locations: Optional[List[str]] = None,
    customer_id: Optional[str] = None,
    language_id: Optional[str] = "1000",
    format_type: Optional[str] = "summary",
    export_csv: Optional[bool] = False
) -> str:
    """
    Retrieves 24-month historical search volume data with time-series analysis.
    Uses credentials from environment variables (LOGIN_CUSTOMER_ID, DEVELOPER_TOKEN, etc.).

    Args:
        keywords: Comma-separated keywords to analyze (required, up to 50 keywords).
        locations: List of location names to target (defaults to ["United States"]).
        customer_id: Override customer ID (defaults to LOGIN_CUSTOMER_ID from environment).
        language_id: Language targeting ID (default "1000" for English).
        format_type: Output format - "summary", "detailed", or "pivot" (default "summary").
        export_csv: Include CSV format in response (default False).

    Returns:
        Markdown-formatted historical data with monthly search volumes and trends.
    """
    # Set default location to United States
    if locations is None:
        locations = ["United States"]

    final_customer_id = customer_id if customer_id else DEFAULT_CUSTOMER_ID

    if not final_customer_id:
        logger.error("No customer ID provided")
        return "Error: Customer ID is required."

    # Validate and sanitize inputs
    keyword_texts_list = sanitize_keywords(keywords)
    if not keyword_texts_list:
        return "Error: Valid keywords are required."

    # Validate locations
    selected_location_ids, location_error = validate_locations(locations)
    if location_error:
        logger.error(f"Location validation failed: {location_error}")
        return location_error

    try:
        googleads_client = GoogleAdsClient.load_from_dict(CREDENTIALS)
        googleads_client.login_customer_id = CREDENTIALS["login_customer_id"]

        # Get historical metrics
        historical_response = generate_historical_metrics(
            googleads_client,
            final_customer_id,
            selected_location_ids,
            language_id,
            keyword_texts_list
        )

        # Clean and process the data
        df_historical = clean_historical_data(historical_response)

        if df_historical.empty:
            logger.warning("No historical data found")
            return "No historical data found for the specified keywords."

        response = f"## Historical Keyword Data\n\n"
        response += f"**Keywords:** {', '.join(keyword_texts_list)}\n"
        response += f"**Locations:** {', '.join(locations)}\n"
        response += f"**Format:** {format_type}\n\n"

        # Return data based on format type
        if format_type == "summary":
            # Monthly summary across all keywords
            monthly_summary = df_historical.groupby(['month_name', 'month', 'year_month']).agg({
                'monthly_searches': 'sum'
            }).reset_index().sort_values(['year_month'])
            
            response += f"### Monthly Search Volume Summary\n\n{monthly_summary.to_markdown(index=False)}"
            result_df = monthly_summary
            
        elif format_type == "pivot":
            # Pivot table format
            df_pivot = df_historical.pivot_table(
                index=['year', 'month', 'year_month', 'month_name', 'month_abbr'],
                columns='keyword',
                values='monthly_searches',
                aggfunc='first'
            ).reset_index()
            df_pivot.columns.name = None
            df_pivot = df_pivot.sort_values(['year', 'month']).reset_index(drop=True)
            
            response += f"### Historical Search Volume Data (Pivot)\n\n{df_pivot.to_markdown(index=False)}"
            result_df = df_pivot
            
        else:  # detailed
            # Full detailed data
            df_sorted = df_historical.sort_values(['keyword', 'year', 'month'])
            response += f"### Detailed Historical Search Volume Data\n\n{df_sorted.to_markdown(index=False)}"
            result_df = df_sorted

        if export_csv:
            response += "\n\n## CSV Export\n\n```csv\n"
            response += export_to_csv(result_df, f"historical_data_{format_type}")
            response += "\n```"

        logger.info(f"Successfully returned historical data for {len(keyword_texts_list)} keywords")
        return response

    except GoogleAdsException as ex:
        error_msg = f"Google Ads API Error: {str(ex.error.code)} - {ex.error.message}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return error_msg

@app.tool()
async def get_month_over_month_analysis(
    keywords: str,
    locations: Optional[List[str]] = None,
    customer_id: Optional[str] = None,
    language_id: Optional[str] = "1000",
    export_csv: Optional[bool] = False
) -> str:
    """
    Analyzes month-over-month search volume changes with percentage calculations and trends.
    Uses credentials from environment variables (LOGIN_CUSTOMER_ID, DEVELOPER_TOKEN, etc.).

    Args:
        keywords: Comma-separated keywords to analyze (required, up to 50 keywords).
        locations: List of location names to target (defaults to ["United States"]).
        customer_id: Override customer ID (defaults to LOGIN_CUSTOMER_ID from environment).
        language_id: Language targeting ID (default "1000" for English).
        export_csv: Include CSV format in response (default False).

    Returns:
        Markdown-formatted analysis with monthly changes, percentages, and trend insights.
    """
    # Set default location to United States
    if locations is None:
        locations = ["United States"]

    final_customer_id = customer_id if customer_id else DEFAULT_CUSTOMER_ID

    if not final_customer_id:
        logger.error("No customer ID provided")
        return "Error: Customer ID is required."

    # Validate and sanitize inputs
    keyword_texts_list = sanitize_keywords(keywords)
    if not keyword_texts_list:
        return "Error: Valid keywords are required."

    # Validate locations
    selected_location_ids, location_error = validate_locations(locations)
    if location_error:
        logger.error(f"Location validation failed: {location_error}")
        return location_error

    try:
        googleads_client = GoogleAdsClient.load_from_dict(CREDENTIALS)
        googleads_client.login_customer_id = CREDENTIALS["login_customer_id"]

        # Get historical metrics
        historical_response = generate_historical_metrics(
            googleads_client,
            final_customer_id,
            selected_location_ids,
            language_id,
            keyword_texts_list
        )

        # Clean and process the data
        df_historical = clean_historical_data(historical_response)

        if df_historical.empty:
            logger.warning("No historical data found for month-over-month analysis")
            return "No historical data found for the specified keywords."

        # Calculate month-over-month changes
        df_analysis = []
        
        for keyword in df_historical['keyword'].unique():
            keyword_data = df_historical[df_historical['keyword'] == keyword].sort_values(['year', 'month'])
            keyword_data['previous_month_searches'] = keyword_data['monthly_searches'].shift(1)
            keyword_data['mom_change'] = keyword_data['monthly_searches'] - keyword_data['previous_month_searches']
            keyword_data['mom_change_pct'] = ((keyword_data['monthly_searches'] - keyword_data['previous_month_searches']) / keyword_data['previous_month_searches'] * 100).round(2)
            
            # Replace infinite values with 0
            keyword_data['mom_change_pct'] = keyword_data['mom_change_pct'].replace([float('inf'), -float('inf')], 0)
            keyword_data['mom_change_pct'] = keyword_data['mom_change_pct'].fillna(0)
            
            df_analysis.append(keyword_data)

        df_combined = pd.concat(df_analysis, ignore_index=True)
        
        # Select relevant columns for display
        df_mom = df_combined[['keyword', 'year_month', 'month_name', 'monthly_searches', 'previous_month_searches', 'mom_change', 'mom_change_pct']].sort_values(['keyword', 'year_month'])
        
        # Filter out first month (no previous data)
        df_mom = df_mom[df_mom['previous_month_searches'].notna()]

        response = f"## Month-over-Month Analysis\n\n"
        response += f"**Keywords:** {', '.join(keyword_texts_list)}\n"
        response += f"**Locations:** {', '.join(locations)}\n"
        response += f"**Total Data Points:** {len(df_mom)}\n\n"
        response += df_mom.to_markdown(index=False)
        
        if export_csv:
            response += "\n\n## CSV Export\n\n```csv\n"
            response += export_to_csv(df_mom, "mom_analysis")
            response += "\n```"

        logger.info(f"Successfully returned month-over-month analysis for {len(keyword_texts_list)} keywords")
        return response

    except GoogleAdsException as ex:
        error_msg = f"Google Ads API Error: {str(ex.error.code)} - {ex.error.message}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return error_msg

@app.tool()
async def get_keyword_search_volumes(
    keywords: str,
    locations: Optional[List[str]] = None,
    customer_id: Optional[str] = None,
    language_id: Optional[str] = "1000",
    export_csv: Optional[bool] = False
) -> str:
    """
    Retrieves 12-month average search volumes for keywords in a simple format.
    Uses credentials from environment variables (LOGIN_CUSTOMER_ID, DEVELOPER_TOKEN, etc.).
    
    Args:
        keywords: Comma-separated keywords to analyze (required, up to 50 keywords).
        locations: List of location names to target (defaults to ["United States"]).
        customer_id: Override customer ID (defaults to LOGIN_CUSTOMER_ID from environment).
        language_id: Language targeting ID (default "1000" for English).
        export_csv: Include CSV format in response (default False).
    
    Returns:
        Markdown table with keywords and their average monthly search volumes.
    """
    # Set default location to United States
    if locations is None:
        locations = ["United States"]

    final_customer_id = customer_id if customer_id else DEFAULT_CUSTOMER_ID

    if not final_customer_id:
        logger.error("No customer ID provided")
        return "Error: Customer ID is required."

    # Validate and sanitize inputs
    keyword_texts_list = sanitize_keywords(keywords)
    if not keyword_texts_list:
        return "Error: Valid keywords are required."

    # Validate locations
    selected_location_ids, location_error = validate_locations(locations)
    if location_error:
        logger.error(f"Location validation failed: {location_error}")
        return location_error

    try:
        googleads_client = GoogleAdsClient.load_from_dict(CREDENTIALS)
        googleads_client.login_customer_id = CREDENTIALS["login_customer_id"]

        # Get historical metrics
        historical_response = generate_historical_metrics(
            googleads_client,
            final_customer_id,
            selected_location_ids,
            language_id,
            keyword_texts_list
        )

        # Extract average monthly searches directly from API response
        results = []
        for result in historical_response.results:
            keyword = result.text
            avg_monthly_searches = result.keyword_metrics.avg_monthly_searches
            results.append({
                "keyword": keyword,
                "avg_monthly_searches": avg_monthly_searches
            })

        if not results:
            logger.warning("No search volume data found")
            return "No search volume data found for the specified keywords."

        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by="avg_monthly_searches", ascending=False)
        
        response = f"## Keyword Search Volumes (12-Month Average)\n\n"
        response += f"**Keywords:** {', '.join(keyword_texts_list)}\n"
        response += f"**Locations:** {', '.join(locations)}\n"
        response += f"**Total Keywords:** {len(df_sorted)}\n\n"
        response += df_sorted.to_markdown(index=False)
        
        if export_csv:
            response += "\n\n## CSV Export\n\n```csv\n"
            response += export_to_csv(df_sorted, "keyword_search_volumes")
            response += "\n```"

        logger.info(f"Successfully returned search volumes for {len(keyword_texts_list)} keywords")
        return response

    except GoogleAdsException as ex:
        error_msg = f"Google Ads API Error: {str(ex.error.code)} - {ex.error.message}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return error_msg

@app.tool()
async def list_available_locations() -> str:
    """
    Lists all supported location names and IDs for geographic targeting.
    Includes countries and major cities with their corresponding Google Ads location IDs.

    Returns:
        Markdown-formatted table of available locations organized by countries and cities.
    """
    try:
        locations_df = pd.DataFrame(list(LOCATION_OPTIONS.items()), columns=['Location Name', 'Location ID'])
        # Sort by location name for better readability
        locations_df = locations_df.sort_values('Location Name')
        
        response = f"## Available Locations ({len(locations_df)} total)\n\n"
        response += "### Countries\n"
        countries = locations_df[~locations_df['Location Name'].str.contains(',')].to_markdown(index=False)
        response += countries
        
        response += "\n\n### Cities\n"
        cities = locations_df[locations_df['Location Name'].str.contains(',')].to_markdown(index=False)
        response += cities
        
        logger.info("Successfully listed available locations")
        return response
    except Exception as e:
        error_msg = f"Error listing locations: {str(e)}"
        logger.error(error_msg)
        return error_msg

if __name__ == "__main__":
    logger.info("Starting Google Ads Keyword Planner MCP Server")
    app.run(transport='stdio')
