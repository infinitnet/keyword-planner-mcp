import os
from mcp.server.fastmcp import FastMCP
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

app = FastMCP("Google Ads Keyword Planner")

CREDENTIALS = {
    "developer_token": os.getenv("DEVELOPER_TOKEN"),
    "client_id": os.getenv("CLIENT_ID"),
    "client_secret": os.getenv("CLIENT_SECRET"),
    "refresh_token": os.getenv("REFRESH_TOKEN"),
    "login_customer_id": os.getenv("LOGIN_CUSTOMER_ID"),
    "use_proto_plus": True,
    "token_uri": "https://oauth2.googleapis.com/token",
}
LINKED_CUSTOMER_ID = os.getenv("LINKED_CUSTOMER_ID")

LOCATION_OPTIONS = {
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
    "New York, NY": "1023191",
    "Los Angeles, CA": "1012429",
    "London, UK": "1006879",
    "Mumbai, India": "1007751",
    "Delhi, India": "1007789",
    "Bangalore, India": "1007785",
    "Toronto, Canada": "1009714",
    "Sydney, Australia": "1009714"
}

def map_locations_ids_to_resource_names(client, location_ids):
    build_resource_name = client.get_service("GeoTargetConstantService").geo_target_constant_path
    return [build_resource_name(location_id) for location_id in location_ids]

def generate_keyword_ideas_core(client, customer_id, location_ids, language_id, keyword_texts, page_url):
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

    if not keyword_texts and page_url:
        request.url_seed.url = page_url
    elif keyword_texts and not page_url:
        request.keyword_seed.keywords.extend(keyword_texts)
    elif keyword_texts and page_url:
        request.keyword_and_url_seed.url = page_url
        request.keyword_and_url_seed.keywords.extend(keyword_texts)

    try:
        keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
        return keyword_ideas
    except GoogleAdsException as ex:
        print(f"Google Ads API Error: {ex}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

@app.tool()
async def get_keyword_ideas_mcp(
    keywords: Optional[str] = "",
    page_url: Optional[str] = "",
    locations: Optional[List[str]] = None,
    customer_id: Optional[str] = None,
    language_id: Optional[str] = "1000",
    result_limit: Optional[int] = 20
) -> str:
    """
    Generates keyword ideas from Google Ads Keyword Planner.

    Args:
        keywords: Comma-separated list of keywords.
        page_url: A URL to generate keyword ideas from.
        locations: A list of location names to target.
        customer_id: Google Ads customer ID.
        language_id: The language ID (default '1000' for English).
        result_limit: Number of top results to return (default is 20).

    Returns:
        Markdown string with keyword data or an error message.
    """
    if locations is None:
        locations = ["New York, NY"]

    final_customer_id = customer_id if customer_id else LINKED_CUSTOMER_ID

    if not final_customer_id:
        return "Error: Customer ID is required."

    keyword_texts_list = [k.strip() for k in keywords.split(",")] if keywords else []

    selected_location_ids = []
    for loc_name in locations:
        if loc_name in LOCATION_OPTIONS:
            selected_location_ids.append(LOCATION_OPTIONS[loc_name])
        else:
            return f"Error: Invalid location: {loc_name}. Available locations: {list(LOCATION_OPTIONS.keys())}"

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
        )

        if keyword_ideas:
            results = []
            for idea in keyword_ideas:
                competition_value = idea.keyword_idea_metrics.competition.name
                results.append({
                    "keyword": idea.text,
                    "avg_monthly_searches": idea.keyword_idea_metrics.avg_monthly_searches,
                    "competition": competition_value,
                    "low_top_of_page_bid_micros": idea.keyword_idea_metrics.low_top_of_page_bid_micros,
                    "high_top_of_page_bid_micros": idea.keyword_idea_metrics.high_top_of_page_bid_micros
                })

            df = pd.DataFrame(results)
            df_sorted = df.sort_values(by="avg_monthly_searches", ascending=False).head(result_limit)
            return df_sorted.to_markdown(index=False)
        else:
            return "No keyword ideas found."

    except GoogleAdsException as ex:
        return f"Google Ads API Error: {ex.error.message}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(transport='stdio')
