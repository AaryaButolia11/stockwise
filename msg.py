# msg.py
import requests
from twilio.rest import Client
import os

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_sms_number = os.getenv("TWILIO_SMS_NUMBER")
twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")
STOCK_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# --- API Keys (Load from Environment Variables) ---
# STOCK_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "KUFM7ZOE2RPKQ7X7")
# NEWS_API_KEY = os.getenv("NEWS_API_KEY", "451415d68a1b4f3e8a055047d2509f38")

# Initialize Twilio client once
client = Client(account_sid, auth_token)

# --- Function to fetch current stock price from Alpha Vantage ---
def fetch_current_price(symbol):
    """
    Fetch the most recent daily closing stock price using Alpha Vantage's TIME_SERIES_DAILY.
    Returns (price, symbol) or (None, None) if an error occurs.
    """
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": STOCK_API_KEY,
            "outputsize": "compact"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        time_series_key = "Time Series (Daily)"
        if time_series_key not in data or not data[time_series_key]:
            search_params = {
                "function": "SYMBOL_SEARCH",
                "keywords": symbol,
                "apikey": STOCK_API_KEY
            }
            search_response = requests.get(url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            company_name_for_error = symbol
            if "bestMatches" in search_data and search_data["bestMatches"]:
                for match in search_data["bestMatches"]:
                    if match.get("1. symbol", "").upper() == symbol.upper():
                        company_name_for_error = match.get("2. name", symbol)
                        break
            raise ValueError(f"No daily time series data found for {symbol}. Found company name: {company_name_for_error}")

        daily_data = data[time_series_key]
        most_recent_date = next(iter(daily_data))
        most_recent_day_data = daily_data[most_recent_date]

        price = float(most_recent_day_data.get("4. close"))

        return price, symbol

    except requests.exceptions.RequestException as req_err:
        print(f"Network or API error fetching price for {symbol}: {req_err}")
        return None, None
    except ValueError as val_err:
        print(f"Data error for {symbol}: {val_err}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred fetching price for {symbol}: {e}")
        return None, None

def send_alert_sms(to_phone_number, message):
    """
    Sends an SMS alert to the specified phone number.
    Returns True on success, False on failure.
    """
    if not all([account_sid, auth_token, twilio_sms_number]):
        print("Error: Twilio SMS credentials are not fully set.")
        return False

    try:
        message_response = client.messages.create(
            body=message,
            from_=twilio_sms_number,
            to=to_phone_number
        )
        print(f"SMS Message SID: {message_response.sid}")
        return True
    except Exception as e:
        print(f"Error sending SMS to {to_phone_number}: {e}")
        return False

def send_alert_whatsapp(to_whatsapp_number, message):
    """
    Sends a WhatsApp message to the specified number.
    Note: Ensure your Twilio WhatsApp sandbox is configured or a Twilio number is WhatsApp enabled.
    Returns True on success, False on failure.
    """
    if not all([account_sid, auth_token, twilio_whatsapp_number]):
        print("Error: Twilio WhatsApp credentials are not fully set.")
        return False

    if not to_whatsapp_number.startswith("whatsapp:"):
        to_whatsapp_number = "whatsapp:" + to_whatsapp_number.lstrip('+')

    try:
        message_response = client.messages.create(
            body=message,
            from_=twilio_whatsapp_number,
            to=to_whatsapp_number
        )
        print(f"WhatsApp Message SID: {message_response.sid}")
        return True
    except Exception as e:
        print(f"Error sending WhatsApp message to {to_whatsapp_number}: {e}")
        return False

# --- New function to encapsulate the stock analysis and news alert logic ---
def send_stock_news_alert(stock_symbol, company_name, phone_number, threshold_percent=1):
    """
    Fetches stock data, checks for significant price change, and sends news alerts
    via SMS and WhatsApp if the change exceeds the threshold.

    Args:
        stock_symbol (str): The ticker symbol of the stock (e.g., "TSLA").
        company_name (str): The full name of the company (e.g., "Tesla Inc").
        phone_number (str): The recipient's phone number for SMS and WhatsApp.
        threshold_percent (int): The percentage change threshold to trigger news alerts.
    Returns:
        bool: True if alerts were sent, False otherwise.
    """
    try:
        stock_params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": stock_symbol,
            "apikey": STOCK_API_KEY,
        }
        stock_response = requests.get(url="https://www.alphavantage.co/query", params=stock_params)
        stock_response.raise_for_status()
        stock_data = stock_response.json()["Time Series (Daily)"]

        data_list = [value for (key, value) in stock_data.items()]

        if len(data_list) < 2:
            print(f"Insufficient historical data for {stock_symbol} to calculate price change.")
            return False

        yesterday_closing_price = float(data_list[0]["4. close"])
        day_before_yesterday_closing_price = float(data_list[1]["4. close"])

        diff = yesterday_closing_price - day_before_yesterday_closing_price
        up_down = "🔺" if diff > 0 else "🔻"
        diff_percent = round((diff / yesterday_closing_price) * 100)

        if abs(diff_percent) >= threshold_percent:
            news_params = {
                "apiKey": NEWS_API_KEY,
                "qInTitle": company_name, # Search for news related to the company name
                "pageSize": 3 # Limit to 3 articles
            }

            news_response = requests.get(url="https://newsapi.org/v2/everything", params=news_params)
            news_response.raise_for_status()
            articles = news_response.json()["articles"]

            if not articles:
                print(f"No relevant news articles found for {company_name}.")
                return False

            formatted_articles = [
                f"{stock_symbol}: {up_down}{diff_percent}%\nHeadline: {article['title']}. \nBrief: {article['description']}"
                for article in articles
            ]

            # Send alerts for each article
            alert_sent_status = False
            for article_msg in formatted_articles:
                sms_success = send_alert_sms(phone_number, article_msg)
                whatsapp_success = send_alert_whatsapp(phone_number, article_msg) # Uses same phone_number for WhatsApp
                if sms_success or whatsapp_success:
                    alert_sent_status = True
            
            if alert_sent_status:
                print(f"News alerts sent for {stock_symbol} to {phone_number}!")
                return True
            else:
                print(f"Failed to send any news alerts for {stock_symbol} to {phone_number}.")
                return False
        else:
            print(f"Price change for {stock_symbol} ({diff_percent}%) is below the {threshold_percent}% threshold. No alerts sent.")
            return False

    except requests.exceptions.RequestException as req_err:
        print(f"API or network error in send_stock_news_alert for {stock_symbol}: {req_err}")
        return False
    except KeyError as key_err:
        print(f"Data structure error in API response for {stock_symbol}: {key_err}. Check API documentation.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred in send_stock_news_alert for {stock_symbol}: {e}")
        return False