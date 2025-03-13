from data_handler import DataHandler

# Define periods for different market conditions
periods = [
    # Bull Market: October 2023 - February 2024
    {
        'name': 'bull_2023-10-01_to_2024-02-28',
        'start_date': '2023-10-01 00:00:00',
        'end_date': '2024-02-28 00:00:00'
    },
    # Bear Market: May 2023 - August 2023
    {
        'name': 'bear_2023-05-01_to_2023-08-31',
        'start_date': '2023-05-01 00:00:00',
        'end_date': '2023-08-31 00:00:00'
    },
    # Sideways Market: August 2024 - October 2024
    {
        'name': 'sideways_2024-08-01_to_2024-10-31',
        'start_date': '2024-08-01 00:00:00',
        'end_date': '2024-10-31 00:00:00'
    }
]

# Define timeframes
timeframes = ['1h', '4h', '15m']

# Fetch data for each period and timeframe
handler = DataHandler()
for period in periods:
    for timeframe in timeframes:
        handler.fetch_historical_data(
            timeframe=timeframe,
            start_date=period['start_date'],
            end_date=period['end_date']
        )