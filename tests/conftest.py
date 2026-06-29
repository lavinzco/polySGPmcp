import pytest


@pytest.fixture
def sample_temperature_events():
    return [
        {
            "id": "ev-1",
            "title": "Highest temperature in NYC on June 26?",
            "slug": "highest-temperature-nyc-jun-26",
            "active": True,
            "closed": False,
            "markets": [
                {
                    "id": "m-1",
                    "question": "Will the highest temperature in NYC be 75°F or below on June 26?",
                    "outcomePrices": '["0.15", "0.85"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "id": "m-2",
                    "question": "Will the highest temperature in NYC be between 76-80°F on June 26?",
                    "outcomePrices": '["0.25", "0.75"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "id": "m-3",
                    "question": "Will the highest temperature in NYC be 90°F or above on June 26?",
                    "outcomePrices": '["0.10", "0.90"]',
                    "active": True,
                    "closed": False,
                },
            ],
        },
        {
            "id": "ev-2",
            "title": "Highest temperature in London on June 26?",
            "slug": "highest-temperature-london-jun-26",
            "active": True,
            "closed": False,
            "markets": [
                {
                    "id": "m-4",
                    "question": "Will the high in London be 60°F or below on June 26?",
                    "outcomePrices": '["0.30", "0.70"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "id": "m-5",
                    "question": "Will the high in London be between 61-65°F on June 26?",
                    "outcomePrices": '["0.40", "0.60"]',
                    "active": True,
                    "closed": False,
                },
            ],
        },
        {
            "id": "ev-3",
            "title": "Where will 2026 rank among the hottest years on record?",
            "slug": "hottest-year-2026",
            "active": True,
            "closed": False,
            "markets": [
                {
                    "id": "m-6",
                    "question": "Will 2026 be the hottest year on record?",
                    "outcomePrices": '["0.17", "0.83"]',
                    "active": True,
                    "closed": False,
                },
            ],
        },
    ]


@pytest.fixture
def sample_wttr_response():
    return {
        "current_condition": [
            {
                "temp_C": "32",
                "temp_F": "90",
                "humidity": "75",
                "windspeedKmph": "20",
                "winddir16Point": "SSE",
                "weatherDesc": [{"value": "Partly cloudy"}],
                "FeelsLikeC": "36",
                "pressure": "1012",
                "precipMM": "0.5",
                "visibility": "10",
                "uvIndex": "8",
            }
        ]
    }


@pytest.fixture
def sample_gamma_markets():
    return [
        {
            "id": "1",
            "question": "Will a Category 4+ hurricane hit Florida in 2026?",
            "description": "This market resolves YES if a hurricane of category 4 or higher makes landfall in Florida.",
            "slug": "hurricane-florida-2026",
            "active": True,
            "closed": False,
            "liquidity": "50000",
            "volume": "120000",
        },
        {
            "id": "2",
            "question": "Will Bitcoin reach $100k?",
            "description": "Resolves YES if BTC price reaches 100000 USD.",
            "slug": "btc-100k",
            "active": True,
            "closed": False,
            "liquidity": "900000",
            "volume": "5000000",
        },
        {
            "id": "3",
            "question": "Will NYC temperature exceed 110°F this summer?",
            "description": "Heat wave record temperature in New York City, fahrenheit reading.",
            "slug": "nyc-temp-110",
            "active": True,
            "closed": False,
            "liquidity": "20000",
            "volume": "45000",
        },
        {
            "id": "4",
            "question": "Total snowfall in Chicago above 60 inches?",
            "description": "Will precipitation as snow exceed 60 inches in the 2025-2026 season?",
            "slug": "chicago-snow-60",
            "active": True,
            "closed": False,
            "liquidity": "15000",
            "volume": "30000",
        },
    ]
