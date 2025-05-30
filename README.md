Simple installation steps:
1. Clone or Download the zip
2. Ensure you place your .env file with credentials in the same folder
4. In the directory of the folder : pip install uv ( if you dont have it already)
5. uv venv
6. uv sync

ensure you edit the claude configuration to :
```
"Keyword Planner": {
            "command": "uv",
            "args": [
                "--directory",
                "C:\\Users\\Rentla.in\\Documents\\Keyword Planner MCP",
                "run",
                "google_ads_server.py"
            ]
        }
```

then you can open claude and query.
