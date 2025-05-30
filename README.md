Simple installation steps:
1. clone the repo 
2. in the directory of the folder : pip install uv ( if you dont have it already)
3. uv venv
4. uv sync

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
