# AI-Backtest-Strategy-G-1

AI Memecoin backtester, first generation.

## Run locally

From the repository root, install the dependencies and launch the Streamlit app:

```bash
python -m pip install -r requirements.txt
streamlit run app_standalone.py
```

Streamlit will print a local URL to open in your browser.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub, including `app_standalone.py` and `requirements.txt`.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app** and connect the GitHub repository.
3. Select the branch to deploy and set **Main file path** to `app_standalone.py`.
4. Deploy the app.

Community Cloud installs the packages listed in `requirements.txt` during deployment.
