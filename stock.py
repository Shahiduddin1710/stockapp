import base64
from datetime import datetime as dt
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from nsepython import equity_history
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.svm import SVR
from datetime import date, timedelta

# Read the image file
with open("Assets/stock-icon.png", "rb") as f:
    image_data = f.read()

# Encode the image as a base64 string
encoded_image = base64.b64encode(image_data).decode()

# Define the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(children="Stock App"),
                html.Img(
                    src=f"data:image/png;base64,{encoded_image}",
                    style={"width": "100px"},
                ),
            ],
            className="banner",
        ),
        html.Div(
            [
                html.Label("✨Enter a valid Indian Stock Code✨"),
                html.Br(),
                dcc.Input(
                    id="stock_input", placeholder="Ex: SBIN", type="text", value=""
                ),
                html.Button("Submit", id="submit-button", n_clicks=0),
            ],
            className="input",
        ),
        html.Div(
            [
                dcc.DatePickerRange(
                    id="date-picker",
                    display_format="DD/MM/YYYY",
                    min_date_allowed=dt(1995, 8, 5),
                    max_date_allowed=dt.now(),
                    initial_visible_month=dt.now(),
                    end_date=dt.now().date(),
                )
            ],
            className="input",
        ),
        html.Div(
            dcc.Graph(id="Stock Chart", figure={}),
            className="frame",
        ),
        html.Div(
            id="alert-container",
            children=[
                dbc.Alert(
                    id="alert",
                    children="⚠️Invalid Stock Code⚠️",
                    color="warning",
                    is_open=False,
                    dismissable=True,
                )
            ],
            className="alert-container",
        ),
        html.Div(
            [
                html.Label("Enter number of days to forecast"),
                html.Br(),
                dcc.Input(
                    id="forecast-input", type="text", value="", placeholder="Ex.10 "
                ),
                html.Button("Forecast", id="forecast-btn", n_clicks=0),
            ],
            className="input",
        ),
        html.Div(dcc.Graph(id="forecast-graph", figure={}), className="frame"),
    ],
    className="main-div",
)


# Callback to update stock chart
@app.callback(
    [Output("Stock Chart", "figure"), Output("alert", "is_open")],
    [Input("submit-button", "n_clicks")],
    [
        State("date-picker", "start_date"),
        State("date-picker", "end_date"),
        State("stock_input", "value"),
    ],
)
def update_chart(n_clicks, start_date, end_date, stocks):
    if not n_clicks:
        raise PreventUpdate

    if not stocks:
        raise PreventUpdate
    else:
        try:
            start_date = dt.strptime(start_date.split("T")[0], "%Y-%m-%d").strftime(
                "%d-%m-%Y"
            )
            end_date = dt.strptime(end_date.split("T")[0], "%Y-%m-%d").strftime(
                "%d-%m-%Y"
            )
            df = equity_history(stocks, "EQ", start_date, end_date)
            if "CH_TIMESTAMP" not in df.columns:
                raise Exception(f"No data found for {stocks}")
        except Exception as e:
            print(e)
            return {"data": [], "layout": {"title": "No Data Entered"}}, True

    fig = {
        "data": [get_stock_graph(df)],
        "layout": dict(
            title=("📊" + stocks + "📊"), height=500, margin=dict(l=100, r=0, t=50, b=0)
        ),
    }

    return fig, False


# Callback to generate forecast and display
@app.callback(
    Output("forecast-graph", "figure"),
    [Input("forecast-btn", "n_clicks")],
    [State("forecast-input", "value"), State("stock_input", "value")],
)
def generate_forecast(n_clicks, days, stocks):
    if not n_clicks or not days or not stocks:
        raise PreventUpdate
    else:
        fig = prediction(stocks, days)
    return fig


# Function to create candlestick graph
def get_stock_graph(df):
    Candlefig = go.Candlestick(
        x=df["CH_TIMESTAMP"],
        open=df["CH_OPENING_PRICE"],
        high=df["CH_TRADE_HIGH_PRICE"],
        low=df["CH_TRADE_LOW_PRICE"],
        close=df["CH_CLOSING_PRICE"],
    )
    return Candlefig


def prediction(stocks, n_days):
    st = (dt.today() - timedelta(days=100)).strftime("%d-%m-%Y")
    en = dt.today().strftime("%d-%m-%Y")
    df = equity_history(stocks, "EQ", st, en)
    df["CH_TIMESTAMP"] = df.index

    days = list()
    for i in range(len(df.CH_TIMESTAMP)):
        days.append([i])

    X = days
    Y = df["CH_CLOSING_PRICE"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, shuffle=False
    )

    gsc = GridSearchCV(
        estimator=SVR(kernel="rbf"),
        param_grid={
            "C": [0.001, 0.01, 0.1, 10, 100, 1000],
            "epsilon": [
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.1,
                1,
                5,
                10,
                50,
                100,
                150,
                1000,
            ],
            "gamma": [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000],
        },
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=0,
        n_jobs=-1,
    )

    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(
        kernel="rbf",
        C=best_params["C"],
        epsilon=best_params["epsilon"],
        gamma=best_params["gamma"],
        max_iter=-1,
    )

    rbf_svr = best_svr

    rbf_svr.fit(x_train, y_train)

    n_days = int(n_days)
    output_days = list()
    for i in range(1, n_days + 1):
        output_days.append([i + x_test[-1][0]])

    dates = []
    current = date.today()
    for i in range(n_days):
        current += timedelta(days=1)
        dates.append(current)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=rbf_svr.predict(output_days), mode="lines+markers", name="data"
        )
    )
    fig.update_layout(
        title="Predicted Close Price of next " + str(n_days) + " days for " + stocks,
        xaxis_title="Date",
        yaxis_title="Closed Price",
    )

    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
