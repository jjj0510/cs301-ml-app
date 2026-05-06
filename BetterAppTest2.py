import base64
import io
import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#--------------------------------
#--[INITIALIZE THE APP]--
#--------------------------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={"fontFamily": "Arial", "maxWidth": "1250px", "margin": "auto", "backgroundColor": "#ffffff", "padding": "20px"}, children=[

    dcc.Store(id="stored-df"),
    dcc.Store(id="store-featureOrder"),

    #Upload section
    html.Div([
        dcc.Upload(
            id="upload-data",
            children=html.A("Upload File", style={"color": "black", "textDecoration": "none"}),
            style={
                "width": "100%", "height": "50px", "lineHeight": "50px",
                "textAlign": "center", "backgroundColor": "#f2f2f2", "cursor": "pointer"
            },
            multiple=False, accept=".csv"
        ),
        html.Div(id="upload-status", style={"fontSize": "0.8em", "textAlign": "center"})
    ], style={"padding": "10px"}),

    #Target selection
    html.Div([
        html.Span("Select Target: ", style={"fontWeight": "bold",}),
        dcc.Dropdown(
            id="dropdown-target", 
            style={"width": "200px", "display": "inline-block", "verticalAlign": "middle"}
        )
    ], style={
        "padding": "15px", 
        "textAlign": "center", 
        "backgroundColor": "#FFFFFF18", 
        "borderBottom": "1px solid #ddd"
    }),

    #Charts
    html.Div([
        #Left chart
        html.Div([
            html.Div([
                dcc.RadioItems(
                    id="radio-cat", 
                    inline=True,
                    style={"marginBottom": "10px"},
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"marginRight": "15px"}
                ),
            ], style={"textAlign": "center"}),
            
            dcc.Graph(id="chart-cat", style={"height": "400px"})
        ], style={"flex": "1", "backgroundColor": "#fdfdfd", "padding": "15px", "margin": "10px", "border": "1px solid #eeeeee"}),

        #Right chart
        html.Div([
            dcc.Graph(id="chart-correlation", style={"height": "400px", "marginTop": "32px"})
        ], style={"flex": "1", "backgroundColor": "#fdfdfd", "padding": "15px", "margin": "10px", "border": "1px solid #eeeeee"})
    ], style={"display": "flex", "flexDirection": "row"}),

    #Feature selection and training section
    html.Div([
        dcc.Checklist(
            id="checklist-features", 
            inline=True,
            style={"marginBottom": "15px", "display": "flex", "justifyContent": "center", "flexWrap": "wrap", "gap": "15px"}
        ),
        html.Button(
            "Train", 
            id="button-train", 
            style={
                "width": "300px", 
                "height": "42px", 
                "fontWeight": "normal", 
                "fontSize": "16px",
                "cursor": "pointer",
                "border": "1px solid #ccc"
            }
        ),
        html.Div(id="train-status", style={"marginTop": "20px", "fontWeight": "normal", "fontSize": "16px"})
    ], style={
        "textAlign": "center", 
        "padding": "30px", 
        "margin": "10px", 
        "backgroundColor": "#fdfdfd",
        "border": "1px solid #eeeeee",
        "borderRadius": "5px"
    }),

    #Prediction section
    html.Div([
        html.Div([
            dcc.Input(
                id="input-predict-values",
                placeholder="Enter values (e.g. 20, dinner)",
                type="text",
                style={"width": "400px", "marginRight": "10px", "padding": "8px"}
            ),
            html.Button(
                "Predict", 
                id="button-predict", 
                style={"padding": "8px 20px", "fontWeight": "normal", "border": "1px solid #ccc"}
            ),
            html.Span(id="predict-result", style={"marginLeft": "15px", "fontWeight": "normal", "fontSize": "16px"})
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "center"})
    ], style={
        "padding": "40px", 
        "margin": "10px", 
        "border": "1px solid #eeeeee",
        "borderRadius": "5px"
    })
])

#Global values
TRAIN_PIPELINE = None
LABEL_ENCODER = None
TASK_TYPE = None

def ensure_numeric(df):
    return df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == "object" else col)

#--------------------------------
#--[Upload CSV]--
#--------------------------------
@app.callback(
    Output("stored-df", "data"),
    Output("upload-status", "children"),
    Output("dropdown-target", "options"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def parse_upload(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    options = [{"label": col, "value": col} for col in df.columns]
    return df.to_json(date_format="iso", orient="split"), f"Loaded: {filename}", options

#------------------------------------
#--[Update UI]--
#------------------------------------
@app.callback(
    Output("radio-cat", "options"),
    Output("radio-cat", "value"),
    Output("chart-cat", "figure"),
    Output("chart-correlation", "figure"),
    Output("checklist-features", "options"),
    Input("dropdown-target", "value"),
    Input("radio-cat", "value"),
    State("stored-df", "data"),
    prevent_initial_call=True
)
def update_ui_elements(target_col, selected_cat, df_json):
    if not df_json or not target_col:
        return [], None, go.Figure(), go.Figure(), []

    df = pd.read_json(io.StringIO(df_json), orient="split")

    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != target_col]
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]

    radio_options = [{"label": c, "value": c} for c in cat_cols]
    
    trigger = ctx.triggered_id
    if trigger == "dropdown-target":
        selected_cat = cat_cols[0] if cat_cols else None

    feat_options = [{"label": c, "value": c} for c in df.columns if c != target_col]

    return (
        radio_options,
        selected_cat,
        make_cat_chart(df, selected_cat, target_col),
        make_corr_chart(df, num_cols, target_col),
        feat_options
    )

#-----------------------------------
#--[TRAIN MODEL]--
#-----------------------------------
@app.callback(
    Output("train-status", "children"),
    Output("store-featureOrder", "data"),
    Input("button-train", "n_clicks"),
    State("stored-df", "data"),
    State("dropdown-target", "value"),
    State("checklist-features", "value"),
    prevent_initial_call=True
)
def train_model(n_clicks, df_json, target_col, selected_features):
    global TRAIN_PIPELINE, LABEL_ENCODER, TASK_TYPE

    if not selected_features or len(selected_features) == 0:
        return "Select features first", None

    #Prevent target leaking
    if target_col in selected_features:
        selected_features = [f for f in selected_features if f != target_col]

    df = pd.read_json(io.StringIO(df_json), orient="split")
    X, y = df[selected_features], df[target_col]

    mask = y.notna()
    X, y = X[mask], y[mask]

    if pd.api.types.is_numeric_dtype(y):
        TASK_TYPE = "regression"
        model = RandomForestRegressor()
    else:
        TASK_TYPE = "classification"
        LABEL_ENCODER = LabelEncoder()
        y = LABEL_ENCODER.fit_transform(y.astype(str))
        model = RandomForestClassifier()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Detect safe columns
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("o", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    pipe = Pipeline([("pre", preprocessor), ("m", model)])

    #Prevent any silent crashes
    try:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
    except Exception as e:
        return f"Training error: {str(e)}", None

    TRAIN_PIPELINE = pipe

    if TASK_TYPE == "regression":
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return f"R2: {r2:.2f}, RMSE: {rmse:.2f}", selected_features
    else:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        return f"Accuracy: {acc:.2f}, F1: {f1:.2f}", selected_features

#--------------------------------
#--[PREDICT]--
#--------------------------------
@app.callback(
    Output("predict-result", "children"),
    Input("button-predict", "n_clicks"),
    State("input-predict-values", "value"),
    State("store-featureOrder", "data"),
    State("dropdown-target", "value"),
    prevent_initial_call=True
)
def predict(n_clicks, raw_input, feature_order, target_col):
    if not raw_input:
        return "Please enter input values"
    if TRAIN_PIPELINE is None:
        return "Train model first"
    if not raw_input:
        return "Please enter input values"

    tokens = [t.strip() for t in raw_input.split(",")]
    if len(tokens) != len(feature_order):
        return f"Expected {len(feature_order)} values"

    df = pd.DataFrame([tokens], columns=feature_order)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    pred = TRAIN_PIPELINE.predict(df)[0]
    if TASK_TYPE == "classification":
        pred = LABEL_ENCODER.inverse_transform([int(pred)])[0]
        return f"Predicted class: {pred}"
    else:
        return f"Predicted {target_col} is: {pred:.2f}"

#--------------------------------
#--[CHARTS]--
#--------------------------------
def make_cat_chart(df, cat_col, target_col):
    if not cat_col:
        return go.Figure()
    avg = df.groupby(cat_col)[target_col].mean().reset_index()
    fig = go.Figure(go.Bar(x=avg[cat_col], y=avg[target_col], marker_color="#B2DDEE"))
    fig.update_layout(
        title=f"Average {target_col} by {cat_col}",
        plot_bgcolor="#fdfdfd", paper_bgcolor="#fdfdfd",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def make_corr_chart(df, num_cols, target_col):
    if not num_cols or not pd.api.types.is_numeric_dtype(df[target_col]):
        return go.Figure()
    corrs = df[num_cols].corrwith(df[target_col]).abs()
    fig = go.Figure(go.Bar(x=corrs.index, y=corrs, marker_color="#4C99F8"))
    fig.update_layout(
        title=f"Correlation Strength with {target_col}",
        plot_bgcolor="#fdfdfd", paper_bgcolor="#fdfdfd",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

#--------------------------------------------
#--[CLEAR FEATURES WHEN TARGET IS CHANGED]--
#--------------------------------------------
@app.callback(
    Output("checklist-features", "value"),
    Input("dropdown-target", "value"),
    prevent_initial_call=True
)
def clear_features_on_target_change(target):
    return []

if __name__ == "__main__":
    app.run(debug=True)