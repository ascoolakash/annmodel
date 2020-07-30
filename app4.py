import base64
import datetime
import io
import sys
import dash
#import urllib
import six.moves.urllib as urllib
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
f = open('out.txt', 'w')
f.close()
app = dash.Dash(__name__)
df = []
app.layout = html.Div([
                html.Div([
                    html.H1("Python Dash ANN Dashboard",style={"textAlign": "center"}),
                    #html.Img(src="/assets/CollegeofEngineeringLogog.png"),
                    html.Img(src="/assets/UGALogo.jpg")
                ],className="banner"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.Button('Upload File')
                    ],style={
                        'textAlign': 'left',
                    }),
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.A('Download',
                    id='download-link',
                    download="predictions.csv",
                    href="",
                    target="_blank"
                ),
                dcc.Tabs(id="tabs", children=[
                    dcc.Tab(label='ANN MODELING', children=[
                        #html.H6("Remove N/A"),
                        #html.Div(id='model-accuracy'),
                        html.Div([
                            html.Div(className='row',children=[
                                html.Div([
                                html.H6(""),
                                html.H6(""),
                                html.H6(""),
                                #html.H6("Drop feature:"),
                                dcc.Input(id="input4", type="Text", placeholder="Input Nodes :"),
                                html.P(),
                                dcc.Input(id="input_1", type="Text", placeholder="Output Nodes :"),
                                html.P(),
                                #html.H6("Scaler:"),
                                #html.Div([
                                #    dcc.Dropdown(
                                #        id='select-scaler',
                                #        options=[
                                #            {'label': 'Standard Scaler', 'value': 'SS'},
                                #            {'label': 'Min Max Scalar', 'value': 'SMMS'},
                                #            {'label': 'Max Abs Scaler', 'value': 'MAS'},
                                #            {'label': 'Normalizer', 'value': 'NORM'}
                                #            ],
                                        #value='SS',
                                #        placeholder='Select Scalar...',
                                #        style={
                                #            'width':'50%'
                                #        }
                                #    )
                                #]),
                                #html.H6("Number of layer"),
                                dcc.Input(id="input5", type="number", placeholder="Number of Hidden Layer"),
                                #html.P(),
                                #html.I("Type Activation Function From Below List:"),
                                #html.P(),
                                #html.I("1.>Sigmoid activation function 2.> Linear activation function 3.>RELU 4.>ELU"),
                                #html.P(),
                                #dcc.Input(id="input6", type="text", placeholder="Activation Function"),
                                #html.H6("Select activation function:"),
                                html.Div([
                                    #html.H6("Enter Test, Train and Validation Ratio:"),
                                    dcc.Input(id="input1", type="number", placeholder="Train %"),
                                    dcc.Input(id="input2", type="number", placeholder="Test %"),
                                    dcc.Input(id="input3", type="number", placeholder="Validation %"),
                                    ]
                                ),
                                html.Div([
                                    dcc.Dropdown(
                                        id='select-activation-function',
                                        options=[
                                            {'label': 'Sigmoid activation function', 'value': 'sigmoid'},
                                            {'label': 'Linear activation function', 'value': 'linear'},
                                            {'label': 'Rectified linear unit activation function', 'value': 'relu'},
                                            {'label': 'Exponential linear unit', 'value': 'elu'}
                                            ],
                                        #value='sigmoid',
                                        placeholder='Select activation function.....',
                                        style={
                                            'width':'50%'
                                        }
                                    )
                                ]),
                                #html.H6(""),
                                #html.H6("Select Optimizer"),
                                html.Div([
                                    dcc.Dropdown(
                                        id='select-optimizer',
                                        options=[
                                            #{'label':'Adadelta algorithm', 'value':'Adadelta'},
                                            #{'label':'Adagrad algorithm', 'value':'Adagrad'},
                                            {'label':'ADAM Optimizer','value':'Adam'},
                                            {'label':'RMSprop algorithm','value':'RMSprop'},
                                            {'label': 'Stochastic Gradient Descent ', 'value': 'SGD'},
                                            #{'label': 'Stochastic Gradient Descent with Momentum', 'value': 'SGDM'},
                                            {'label': 'Particle Swarm Optimization', 'value': 'PSO'},
                                            {'label': 'Gradient Particle Swarm Optimization', 'value': 'GPSO'}
                                            ],
                                        #value='SGD',
                                        placeholder='Select Optimizer...',
                                        style={
                                            'width':'50%'
                                        }
                                    )
                                ]),
                                html.P(),
                                html.Button(id='submit-button-1',
                                            n_clicks=0,
                                            children='Run',
                                            style={'fontSize':14}),
                                html.P(),
                                #html.H6("The accuracy obtained for the model:"),
                                html.Div(id='model-create')
                                ],id='model-training-1',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'background-color':'white',
                                    'border':'2px black solid',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                }),
                                html.Div([
                                    html.Iframe(id='console-out',srcDoc='',style={'width':'100%','background-color': 'rgb(250,250,250)','height':650})
                                ],id='model-training-2',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'background-color':'white',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                })

                            ])
                        ],className="inside_block"),
                        html.Div(id='model-prediction',className="border"),
                        html.Div([
                            html.Div(className='row',children=[
                                html.Div(id='block-label',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'borderRadius': '5px',
                                    'background-color':'white',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                }),
                                html.Div(id='block-label1',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'background-color':'white',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                })
                        ])],className="border"),
                        html.Div([
                            html.Div(className='row',children=[
                                html.Div(id='ann-graph',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'borderRadius': '5px',
                                    'background-color':'white',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                }),
                                html.Div(id='ann-graph1',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'background-color':'white',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                })
                        ])],className="border")]),
                dcc.Tab(label='IANN MODELING', children=[html.Div([
                    html.Div(className='row',children=[
                        html.Div([
                        html.H6(""),
                        html.H6(""),
                        html.H6(""),
                        html.Div([
                            #html.H6("Enter Test, Train and Validation Ratio:"),
                            dcc.Input(id="input6", type="number", placeholder="Train %"),
                            dcc.Input(id="input7", type="number", placeholder="Test %"),
                            dcc.Input(id="input8", type="number", placeholder="Validation %"),
                            ]
                        ),
                        #html.H6("Drop feature:"),
                        dcc.Input(id="input9", type="Text", placeholder="Drop feature"),
                        #html.H6("Scaler:"),
                        #html.Div([
                        #    dcc.Dropdown(
                        #        id='select-scaler',
                        #        options=[
                        #            {'label': 'Standard Scaler', 'value': 'SS'},
                        #            {'label': 'Min Max Scalar', 'value': 'SMMS'},
                        #            {'label': 'Max Abs Scaler', 'value': 'MAS'},
                        #            {'label': 'Normalizer', 'value': 'NORM'}
                        #            ],
                                #value='SS',
                        #        placeholder='Select Scalar...',
                        #        style={
                        #            'width':'50%'
                        #        }
                        #    )
                        #]),
                        #html.H6("Number of layer"),
                        dcc.Input(id="input10", type="number", placeholder="Number of Layer"),
                        #html.P(),
                        #html.I("Type Activation Function From Below List:"),
                        #html.P(),
                        #html.I("1.>Sigmoid activation function 2.> Linear activation function 3.>RELU 4.>ELU"),
                        #html.P(),
                        #dcc.Input(id="input6", type="text", placeholder="Activation Function"),
                        #html.H6("Select activation function:"),
                        html.Div([
                            dcc.Dropdown(
                                id='select-activation-function-1',
                                options=[
                                    {'label': 'Sigmoid activation function', 'value': 'sigmoid'},
                                    {'label': 'Linear activation function', 'value': 'linear'},
                                    {'label': 'Rectified linear unit activation function', 'value': 'relu'},
                                    {'label': 'Exponential linear unit', 'value': 'elu'}
                                    ],
                                #value='sigmoid',
                                placeholder='Select activation function.....',
                                style={
                                    'width':'50%'
                                }
                            )
                        ]),
                        #html.H6(""),
                        #html.H6("Select Optimizer"),
                        html.Div([
                            dcc.Dropdown(
                                id='select-optimizer-1',
                                options=[
                                    {'label': 'Stochastic Gradient Descent ', 'value': 'SGD'},
                                    {'label': 'Stochastic Gradient Descent with Momentum', 'value': 'SGDM'},
                                    {'label': 'Particle Swarm Optimization', 'value': 'PSO'},
                                    {'label': 'Gradient Particle Swarm Optimization', 'value': 'GPSO'}
                                    ],
                                #value='SGD',
                                placeholder='Select Optimizer...',
                                style={
                                    'width':'50%'
                                }
                            )
                        ]),
                        html.P(),
                        html.Button(id='submit-button-2',
                                    n_clicks=0,
                                    children='Submit',
                                    style={'fontSize':14}),
                        html.P(),
                        html.H6("The accuracy obtained for the model:"),
                        html.Div(id='model-create-1')
                        ],id='model-training-3',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'background-color':'white',
                            'borderRadius': '5px',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        }),
                        html.Div(id='model-training-4',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'background-color':'white',
                            'borderRadius': '5px',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        })

                    ])
                ],className="border"),
                html.Div([
                    html.Div(className='row',children=[
                        html.Div(id='block-label2',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'borderRadius': '5px',
                            'background-color':'white',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        }),
                        html.Div(id='block-label3',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'background-color':'white',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'borderRadius': '5px',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        })
                ])],className="border"),
                html.Div([
                    html.Div(className='row',children=[
                        html.Div(id='iann-graph',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'borderRadius': '5px',
                            'background-color':'white',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        }),
                        html.Div(id='iann-graph1',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'background-color':'white',
                            'borderRadius': '5px',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        })
                ])],className="border")
                    ]),
                    dcc.Tab(label='TRAIN IMAGES', children=[
                        html.H6("Select Optimizer"),
                        dcc.Dropdown(
                            id='select-optimizer-2',
                            options=[
                                {'label': 'Stochastic Gradient Descent ', 'value': 'SGD'},
                                {'label': 'Stochastic Gradient Descent with Momentum', 'value': 'SGDM'},
                                {'label': 'Particle Swarm Optimization', 'value': 'PSO'},
                                {'label': 'Gradient Particle Swarm Optimization', 'value': 'GPSO'}
                                ],
                            value='SGD'
                        ),
                        html.Button('Add',id='select-all-checkbox-2'),
                        html.Button('None',id='select-none-checkbox-2'),
                        html.Button('Invert',id='invert-checkbox-2'),
                        html.Div([
                            html.Div(id='block-label-2',
                            className="inline"),
                            html.Div(id='block-label1-2',
                            className="inline")
                        ]),
                        html.H6("The accuracy obtained for the model:"),
                        html.Div(id='model-accuracy-2')
                        ])
                ])
],className="dashboard")

def datasetData(contents,filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

def datasetDataWithoutException(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return df


def labelName(contents, filename, date):
    df = datasetData(contents, filename)
    return html.Div([
        html.H6('Select Column Name:'),
        dcc.Checklist(
            id='labelCheckList',
            options=[{'label': i, 'value': i} for i in df.columns],
            value=[df.columns[0]],
            labelStyle={'display': 'inlineBlock', 'marginRight': '40px'},
			inputStyle={'marginRight': '7.5px'},
            #style={
            #    'textAlign': 'left',
            #}
        )
])

def data_table(contents, filename, date):
    df = datasetData(contents, filename)
    return html.Div([
        html.H6("Data Table of Selected Data:"),
        html.Div(id = "dash-table")
])

def annGraph1(contents, filename, date):
    df = datasetData(contents, filename)
    return html.Div([
        dcc.Dropdown(
        id='graph-input1',
        options=[
            {'label': i, 'value': i} for i in df.columns
        ],
        value= df.columns[0]
        ),
        dcc.Dropdown(
        id='graph-input2',
        options=[
            {'label': i, 'value': i} for i in df.columns
        ],
        value= df.columns[1]
        ),
        html.Div(id = 'graph1')
])


def annGraph2(contents, filename, date):
    df = datasetData(contents, filename)
    return html.Div([
        dcc.Dropdown(
        id='graph-input3',
        options=[
            {'label': i, 'value': i} for i in df.columns
        ],
        value= df.columns[0]
        ),
        dcc.Dropdown(
        id='graph-input4',
        options=[
            {'label': i, 'value': i} for i in df.columns
        ],
        value= df.columns[1]
        ),
        html.Div(id = 'graph2')
])

@app.callback(Output('block-label', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def labelCheckList(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            labelName(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('block-label1', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def dashDataTable(list_of_contents, list_of_names, list_of_dates):
    #print(list_of_contents, list_of_names, list_of_dates)
    if list_of_contents is not None:
        #print("Akash Saurabh")
        #print(zip(list_of_contents, list_of_names, list_of_dates))
        children = [
            data_table(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('dash-table', 'children'),
              [Input('labelCheckList', 'value'),
               Input('upload-data', 'contents')])
def dataTable(labelName, list_of_contents):
    content_type, content_string = list_of_contents[0].split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df=df[labelName]
    df=df.head(8)
    return html.Div([
        #html.H6("To Test the function work"),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        )
])

@app.callback(Output('ann-graph', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def graphPart1(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            annGraph1(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('graph1', 'children'),
              [Input('upload-data', 'contents'),
               Input('graph-input1','value'),
               Input('graph-input2','value')])
def Graph1(list_of_contents1, value1, value2):
    content_type, content_string = list_of_contents1[0].split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return html.Div(dcc.Graph(
        figure={'data':[
            go.Scatter(
                x=df[value1],
                y=df[value2],
                dy=1,
                mode='markers',
                marker={'size':15}
    )],'layout':go.Layout(title='First Figure',hovermode='closest')
    }))


@app.callback(Output('ann-graph1', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def graphPart2(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            annGraph2(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('graph2', 'children'),
              [Input('upload-data', 'contents'),
               Input('graph-input3','value'),
               Input('graph-input4','value')])
def Graph1(list_of_contents1, value1, value2):
    content_type, content_string = list_of_contents1[0].split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return html.Div(dcc.Graph(
        figure={'data':[
            go.Scatter(
                x=df[value1],
                y=df[value2],
                dy=1,
                mode='lines',
                marker={'size':15}
    )],'layout':go.Layout(title='Second Figure',hovermode='closest')
    }))

#@app.callback(Output('model-accuracy', 'children'),
#              [Input('upload-data', 'contents'),
#               Input('input1','value'),
#               Input('input2','value'),
#               Input('input3','value'),
#               Input('input4','value'),
#               Input('input5','value'),
#               Input('input6','value'),
#               Input('select-optimizer','value')])
#def createModel1(list_of_contents2, value1, value2, value3, value4, value5, value6, value7):
    #print(value1, value2, value3, value4)
#    content_type, content_string = list_of_contents2[0].split(',')
#    decoded = base64.b64decode(content_string)
#    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#    return html.Div(df)



@app.callback(Output('model-prediction', 'children'),
              [Input('upload-data','contents'),
               Input('submit-button-1', 'n_clicks')],
              [State('input1','value'),
               State('input2','value'),
               State('input3','value'),
               State('input4','value'),
               State('input_1','value'),
               State('input5','value'),
               #State('select-scalar','value'),
               #State('input6','value'),
               State('select-activation-function','value'),
               State('select-optimizer','value')])
def createModel1_1(list_of_contents2,num_clicks, train, test, val, inputNode, outputNode, numLayer, actFunc, optimizer):
    #print(value1, value2, value3, value4)
    print(train, test, val, inputNode, outputNode, numLayer, actFunc, optimizer)
    #print(list_of_contents2)
    content_type, content_string = list_of_contents2[0].split(',')
    #print(content_type)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    #print(df)
    #print(df[0])
    #print(outputNode)
    inputNode = inputNode.split(',')
    outputNode = outputNode.split(',')
    #print(outputNode)
    #print(label[2])
    #df = df.drop(label,axis=1)
    #X = df.drop(df.columns[0],axis=1)
    X = df[inputNode]
    #y = df[df.columns[0]]
    y = df[outputNode]
    #print(X.shape)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dropout
    from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test,random_state=101)
    #print("Shape of training data:")
    #print(X_train.shape)
    #print(y.shape[1])
    scaler = MinMaxScaler()
    X_train= scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential()
    #print(numLayer)
    for x in range(numLayer):
        #print(x)
        model.add(Dense(X_train.shape[1],activation=actFunc))
        #model.add(Dropout(0.5))
    #model.add(Dense(X_train.shape[1],activation='relu'))
    #model.add(Dense(X_train.shape[1],activation='relu'))
    #model.add(Dense(X_train.shape[1],activation='relu'))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer=optimizer,loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    orig_stdout = sys.stdout
    f = open('out.txt', 'a')
    sys.stdout = f
    model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=40,
          verbose=1,
          callbacks=[early_stop])
    #print("Model creation complete...")
    sys.stdout = orig_stdout
    f.close()
    losses = pd.DataFrame(model.history.history)
    predict = model.predict(X_test)
    predictions = []
    for elem in predict:
        predictions.extend(elem)
    pred_df = pd.Series(predictions, name='predictions')
    y_test = y_test.reset_index(drop = True)
    result = pd.concat([y_test, pred_df], axis=1)
    #predictions.insert(0, "predictions")
    #predictions = pd.DataFrame(predictions)
    #print("Prediction1:",predictions)
    #print("Prediction:",y_test)
    mean_absolute_error = mean_absolute_error(y_test,predict)
    mean_square_erro = mean_squared_error(y_test,predict)
    accuracy = explained_variance_score(y_test,predict)
    #pred_df = [predictions,y_test]
    #result = pd.concat(pred_df)
    #print(result)
    #print("Test01")
    #pred_df = pd.DataFrame (pred_df)
    #print("Test02")
    #print(pred_df)

    result.to_csv('predictions.csv')

    #print(losses)
    #print(predictions)
    #print("AKASH")
    #print(pred_df)
    #print(pred_df['price'])

    #print(predictions.shape, y_test.shape)
    loss = go.Scatter(x=losses['loss'],mode='lines')
    val_loss = go.Scatter(x=losses['val_loss'],mode='lines')
    data1 = [loss,val_loss]
    test_data = go.Scatter(x=result['price'], y=result['predictions'], mode='markers')
    real_data = go.Scatter(x=result['price'], y=result['price'], mode='lines')
    data2 = [test_data, real_data]
    return html.Div([
        dcc.Interval(id='interval1', interval=1 * 1000, disabled=True, n_intervals=0),
        dcc.Interval(id='interval2', interval=5 * 1000, disabled=True, n_intervals=0),
        html.Div([

            html.Div(className='row',children=[
                html.Div([
                    #html.H2("Performance Metrics Regression Prediction", style={"textAlign": "left"}),
                    html.H6("The accuracy in the Training set equal to :"),str(accuracy),
                    html.H6("Mean Absolute Error : "), str(mean_absolute_error),
                    dcc.Graph(figure={'data':data1,'layout':go.Layout(title='Loss Vs Validation Loss ',hovermode='closest')})
                ],id='val_loss_graph',
                className="five column",
                style={
                    'width': '48%',
                    'height': '680px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'border':'2px black solid',
                    'borderRadius': '5px',
                    'background-color':'white',
                    'textAlign': 'left',
                    'margin': '10px',
                    'alignSelf': 'stretch',
                    'display': 'inline-block'
                }),
                html.Div([
                    html.H6("Mean Square Error : "),str(mean_square_erro),
                    dcc.Graph(figure={'data':data2,'layout':go.Layout(title='Loss Vs Validation Loss ',hovermode='closest')})
                ],id='test_real_graph',
                className="five column",
                style={
                    'width': '48%',
                    'height': '680px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'border':'2px black solid',
                    'background-color':'white',
                    'borderRadius': '5px',
                    'textAlign': 'left',
                    'margin': '10px',
                    'alignSelf': 'stretch',
                    'display': 'inline-block'
                })
        ])])

    ])

@app.callback(Output('div-out','children'),
             [Input('interval1', 'n_intervals')])
def update_interval(n):
    #orig_stdout = sys.stdout
    #f = open('out.txt', 'a')
    #sys.stdout = f
    print 'Intervals Passed: Akash ' + str(n)
    #sys.stdout = orig_stdout
    #f.close()
    return 'Intervals Passed: ' + str(n)

@app.callback(Output('console-out','srcDoc'),
             [Input('interval2', 'n_intervals')])
def update_output(n):
    file = open('out.txt', 'r')
    data=''
    lines = file.readlines()
    if lines.__len__()<=20:
        last_lines=lines
    else:
        last_lines = lines[-20:]
    for line in last_lines:
        data=data+line + '<BR>'
    file.close()
    print(data)
    return data

@app.callback(Output('download-link','href'),
             [Input('submit-button-1','n_clicks')])
def downloadCSV(num_click):
    dff = pd.read_csv('./predictions.csv')
    csv_string = dff.to_csv(index=False, encoding='utf-8')
    #print("saurabh")
    #print(csv_string)
    #print(urllib.parse.quote(csv_string))
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string

app.css.append_css({
    "external_url":"https://codepen.io/chriddyp/pen/bWLwgP.css"
})


if __name__ == '__main__':
    app.run_server()
