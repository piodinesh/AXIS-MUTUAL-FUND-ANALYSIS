import pandas as pd
import plotly.graph_objects as go
from django.http import HttpResponse
from django.shortcuts import render
import plotly.express as px


def bar_graph(request):
    data = pd.read_csv('myproject/data.csv')
    period = data['QUARTER'].tolist()
    aum_values = data['Funds of funds- Overseas'].tolist()
    funds_values = data['Funds of funds- Domestic'].tolist()

    data1 = pd.read_csv('myproject/new.csv')
    data3= pd.read_csv('myproject/data3.csv')
 
    #GRAPH 1
    # Create a bar graph for Funds of funds- Overseas using Plotly
    fig_aum = go.Figure()
    fig_aum.add_trace(go.Bar(
        x=period,
        y=aum_values,
        name='Funds of funds- Overseas',
        showlegend=True,
        marker=dict(
            color='rgba(0, 0, 255, 0.7)',
            line=dict(
                color='rgba(8, 48, 107, 1.0)',
                width=1,
            ),
        )
    ))

    fig_aum.update_layout(
        title={'text': "Average Assets Under Management (AAUM)", 'x': 0.5},
        xaxis_title="Period",
        yaxis_title="Funds Of Funds - Overseas",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )
    grouped_data = data.groupby('QUARTER').sum()
    
    quarters = grouped_data.index.tolist()
    aum_values_overseas = grouped_data['Funds of funds- Overseas'].tolist()
    aum_values_domestic = grouped_data['Funds of funds- Domestic'].tolist()
    
    fig_aum_overseas = go.Figure()

    fig_aum_overseas.add_trace(go.Bar(
        x=quarters,
        y=aum_values_overseas,
        name='Funds of funds- Overseas',
        marker=dict(
            color='rgba(0, 0, 255, 0.7)',
            line=dict(
                color='rgba(8, 48, 107, 1.0)',
                width=1,
            ),
        )
    ))

    fig_aum_overseas.update_layout(
        title={'text': "Average Assets Under Management (AAUM) - Overseas", 'x': 0.5},
        xaxis_title="Quarter",
        yaxis_title="Funds Under Management",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )

    fig_aum_domestic = go.Figure()

    fig_aum_domestic.add_trace(go.Bar(
        x=quarters,
        y=aum_values_domestic,
        name='Funds of funds- Domestic',
        marker=dict(
            color='rgba(8, 48, 107, 1.0)',
            line=dict(
                color='rgba(8, 48, 107, 1.0)',
                width=1,
            ),
        )
    ))

    fig_aum_domestic.update_layout(
        title={'text': "Average Assets Under Management (AAUM) - Domestic", 'x': 0.5},
        xaxis_title="Quarter",
        yaxis_title="Funds Under Management",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )

    #GRAPH 2 and 3
    data2 = pd.read_csv('myproject/data2.csv')

    period_stats_aum = data2.groupby('PERIOD')['Funds of funds- Overseas'].agg(['mean', 'median', lambda x: x.mode().iloc[0]]).reset_index()
    period_stats_aum.rename(columns={'<lambda>': 'mode'}, inplace=True)

    period_stats_funds = data2.groupby('PERIOD')['Funds of funds- Domestic'].agg(['mean', 'median', lambda x: x.mode().iloc[0]]).reset_index()
    period_stats_funds.rename(columns={'<lambda>': 'mode'}, inplace=True)

    # Create box plot traces for Funds of funds- Overseas
    period_box_plots_aum = []
    for _, row in period_stats_aum.iterrows():
        box_plot_trace_aum = go.Box(y=data2[data2['PERIOD'] == row['PERIOD']]['Funds of funds- Overseas'],
                                name=row['PERIOD'],
                                boxpoints='all',
                                jitter=0.3,
                                pointpos=-1.8)
        period_box_plots_aum.append(box_plot_trace_aum)

    # Create box plot traces for Funds of funds- Domestic
    period_box_plots_funds = []
    for _, row in period_stats_funds.iterrows():
        box_plot_trace_funds = go.Box(y=data2[data2['PERIOD'] == row['PERIOD']]['Funds of funds- Domestic'],
                                  name=row['PERIOD'],
                                  boxpoints='all',
                                  jitter=0.3,
                                  pointpos=1.8)  # Adjust the point position for separation
        period_box_plots_funds.append(box_plot_trace_funds)

    # Create annotations for mean values
    annotations_aum = []
    for _, row in period_stats_aum.iterrows():
        annotation_aum = dict(x=row['PERIOD'], y=row['mean'], text=f"Mean: {row['mean']:.2f}", showarrow=False, textangle=-90, yshift=80)
        annotations_aum.append(annotation_aum)

    annotations_funds = []
    for _, row in period_stats_funds.iterrows():
        annotation_funds = dict(x=row['PERIOD'], y=row['mean'], text=f"Mean: {row['mean']:.2f}", showarrow=False, textangle=-90, yshift=80)
        annotations_funds.append(annotation_funds)

    # Create layouts for the box plots
    layout_aum = go.Layout(title="Box Plot of Funds of funds- Overseas for Different Periods",
                       xaxis=dict(title="Period"),
                       yaxis=dict(title="Funds of funds- Overseas"),
                       height=700,
                       annotations=annotations_aum)  # Add annotations here

    layout_funds = go.Layout(title="Box Plot of Funds of Funds - Domestic for Different Periods",
                         xaxis=dict(title="Period"),
                         yaxis=dict(title="Funds of Funds - Domestic"),
                         height=700,
                         annotations=annotations_funds)  # Add annotations here

    # Create figures using the traces and layouts
    fig_aum_a = go.Figure(data=period_box_plots_aum, layout=layout_aum)
    fig_funds_a = go.Figure(data=period_box_plots_funds, layout=layout_funds)

    #GRAPH 4
    # Create a bar graph for Funds of funds- Domestic using Plotly
    fig_funds = go.Figure()
    fig_funds.add_trace(go.Bar(
        x=period,
        y=funds_values,
        name='Funds of funds- Domestic',
        showlegend=True,
        marker=dict(
            color='rgba(0, 0, 255, 0.7)',
            line=dict(
                color='rgba(8, 48, 107, 1.0)',
                width=1.5,
            ),
        )
    ))

    fig_funds.update_layout(
        title={'text': "Average Assets Under Management (AAUM)", 'x': 0.5},
        xaxis_title="Period",
        yaxis_title="Funds of funds- Domestic",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )

    # Sort the data by LIQUID SCHEMES in descending order
    data1_sorted = data1.sort_values(by='LIQUID SCHEMES', ascending=False)

    # Get the top 7 states and the remaining states
    top_7_states = data1_sorted.head(7)
    remaining_states = data1_sorted.iloc[7:]

    # Create a line graph for the top 7 states using Plotly Express
    fig_top_7 = px.line(
        top_7_states,
        x='Name of the States/ Union Territories',
        y='LIQUID SCHEMES',
        title="Top 7 Liquid Schemes Contribution by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT', 'LIQUID SCHEMES': 'Liquid Schemes'},
    )
    fig_top_7.update_traces(line=dict(color='rgba(0, 128, 0, 0.7)'))
    fig_top_7.update_layout(
        xaxis_title="State/Union Territory",
        yaxis_title="Contribution (Crores)",
        title_x=0.5,
        title_y=0.9,
        legend=dict(
            x=0.5,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )

    # Create a line graph for the remaining states using Plotly Express
    fig_remaining = px.line(
        remaining_states,
        x='Name of the States/ Union Territories',
        y='LIQUID SCHEMES',
        title="Remaining Liquid Schemes Contribution by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT', 'LIQUID SCHEMES': 'Liquid Schemes'},
    )
    fig_remaining.update_traces(line=dict(color='rgba(0, 0, 128, 0.7)'))
    fig_remaining.update_layout(
        xaxis_title="State/Union Territory",
        yaxis_title="Contribution (Crores)",
        title_x=0.5,
        title_y=0.9,
        legend=dict(
            x=0.5,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )

    #GRAPH 6
    # Create a pie chart for BALANCED SCHEMES using Plotly Express
    fig_balanced_pie = px.pie(
        data1,
        names='Name of the States/ Union Territories',
        values='BALANCED SCHEMES',
        title="Distribution of Balanced Schemes by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT'},
    )

    fig_balanced_pie.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1, 0.1])  # Pull the slices slightly

    fig_balanced_pie.update_layout(
    height=1400,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )
    # Create a pie chart for Mutual Fund using Plotly Express
    fig_category_pie = px.pie(
        data3,
        names='Mutual Fund Category',
        values='Aum(Cr)',
        title="Distribution of Different Categories of Mutual Fund Currently",
        labels={'Mutual Fund Category': 'Catergory'},
    )

    fig_category_pie.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1, 0.1])  # Pull the slices slightly

    fig_category_pie.update_layout(
    height=1400,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )
    fig_category_pie.update_traces(hole=0.4) 
    # Create a pie chart for Scheme using Plotly Express
    fig_scheme_pie = px.pie(
        data3,
        names='Scheme Name',
        values='Aum(Cr)',
        title="Distribution of Different Mutual Fund Schemes Currently",
        labels={'Scheme Name': 'Scheme'},
    )

    fig_scheme_pie.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1, 0.1])  # Pull the slices slightly

    fig_scheme_pie.update_layout(
    height=4000,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )
    # Create a pie chart for LIQUID SCHEMES using Plotly Express
    fig_LIQUID_stacked_bar = px.bar(
        data1,
        x='Name of the States/ Union Territories',
        y='LIQUID SCHEMES',
        title="Distribution of LIQUID Schemes by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT'},
    )


    fig_LIQUID_stacked_bar.update_layout(
    height=1400, # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )
    # Create a pie chart for OTHER DEBT ORIENTED SCHEMES SCHEMES using Plotly Express
    fig_OTHER_DEBT_ORIENTED_SCHEMES_pie = px.pie(
        data1,
        names='Name of the States/ Union Territories',
        values='OTHER DEBT ORIENTED SCHEMES',
        title="Distribution of OTHER DEBT ORIENTED SCHEMES Schemes by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT'},
    )

    fig_OTHER_DEBT_ORIENTED_SCHEMES_pie.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1, 0.1])  # Pull the slices slightly

    fig_OTHER_DEBT_ORIENTED_SCHEMES_pie.update_layout(
    height=1400,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
        
    )
    )
    # Create a pie chart for GROWTH / EQUITY ORIENTED SCHEMES SCHEMES using Plotly Express
    fig_GROWTH_EQUITY_ORIENTED_SCHEMES_pie = px.pie(
        data1,
        names='Name of the States/ Union Territories',
        values='GROWTH / EQUITY ORIENTED SCHEMES',
        title="Distribution of GROWTH / EQUITY ORIENTED SCHEMES Schemes by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT'},
    )


    fig_GROWTH_EQUITY_ORIENTED_SCHEMES_pie.update_layout(
    height=1400,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )
    # Create a pie chart for FUND OF FUNDS INVESTING DOMESTIC SCHEMES using Plotly Express
    fig_FUND_OF_FUNDS_INVESTING_DOMESTIC_pie = px.pie(
        data1,
        names='Name of the States/ Union Territories',
        values='FUND OF FUNDS INVESTING DOMESTIC',
        title="Distribution of FUND OF FUNDS INVESTING DOMESTIC Schemes by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT'},
    )

    fig_FUND_OF_FUNDS_INVESTING_DOMESTIC_pie.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1, 0.1])  # Pull the slices slightly

    fig_FUND_OF_FUNDS_INVESTING_DOMESTIC_pie.update_layout(
    height=1400,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )

    # Create a pie chart for FUND OF FUNDS INVESTING OVERSEAS SCHEMES using Plotly Express
    fig_FUND_OF_FUNDS_INVESTING_OVERSEAS_pie = px.pie(
        data1,
        names='Name of the States/ Union Territories',
        values='FUND OF FUNDS INVESTING OVERSEAS',
        title="Distribution of FUND OF FUNDS INVESTING OVERSEAS Schemes by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT'},
    )

    fig_FUND_OF_FUNDS_INVESTING_OVERSEAS_pie.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1, 0.1])  # Pull the slices slightly

    fig_FUND_OF_FUNDS_INVESTING_OVERSEAS_pie.update_layout(
    height=1400,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )
    # Create a pie chart for GOLD EXCHANGE TRADED FUND SCHEMES using Plotly Express
    fig_GOLD_EXCHANGE_TRADED_FUND_pie = px.pie(
        data1,
        names='Name of the States/ Union Territories',
        values='GOLD EXCHANGE TRADED FUND',
        title="Distribution of GOLD EXCHANGE TRADED FUND Schemes by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT'},
    )

    fig_GOLD_EXCHANGE_TRADED_FUND_pie.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1, 0.1])  # Pull the slices slightly

    fig_GOLD_EXCHANGE_TRADED_FUND_pie.update_layout(
    height=1400,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )
    # Create a pie chart for OTHER EXCHANGE TRADED FUND SCHEMES using Plotly Express
    fig_OTHER_EXCHANGE_TRADED_FUND_pie = px.pie(
        data1,
        names='Name of the States/ Union Territories',
        values='OTHER EXCHANGE TRADED FUND',
        title="Distribution of OTHER EXCHANGE TRADED FUND Schemes by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT'},
    )

    fig_OTHER_EXCHANGE_TRADED_FUND_pie.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1, 0.1])  # Pull the slices slightly

    fig_OTHER_EXCHANGE_TRADED_FUND_pie.update_layout(
    height=1400,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )
    # Create a area chart for TOTAL SCHEMES using Plotly Express
    fig_TOTAL_area = px.area(
        data1,
        x='Name of the States/ Union Territories',
        y='TOTAL',
        title="Distribution of TOTAL Schemes by State/UT - JULY 2023",
        labels={'Name of the States/ Union Territories': 'State/UT'},
    )


    fig_TOTAL_area.update_layout(
    height=1400,  # Set the height of the chart in pixels
    legend=dict(
        x=1, y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(150, 150, 150, 0.8)',
        borderwidth=1,
        orientation='v',
    )
    )


    # Convert the Plotly graphs to HTML and pass them to the template
    aum_graph_html = fig_aum.to_html()
    box_plot_html = fig_aum_a.to_html()
    box_plott_html = fig_funds_a.to_html()
    funds_graph_html = fig_funds.to_html()
    aum_overseas_html = fig_aum_overseas.to_html()
    aum_domestic_html = fig_aum_domestic.to_html()
    top_7_graph_html = fig_top_7.to_html()
    remaining_graph_html = fig_remaining.to_html()
    balanced_pie_graph_html = fig_balanced_pie.to_html()
    category_pie_graph_html = fig_category_pie.to_html()
    scheme_pie_graph_html = fig_scheme_pie.to_html()
    LIQUID_stacked_bar_graph_html = fig_LIQUID_stacked_bar.to_html()
    OTHER_DEBT_ORIENTED_SCHEMES_pie_graph_html = fig_OTHER_DEBT_ORIENTED_SCHEMES_pie.to_html()
    GROWTH_EQUITY_ORIENTED_SCHEMES_pie_graph_html = fig_GROWTH_EQUITY_ORIENTED_SCHEMES_pie.to_html()
    FUND_OF_FUNDS_INVESTING_DOMESTIC_pie_graph_html = fig_FUND_OF_FUNDS_INVESTING_DOMESTIC_pie.to_html()
    FUND_OF_FUNDS_INVESTING_OVERSEAS_pie_graph_html = fig_FUND_OF_FUNDS_INVESTING_OVERSEAS_pie.to_html()
    GOLD_EXCHANGE_TRADED_pie_graph_html = fig_GOLD_EXCHANGE_TRADED_FUND_pie.to_html()
    OTHER_EXCHANGE_TRADED_FUND_pie_graph_html = fig_OTHER_EXCHANGE_TRADED_FUND_pie.to_html()
    TOTAL_area_graph_html = fig_TOTAL_area.to_html()
     
    # Pass the graph HTML to the template
    return render(request, 'bar_graph.html', {'aum_graph_html': aum_graph_html, 'box_plot_html': box_plot_html, 'box_plott_html': box_plott_html, 'funds_graph_html': funds_graph_html,'aum_overseas_html': aum_overseas_html,'aum_domestic_html': aum_domestic_html, 'top_7_graph_html': top_7_graph_html, 'remaining_graph_html': remaining_graph_html, 'balanced_pie_graph_html': balanced_pie_graph_html, 'category_pie_graph_html': category_pie_graph_html, 'scheme_pie_graph_html': scheme_pie_graph_html,  'LIQUID_stacked_bar_graph_html': LIQUID_stacked_bar_graph_html, 'OTHER_DEBT_ORIENTED_SCHEMES_pie_graph_html': OTHER_DEBT_ORIENTED_SCHEMES_pie_graph_html, 'GROWTH_EQUITY_ORIENTED_SCHEMES_pie_graph_html': GROWTH_EQUITY_ORIENTED_SCHEMES_pie_graph_html, 'FUND_OF_FUNDS_INVESTING_DOMESTIC_pie_graph_html': FUND_OF_FUNDS_INVESTING_DOMESTIC_pie_graph_html, 'FUND_OF_FUNDS_INVESTING_OVERSEAS_pie_graph_html': FUND_OF_FUNDS_INVESTING_OVERSEAS_pie_graph_html, 'GOLD_EXCHANGE_TRADED_pie_graph_html': GOLD_EXCHANGE_TRADED_pie_graph_html, 'OTHER_EXCHANGE_TRADED_FUND_pie_graph_html': OTHER_EXCHANGE_TRADED_FUND_pie_graph_html, 'TOTAL_area_graph_html': TOTAL_area_graph_html})

import pandas as pd
from sklearn.linear_model import LinearRegression
from django.shortcuts import render
from datetime import datetime
import numpy as np
import xgboost as xgb
data = pd.read_csv("myproject/mlpred.csv")
data['QUARTER'] = pd.to_datetime(data['QUARTER'], format='%d.%m.%Y')
data['QUARTER_NUM'] = data['QUARTER'].dt.year + (data['QUARTER'].dt.month - 1) / 12
X = data['QUARTER_NUM'].values.reshape(-1, 1)
y = data['Funds of funds- Overseas'].values
model = LinearRegression()
model.fit(X, y) 
def predict_aum(request):
    selected_date = None
    predicted_aum = None

    if request.method == 'POST':
        input_date_str = request.POST.get('input_date')
        try:
            input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
            selected_date = input_date.strftime('%B %d, %Y')
            input_quarter_num = input_date.year + (input_date.month - 1) / 12

            # Create a Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            prediction_linear = linear_model.predict(np.array([[input_quarter_num]]))

            # Create an XGBoost model
            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(X, y)
            prediction_xgb = xgb_model.predict(np.array([[input_quarter_num]]))

            # Combine predictions using a weighted average
            combined_prediction = 0.5 * prediction_linear + 0.5 * prediction_xgb
            predicted_aum = combined_prediction[0]
        except ValueError:
            pass
    # Create a bar graph for Funds of funds- Overseas using Plotly
    data = pd.read_csv('myproject/mlpred.csv')
    period = data['QUARTER'].tolist()
    aum_values = data['Funds of funds- Overseas'].tolist()
    fig_aum = go.Figure()

    # Add the original data bars
    fig_aum.add_trace(go.Bar(
        x=period,
        y=aum_values,
        name='Funds of funds- Overseas',
        showlegend=True,
        marker=dict(
            color='rgba(0, 0, 255, 0.7)',
            line=dict(
                color='rgba(8, 48, 107, 1.0)',
                width=1,
            ),
        )
    ))

    # Add the predicted value as a red bar
    if predicted_aum is not None:
        fig_aum.add_trace(go.Bar(
            x=[selected_date],
            y=[predicted_aum],
            name='Predicted AUM',
            showlegend=True,
            marker=dict(color='green')
        ))

    fig_aum.update_layout(
        title={'text': "Average Assets Under Management (AAUM) - Overseas", 'x': 0.5},
        xaxis_title="Period",
        yaxis_title="Funds Of Funds - Overseas",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )
    aum_graph_html = fig_aum.to_html()
    return render(request, 'prediction_result.html', {'selected_date': selected_date, 'predicted_aum': predicted_aum, 'aum_graph_html': aum_graph_html})
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
def predict_income_debt(request):
    selected_date2 = None
    predicted_income_debt = None

    if request.method == 'POST':
        input_date_str = request.POST.get('input_date')
        try:
            input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
            selected_date2 = input_date.strftime('%B %d, %Y')
            input_month = input_date.strftime('%d.%m.%Y')
            input_month_num = (input_date.year - 2002) * 12 + input_date.month

            # Load the new CSV file
            data = pd.read_csv('myproject/amff.csv')
            months = data['Month'].tolist()
            income_values = data['Income/Debt'].tolist()

            # Prepare X and y for regression
            X_income = np.array(range(1, len(months) + 1)).reshape(-1, 1)
            y_income = np.array(income_values).reshape(-1, 1)

            # Linear Regression for Income/Debt
            linear_model_income = LinearRegression()
            linear_model_income.fit(X_income, y_income)
            prediction_linear_income = linear_model_income.predict(np.array([[input_month_num]]))

            predicted_income_debt = prediction_linear_income[0][0]

        except ValueError:
            pass
    
    # Create the graph
    fig_income_debt = go.Figure()
    fig_income_debt.add_trace(go.Bar(
        x=months,
        y=income_values,
        name='Income/Debt',
        showlegend=True,
        marker=dict(
            color='rgba(0, 0, 255, 0.7)',
            line=dict(
                color='rgba(8, 48, 107, 1.0)',
                width=1,
            ),
        )
    ))
    if predicted_income_debt is not None:
        fig_income_debt.add_trace(go.Bar(
            x=[selected_date2],
            y=[predicted_income_debt],
            name='Predicted Income/Debt',
            showlegend=True,
            marker=dict(color='green'),
            width=1
        ))

    fig_income_debt.update_layout(
        title={'text': "Income/Debt Predictions", 'x': 0.5},
        xaxis_title="Month",
        yaxis_title="Value",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )
    income_debt_graph_html = fig_income_debt.to_html()
    return render(request, 'prediction_income_debt.html', {'selected_date2': selected_date2, 'predicted_income_debt': predicted_income_debt, 'income_debt_graph_html': income_debt_graph_html})
def predict_growth_equity(request):
    selected_date3 = None
    predicted_growth_equity = None

    if request.method == 'POST':
        input_date_str = request.POST.get('input_date')
        try:
            input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
            selected_date3 = input_date.strftime('%B %d, %Y')
            input_month_num = (input_date.year - 2002) * 12 + input_date.month

            # Load the new CSV file
            data = pd.read_csv('myproject/amff.csv')
            months = data['Month'].tolist()
            growth_values = data['Growth/Equity'].tolist()


            # Prepare X and y for regression
            X_income = np.array(range(1, len(months) + 1)).reshape(-1, 1)
            y_income = np.array(growth_values).reshape(-1, 1)

            # Linear Regression for Growth/Equity
            linear_model_income = LinearRegression()
            linear_model_income.fit(X_income, y_income)
            prediction_linear_income = linear_model_income.predict(np.array([[input_month_num]]))

            predicted_growth_equity = prediction_linear_income[0][0]

        except ValueError:
            pass
    
    # Create the graph
    fig_growth_equity = go.Figure()
    fig_growth_equity.add_trace(go.Bar(
        x=months,
        y=growth_values,
        name='Growth/Equity',
        showlegend=True,
        marker=dict(
            color='rgba(0, 0, 255, 0.7)',
            line=dict(
                color='rgba(8, 48, 107, 1.0)',
                width=1,
            ),
        )
    ))
    if predicted_growth_equity is not None:
        fig_growth_equity.add_trace(go.Bar(
            x=[selected_date3],
            y=[predicted_growth_equity],
            name='Predicted Growth/Equity',
            showlegend=True,
            marker=dict(color='green'),
            width=1
        ))

    fig_growth_equity.update_layout(
        title={'text': "Growth/Equity Predictions", 'x': 0.5},
        xaxis_title="Month",
        yaxis_title="Value",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )
    growth_equity_graph_html = fig_growth_equity.to_html()
    return render(request, 'predict_growth_equity.html', {'selected_date3': selected_date3, 'predicted_growth_equity': predicted_growth_equity, 'growth_equity_graph_html': growth_equity_graph_html})

def predict_balanced(request):
    selected_date4 = None
    predicted_balanced = None

    if request.method == 'POST':
        input_date_str = request.POST.get('input_date')
        try:
            input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
            selected_date4 = input_date.strftime('%B %d, %Y')
            input_month_num = (input_date.year - 2002) * 12 + input_date.month

            # Load the new CSV file
            data = pd.read_csv('myproject/amff.csv')
            months = data['Month'].tolist()
            balanced_values = data['Balanced'].tolist()

            # Prepare data for SARIMA
            df = pd.DataFrame({'Month': months, 'Balanced': balanced_values})
            df['Month'] = pd.to_datetime(df['Month'])
            df.set_index('Month', inplace=True)

            X_income = np.array(range(1, len(months) + 1)).reshape(-1, 1)
            y_income = np.array(balanced_values).reshape(-1, 1)

            # Linear Regression for Income/Debt
            linear_model_income = LinearRegression()
            linear_model_income.fit(X_income, y_income)
            prediction_linear_income = linear_model_income.predict(np.array([[input_month_num]]))

            predicted_balanced = prediction_linear_income[0][0]

        except ValueError:
            pass
    
    # Create the graph
    fig_balanced = go.Figure()
    fig_balanced.add_trace(go.Bar(
        x=months,
        y=balanced_values,
        name='Balanced',
        showlegend=True,
        marker=dict(
            color='rgba(0, 0, 255, 0.7)',
            line=dict(
                color='rgba(8, 48, 107, 1.0)',
                width=1,
            ),
        )
    ))
    if predicted_balanced is not None:
        fig_balanced.add_trace(go.Bar(
            x=[selected_date4],
            y=[predicted_balanced],
            name='Predicted Balanced',
            showlegend=True,
            marker=dict(color='green'),
            width=1
        ))

    fig_balanced.update_layout(
        title={'text': "Balanced Predictions", 'x': 0.5},
        xaxis_title="Month",
        yaxis_title="Value",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )
    balanced_graph_html = fig_balanced.to_html()
    return render(request, 'predict_balanced.html', {'selected_date4': selected_date4, 'predicted_balanced': predicted_balanced, 'balanced_graph_html': balanced_graph_html})

def predict_liquid(request):
    selected_date5 = None
    predicted_liquid = None

    if request.method == 'POST':
        input_date_str = request.POST.get('input_date')
        try:
            input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
            selected_date5 = input_date.strftime('%B %d, %Y')
            input_month_num = (input_date.year - 2002) * 12 + input_date.month

            # Load the new CSV file
            data = pd.read_csv('myproject/amff.csv')
            months = data['Month'].tolist()
            liquid_values = data['Liquid'].tolist()

            X_income = np.array(range(1, len(months) + 1)).reshape(-1, 1)
            y_income = np.array(liquid_values).reshape(-1, 1)

            # Linear Regression for Income/Debt
            linear_model_income = LinearRegression()
            linear_model_income.fit(X_income, y_income)
            prediction_linear_income = linear_model_income.predict(np.array([[input_month_num]]))

            predicted_liquid = prediction_linear_income[0][0]

        except ValueError:
            pass
    
    # Create the graph
    fig_liquid = go.Figure()
    fig_liquid.add_trace(go.Bar(
        x=months,
        y=liquid_values,
        name='Liquid',
        showlegend=True,
        marker=dict(
            color='rgba(0, 0, 255, 0.7)',
            line=dict(
                color='rgba(8, 48, 107, 1.0)',
                width=1,
            ),
        )
    ))
    if predicted_liquid is not None:
        fig_liquid.add_trace(go.Bar(
            x=[selected_date5],
            y=[predicted_liquid],
            name='Predicted Liquid',
            showlegend=True,
            marker=dict(color='green'),
            width=1
        ))

    fig_liquid.update_layout(
        title={'text': "Liquid Predictions", 'x': 0.5},
        xaxis_title="Month",
        yaxis_title="Value",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(150, 150, 150, 0.8)',
            borderwidth=1,
            orientation='h',
        )
    )
    liquid_graph_html = fig_liquid.to_html()
    return render(request, 'predict_liquid.html', {'selected_date5': selected_date5, 'predicted_liquid': predicted_liquid, 'liquid_graph_html': liquid_graph_html})


def prediction(request):

    return render(request,'prediction.html')
def home(request):
    
    return render(request,'home.html')
def prescription(request):
    
    return render(request,'prescription.html')