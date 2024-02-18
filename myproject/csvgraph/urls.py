from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('bar_graph/', views.bar_graph, name='bar_graph'),
    path('prescription/',views.prescription,name='prescription'),
    path('prediction/', views.prediction, name='prediction'),
    path('predict/', views.predict_aum, name='predict_aum'),
    path('prediction_result/', views.predict_aum, name='prediction_result'), 
    path('predicted_income_debt/', views.predict_income_debt,name='predicted_income_debt'),
    path('predict_growth_equity/', views.predict_growth_equity, name='predict_growth_equity'),
    path('predict_balanced/', views.predict_balanced, name='predict_balanced'),
    path('predict_liquid/', views.predict_liquid, name='predict_liquid'),
]
