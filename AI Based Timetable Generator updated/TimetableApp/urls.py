from django.urls import path
from . import views
from .views import UploadDataset
from .views import preprocess_dataset
from .views import generate_timetable
from .views import analytics

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.Signup, name='signup'),
    path('login/', views.Login, name='login'),
    path('UploadDataset/', UploadDataset, name='UploadDataset'),
    path('preprocess_dataset/', preprocess_dataset, name='preprocess_dataset'),
    path('generate_timetable/', generate_timetable, name='generate_timetable'),
    path('export/excel/', views.export_timetable_excel, name='export_excel'),
    path('export/pdf/', views.export_timetable_pdf, name='export_pdf'),
    path('analytics/', views.analytics, name='analytics'),


]
