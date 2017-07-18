"""DsImage URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import include, url
from django.contrib import admin
from Apps import views

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^image/', views.index, name='index'),
    url(r'^search_keyword/', views.search_keyword, name='search_keyword'),
    url(r'^search_file/', views.search_file, name='search_file'),
    url(r'^search_filter/', views.search_filter, name='search_filter'),
    url(r'^quadratic_search/', views.quadratic_search, name='quadratic_search'),
]
