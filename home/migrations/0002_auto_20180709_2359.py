# -*- coding: utf-8 -*-
# Generated by Django 1.11.10 on 2018-07-09 18:29
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datacenter',
            name='dataset',
            field=models.FileField(blank=True, null=True, upload_to=''),
        ),
        migrations.AlterField(
            model_name='datacenter',
            name='mod',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='datacenter',
            name='mod_type',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='datacenter',
            name='regressor',
            field=models.FileField(blank=True, null=True, upload_to=''),
        ),
    ]
