# Generated by Django 3.0.5 on 2020-05-08 19:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cctv', '0014_auto_20200508_0044'),
    ]

    operations = [
        migrations.AddField(
            model_name='student',
            name='key',
            field=models.BooleanField(default=False),
        ),
    ]
