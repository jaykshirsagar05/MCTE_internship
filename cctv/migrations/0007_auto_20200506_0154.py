# Generated by Django 3.0.5 on 2020-05-05 20:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cctv', '0006_fieldschema_modelschema'),
    ]

    operations = [
        migrations.DeleteModel(
            name='FieldSchema',
        ),
        migrations.DeleteModel(
            name='ModelSchema',
        ),
    ]
