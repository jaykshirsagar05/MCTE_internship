# Generated by Django 3.0.5 on 2020-05-06 17:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cctv', '0010_student'),
    ]

    operations = [
        migrations.AddField(
            model_name='student',
            name='roll_no',
            field=models.CharField(default=1, max_length=50),
            preserve_default=False,
        ),
    ]
