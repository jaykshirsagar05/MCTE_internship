# Generated by Django 3.0.5 on 2020-05-07 09:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cctv', '0011_student_roll_no'),
    ]

    operations = [
        migrations.AlterField(
            model_name='student',
            name='face_feature',
            field=models.BinaryField(blank=True),
        ),
    ]
