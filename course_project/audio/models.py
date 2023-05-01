from django.db import models

class Audio(models.Model):
    title = models.CharField(max_length=200)
    file = models.FileField(upload_to='audio_files/')
