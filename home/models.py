from django.db import models


class datacenter(models.Model):
    dataset = models.FileField(blank=True, null=True)
    scal = models.CharField(max_length=100,blank=True, null=True)
    regressor = models.FileField(blank=True, null=True)