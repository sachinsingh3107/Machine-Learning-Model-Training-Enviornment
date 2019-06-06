from django.shortcuts import render
from ML_models import Reg, Classify
from home.models import datacenter
from django.shortcuts import redirect
from ML_models import featureOps

# Create your views here.
# superuser : admin admin@gmail.com kumaraayush67


def ind(request):
    if request.method == 'GET':
        return render(request, 'home/t0.html')
    if request.method == 'POST':
        a = datacenter()
        a.dataset = request.FILES['dataset']
        a.save()
        aid = a.id
        return redirect('featureSelection', id=aid)


def fs(request,id):
    if request.method == 'GET':
        a = datacenter.objects.get(id=id)
        list = featureOps.getTitles(a.dataset)
        return render(request, 'home/t1.html',{'titles':list})
    if request.method == 'POST':
        a = datacenter.objects.get(id=id)
        lis = featureOps.getTitles(a.dataset)
        lif = [l for l in lis if request.POST[l] == "1"] # list of features
        lil = [l for l in lis if request.POST[l] == "0"] # list of labels
        print(lif)
        f = a.dataset.file
        featureOps.getFea(a.dataset.path, lif, lil)
        return redirect('processing', id=id)


def proc(request,id):
    if request.method == 'GET':
        a = datacenter.objects.get(id=id)
        lis = featureOps.getTitles(a.dataset)
        return render(request, 'home/t2.html',{'titles':lis})
    if request.method == 'POST':
        a = datacenter.objects.get(id=id)
        lis = featureOps.getTitles(a.dataset)
        a.scal = request.POST['scaler']
        a.save()
        lis = [l for l in lis if request.POST[l] == "1"]
        featureOps.dumm(a.dataset.path, lis)
        return redirect('modelSelection', id=id)


def mod(request, id):

    if request.method == 'GET':
        return render(request, 'home/t3.html')

    if request.method == 'POST':
        a = datacenter.objects.get(id=id)
        if request.POST['model_type'] == "Regression":
            score = Reg.reg(a.dataset.path, request.POST['model_reg'], a.scal)
        elif request.POST['model_type'] == "Classification":
            score = Classify.classify(a.dataset.path, request.POST['model_cla'], a.scal)
        else:
            score = "Something Went Wrong"
        print(score)
        return render(request, 'home/t4.html', score)