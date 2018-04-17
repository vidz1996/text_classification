from django import forms

class Text_Form(forms.Form):
    text = forms.CharField(max_length=1000)
