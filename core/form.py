from django import forms

class GroupForm(forms.Form):
    group_name = forms.CharField(max_length=100)