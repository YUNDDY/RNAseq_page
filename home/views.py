from django.shortcuts import render, redirect
from django.http import HttpResponse
from admin_datta.forms import RegistrationForm, LoginForm, UserPasswordChangeForm, UserPasswordResetForm, UserSetPasswordForm
from django.contrib.auth.views import LoginView, PasswordChangeView, PasswordResetConfirmView, PasswordResetView
from django.views.generic import CreateView
from django.contrib.auth import logout
from django.core.files.uploadedfile import InMemoryUploadedFile
import pandas as pd
import os
import zipfile
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
import base64
import io
from django.contrib.auth.decorators import login_required
import numpy as np
import seaborn as sns
from django.conf import settings
import glob
import shutil
from sklearn.preprocessing import scale
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from adjustText import adjust_text

all_test_ids = []


def generate_summary_table(zip_file_path):
    summary_data = []
    result_df = pd.DataFrame()

    directory_path = 'media/Result/'
    file_list = os.listdir(directory_path)
    for filename in file_list:
       if filename.endswith('.txt'):
          file_path = os.path.join(directory_path, filename)
          os.remove(file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zfile:
        deg_files = [name for name in zfile.namelist() if name.startswith('1.DEG') and name.endswith('.xlsx')]
        for deg_file in deg_files:
            with zfile.open(deg_file) as file:
                xls = pd.ExcelFile(io.BytesIO(file.read()))
                sheet_names = xls.sheet_names
                
                for sheet_name in sheet_names:
                    if sheet_name.startswith('2fold & FDR'):
                        sheet = xls.parse(sheet_name)
                        row_count = len(sheet.index)
                        deg_file_name = os.path.basename(deg_file).split('.xlsx')[0]
                        up_count = len(sheet[sheet['log2(fold_change)'] > 1])
                        down_count = len(sheet[sheet['log2(fold_change)'] < -1])
                        up_test_ids = sheet[sheet['log2(fold_change)'] > 1]['test_id']
                        down_test_ids = sheet[sheet['log2(fold_change)'] < -1]['test_id']
                        summary_data.append((deg_file_name, up_count, down_count, row_count))
                    if sheet_name.startswith('UP'):
                        sheet = xls.parse(sheet_name)
                        up_count = len(sheet.index)
                        up_test_ids = sheet[sheet['logFC'] > 1]['ID']
                    if sheet_name.startswith('DOWN'):
                        sheet = xls.parse(sheet_name)
                        down_count = len(sheet.index)
                        down_test_ids = sheet[sheet['logFC'] < -1]['ID']
                        deg_file_name = os.path.basename(deg_file).split('_vs_')[0]+".vs."+os.path.basename(deg_file).split('_vs_')[1].replace(".edgeR.xlsx","")
                        row_count=up_count+down_count
                        summary_data.append((deg_file_name, up_count, down_count, row_count))
                all_test_ids.extend(up_test_ids)
                all_test_ids.extend(down_test_ids)
                gene_list_file_path = 'media/Result/'+deg_file_name+'.txt'
                with open(gene_list_file_path, 'w') as file:
                   file.write('\n'.join(all_test_ids))
            
    summary_df = pd.DataFrame(summary_data, columns=['Comparison', 'Up', 'Down','Total'])
    return summary_df


def index(request):
    uploaded_file = '00000000_Company_Client.ZIP'
    #request.session['uploaded_file'] = uploaded_file.read()
    extracted_folder = handle_uploaded_zip(os.path.join(settings.MEDIA_ROOT, uploaded_file))
    serviceID=os.path.splitext(uploaded_file)[0]

    df_table = read_fpkx_xlsx(extracted_folder)
    df=df_table.drop(df_table.columns[[0]],axis=1)
    Sample_count=df.shape[1]
    df_table.set_index('test_id', inplace=True)
 
    # Check if any numeric columns exist after dropping the first column (test_id)
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    summary_data = generate_summary_table(os.path.join(settings.MEDIA_ROOT, uploaded_file))
    summary_data_html=summary_data.to_html(index=False)
    summary_data_length=summary_data.shape[1]-1
    summary_data_html = f'<table class="table table-hover" <thead>{summary_data_html[36:]}'
    summary_data_html = summary_data_html.replace('<tr style="text-align: right;">', '<tr style="text-align: center;">')
    summary_data_html = summary_data_html.replace('<th>', '<th align="center">')
    summary_data_html = summary_data_html.replace('<tr>', '<tr align="center">')

    if not numeric_columns:
        # If no numeric columns found, return with an error message
        context = {
            'table_html': None,
            'error_message': 'No numeric columns found in the data.',
        }
    else:
        # Box plot
        plt.figure()
        df[numeric_columns] = np.log10(df[numeric_columns]+1)
        df.boxplot(column=numeric_columns)
        plt.xticks(rotation=90)
        plt.title('Box Plot')
        plt.ylabel('Expression value+1')
        plt.xlabel('Sample ID')
        plt.ylim(-2.0,1.0)
        plt.tight_layout()
        bar_buffer = io.BytesIO()
        plt.savefig(bar_buffer, format='png')
        bar_image = bar_buffer.getvalue()
        bar_buffer.close()
        bar_image_base64 = base64.b64encode(bar_image).decode('utf-8')
        bar_plt_path = f'<div style="text-align: left;margin: 0 0 0 0;"><h5>Box plot<br></h5><br><img src="data:image/png;base64,{bar_image_base64}"/></div>'
        bar_download_link = f'data:image/png;base64,{bar_image_base64}'

        # Density plot
        plt.figure()
        for column in numeric_columns:
            sns.kdeplot(df[column], label=column)
        plt.title('Density Plot (log10 transformed)')
        plt.xlabel('log10(Expression value)')
        plt.legend()
        plt.tight_layout()
        density_buffer = io.BytesIO()
        plt.savefig(density_buffer, format='png')
        density_image = density_buffer.getvalue()
        density_buffer.close()
        density_image_base64 = base64.b64encode(density_image).decode('utf-8')
        density_plt_path = f'<div style="text-align: left;margin: 0 0 0 0;"><h5>Density plot<br></h5><br><img src="data:image/png;base64,{density_image_base64}"/></div>'
        density_download_link = f'data:image/png;base64,{density_image_base64}'
        
        # Correlation plot
        correlation_matrix = df.corr()
        vmin = correlation_matrix.values.min()-0.01
        vmax = correlation_matrix.values.max()
        plt.figure(figsize=(7, 7)) 
        sns.heatmap(correlation_matrix, annot=True, cmap='bwr', vmin=vmin, vmax=vmax, fmt='.3f')
        plt.title('')
        plt.tight_layout()
        cor_buffer = io.BytesIO()
        plt.savefig(cor_buffer, format='png')
        cor_image = cor_buffer.getvalue()
        cor_buffer.close()
        cor_image_base64 = base64.b64encode(cor_image).decode('utf-8')
        cor_html = f'<div style="text-align: left;margin: 0 0 0 0;"><br><img src="data:image/png;base64,{cor_image_base64}"/></div>'
        cor_download_link = f'data:image/png;base64,{cor_image_base64}'

        table_html = df_table.head(10).to_html(index_names=False)
        table_html = f'<table class="table table-hover" <thead>{table_html[36:]}'
        table_html = table_html.replace('<th>', '<th align="center">')
        table_html = table_html.replace('<tr>', '<tr align="center">')

        context = {
            'table_html': table_html,
            'bar_html': bar_plt_path,
            'bar_download_link' : bar_download_link,
            'density_html':density_plt_path,
            'density_download_link': density_download_link,
            'cor_html':cor_html,
            'cor_download_link': cor_download_link,
            '2fold_summary': summary_data_html,
            'segments': Sample_count,
            'serviceID': serviceID,
            'summary_data_length': summary_data_length,
            'MENT':"Example Data Uploaded"
        }

        return render(request, 'pages/index.html', context)

    context = {
        'segments': 50
    }
    return render(request, "pages/index.html", context)


def example_upload(request):
    
    if request.method == 'POST' and request.FILES.get('zip_file'):
        uploaded_file = request.FILES['zip_file']
        request.session['uploaded_file_path'] = uploaded_file.name
        extracted_folder = handle_uploaded_zip(uploaded_file)
        serviceID=os.path.splitext(uploaded_file.name)[0]
        
        df_table = read_fpkx_xlsx(extracted_folder)

        result_folder = os.path.join('media', 'Result')
        #shutil.rmtree(result_folder)
        shutil.rmtree(result_folder)
        os.makedirs(result_folder, exist_ok=True)

        df_table.to_excel("media/Result/Exp.xlsx", index=False)
        request.session['table_file_path'] = "media/Result/Exp.xlsx"

        df=df_table.drop(df_table.columns[[0]],axis=1)
        Sample_count=df.shape[1]
        df_table.set_index('test_id', inplace=True)

        # Check if any numeric columns exist after dropping the first column (test_id)
        numeric_columns = df.select_dtypes(include='number').columns.tolist()

        summary_data = generate_summary_table(uploaded_file)
        summary_data_html=summary_data.to_html(index=False)
        summary_data_length=summary_data.shape[1]-1

        summary_data_html = f'<table class="table table-hover" <thead>{summary_data_html[36:]}'
        summary_data_html = summary_data_html.replace('<tr style="text-align: right;">', '<tr style="text-align: center;">')
        summary_data_html = summary_data_html.replace('<th>', '<th align="center">')
        summary_data_html = summary_data_html.replace('<tr>', '<tr align="center">')

        if not numeric_columns:
            # If no numeric columns found, return with an error message
            context = {
                'table_html': None,
                'error_message': 'No numeric columns found in the data.',
            }
        else:
            # Create a box plot using matplotlib
            plt.figure()
            df[numeric_columns] = np.log10(df[numeric_columns]+1)
            df.boxplot(column=numeric_columns)
            plt.xticks(rotation=90)
            plt.title('Box Plot')
            plt.ylabel('Expression value+1')
            plt.xlabel('Sample ID')
            plt.ylim(-2.0,)
            plt.tight_layout()
            # Save the plot as an image
            bar_buffer = io.BytesIO()
            plt.savefig(bar_buffer, format='png')
            bar_image = bar_buffer.getvalue()
            bar_buffer.close()
            bar_image_base64 = base64.b64encode(bar_image).decode('utf-8')
            bar_plt_path = f'<div style="text-align: left;margin: 0 0 0 0;"><h5>Box plot<br></h5><br><img src="data:image/png;base64,{bar_image_base64}"/></div>'
            bar_download_link = f'data:image/png;base64,{bar_image_base64}'
            # Get the HTML representation of the DataFrame with only the first 10 rows
            #table_html = df.to_html(index=False)

            plt.figure()
            for column in numeric_columns:
                sns.kdeplot(df[column], label=column)
            plt.title('Density Plot (log10 transformed)')
            plt.xlabel('log10(Expression value)')
            plt.legend()
            plt.tight_layout()
            density_buffer = io.BytesIO()
            plt.savefig(density_buffer, format='png')
            density_image = density_buffer.getvalue()
            density_buffer.close()
            density_image_base64 = base64.b64encode(density_image).decode('utf-8')
            density_plt_path = f'<div style="text-align: left;margin: 0 0 0 0;"><h5>Density plot<br></h5><br><img src="data:image/png;base64,{density_image_base64}"/></div>'
            density_download_link = f'data:image/png;base64,{density_image_base64}'

            # Correlation plot
            correlation_matrix = df.corr()
            vmin = correlation_matrix.values.min()-0.01
            vmax = correlation_matrix.values.max()
            num_samples = len(correlation_matrix)  # 데이터 샘플 수
            fig_width = min(50, num_samples * 0.9)
            fig_height = min(50, num_samples * 0.9)
            plt.figure(figsize=(fig_width, fig_height))
            sns.heatmap(correlation_matrix, annot=True, cmap='bwr', vmin=vmin, vmax=vmax,fmt='.3f')
            plt.title('')
            plt.tight_layout()
            cor_buffer = io.BytesIO()
            plt.savefig(cor_buffer, format='png')
            cor_image = cor_buffer.getvalue()
            cor_buffer.close()
            cor_image_base64 = base64.b64encode(cor_image).decode('utf-8')
            cor_html = f'<div style="text-align: left;margin: 0 0 0 0;"><br><img src="data:image/png;base64,{cor_image_base64}"/></div>'
            cor_download_link = f'data:image/png;base64,{cor_image_base64}'

            table_html = df_table.head(10).to_html(index_names=False)
            table_html = f'<table class="table table-hover" <thead>{table_html[36:]}'
            table_html = table_html.replace('<th>', '<th align="center">')
            table_html = table_html.replace('<tr>', '<tr align="center">')
            context = {
                'table_html': table_html,
                'bar_html': bar_plt_path,
                'bar_download_link' : bar_download_link,
                'density_html':density_plt_path,
                'density_download_link': density_download_link,
                'cor_html':cor_html,
                'cor_download_link':cor_download_link,
                '2fold_summary': summary_data_html,
                'segments': Sample_count,
                'serviceID': serviceID,
                'summary_data_length': summary_data_length,
                'MENT':"Data Uploaded"
            }

            return render(request, 'pages/index.html', context)

    context = {
        'segments': 50
    }
    return render(request, "pages/index.html", context)


def handle_uploaded_zip(file):
    zip_file = zipfile.ZipFile(file)
    extracted_folder = os.path.join('media', 'extracted_files')
    shutil.rmtree(extracted_folder)
    os.makedirs(extracted_folder, exist_ok=True)
    zip_file.extractall(extracted_folder)
    zip_file.close()
    return extracted_folder

def read_fpkx_xlsx(file_path):
    FPKM_path = os.path.join(file_path, 'FPKM.xlsx')
    if not os.path.exists(FPKM_path):
        #TPM_path=os.path.join(file_path,'COUNT_EXP.matrix')
        file_pattern = 'COUNT_EXP.matrix.xlsx'
        file_list = glob.glob(os.path.join(file_path, f'*{file_pattern}'))
        if file_list:
            TPM_path = file_list[0]
        df = pd.read_excel(TPM_path)
        selected_columns = [col for col in df.columns if col.endswith('.e')]
        new_column_names = [col[:-2] for col in selected_columns]
        df_selected = df[['test_id'] + selected_columns]
        df_selected.columns = ['test_id'] + new_column_names

        table_data=df_selected
        #return None
    else:
        df = pd.read_excel(FPKM_path)
        table_data=df

    return table_data


def tables(request):
    result_folder = os.path.join('media', 'Result')
    print(result_folder)
    data_files = []
    for root, dirs, files in os.walk(result_folder):
        for file in files:
            if file.endswith(".txt"):
                data_files.append(os.path.join(root, file))
    print(data_files)
    sample_list, pca_html, pca_download_link = PCA_plot('media/Result/Exp.xlsx')
    context = {}
    context = {
      'data_files' : data_files,
      'sample_list':sample_list,
      'pca_html':pca_html,
      'pca_download_link':pca_download_link
      
    }
    if request.method == 'POST':
        selected_data_files = request.POST.getlist('selected_data_files')
        selected_data_file = selected_data_files[0] if selected_data_files else None
        cluster_r = request.POST.get('cluster_r')
        cluster_r = True if cluster_r == 'on' else False
        cluster_c = request.POST.get('cluster_c')
        cluster_c = True if cluster_c == 'on' else False
        gene_view = request.POST.get('gene_view')
        gene_view = True if gene_view == 'on' else False
        color = request.POST.get('color')    
        heatmap_html, heatmap_download_link = None, None
        sample_list, pca_html, pca_download_link = PCA_plot('media/Result/Exp.xlsx')
        if selected_data_file:
            heatmap_html, heatmap_download_link = generate_heatmap('media/Result/Exp.xlsx', selected_data_file, cluster_r, cluster_c, gene_view, color)
            
            context = {
                'data_files' : data_files,
                'selected_data_file':selected_data_file,
                'sample_list':sample_list,
                'heatmap_html': heatmap_html,
                'heatmap_download_link': heatmap_download_link,
                'pca_html':pca_html,
                'pca_download_link':pca_download_link
            }
    
    return render(request, 'pages/tables.html', context)

def pca(request):
    result_folder = os.path.join('media', 'Result')
    print(result_folder)
    data_files = []
    for root, dirs, files in os.walk(result_folder):
        for file in files:
            if file.endswith(".txt"):
                data_files.append(os.path.join(root, file))
    print(data_files)
    sample_list, pca_html, pca_download_link = PCA_plot('media/Result/Exp.xlsx')
    
    context = {}
    context = {
      'data_files' : data_files,
      'sample_list':sample_list,
      'pca_html':pca_html,
      'pca_download_link':pca_download_link
      
    }
    if request.method == 'POST':
        selected_data_files = request.POST.getlist('selected_data_files')
        selected_data_file = selected_data_files[0] if selected_data_files else None
        cluster_r = request.POST.get('cluster_r')
        cluster_r = True if cluster_r == 'on' else False
        cluster_c = request.POST.get('cluster_c')
        cluster_c = True if cluster_c == 'on' else False
        gene_view = request.POST.get('gene_view')
        gene_view = True if gene_view == 'on' else False
        color = request.POST.get('color')    
        heatmap_html, heatmap_download_link = None, None
        sample_list, pca_html, pca_download_link = PCA_plot('media/Result/Exp.xlsx')
        if selected_data_file:
            heatmap_html, heatmap_download_link = generate_heatmap('media/Result/Exp.xlsx', selected_data_file, cluster_r, cluster_c, gene_view, color)
            
            context = {
                'data_files' : data_files,
                'selected_data_file':selected_data_file,
                'sample_list':sample_list,
                'heatmap_html': heatmap_html,
                'heatmap_download_link': heatmap_download_link,
                'pca_html':pca_html,
                'pca_download_link':pca_download_link
            }
    
    return render(request, 'pages/pca.html', context)

def generate_heatmap(exp_file, gene_file, cluster_r, cluster_c, gene_view, color):
    # Read the selected data file and perform necessary operations
    with open(gene_file, 'r') as f:
        gene_list = f.read().splitlines()

    # Replace the following line with your data loading logic
    expression_df = pd.read_excel(exp_file, index_col=0)

    # Merge gene and expression data using gene as the key
    merged_df = expression_df.loc[gene_list]
    print(merged_df)
    scaled_df = scale(merged_df.iloc[:, 1:], axis=1)

    # Convert the scaled data back to a DataFrame
    scaled_df = pd.DataFrame(scaled_df, index=merged_df.index, columns=merged_df.columns[1:])

    rows = merged_df.shape[0]
    cols = merged_df.shape[1] - 1
    fig_width = cols * 1
    fig_height = min(15,rows * 0.07)
    cbar_pos = [0,0,1,0.005]
    cbar_kws = {"orientation": "horizontal", "aspect": 1.0}
    if cluster_c and cluster_r:
        g = sns.clustermap(scaled_df.iloc[:, 1:], z_score=1,cmap=color, square=True, annot=False, figsize=(fig_width, fig_height), col_cluster=True, row_cluster=True, cbar_pos=cbar_pos, cbar_kws=cbar_kws, dendrogram_ratio=(0.15,0.009))
    elif cluster_c:
        g = sns.clustermap(scaled_df.iloc[:, 1:], z_score=1,cmap=color, square=True, annot=False, figsize=(fig_width, fig_height), col_cluster=True, row_cluster=False, cbar_pos=cbar_pos, cbar_kws=cbar_kws, dendrogram_ratio=0.009)
    elif cluster_r:
        g = sns.clustermap(scaled_df.iloc[:, 1:], z_score=1,cmap=color, square=True, annot=False, figsize=(fig_width, fig_height), col_cluster=False, row_cluster=True, cbar_pos=cbar_pos, cbar_kws=cbar_kws, dendrogram_ratio=0.15)
    else:
        g = sns.clustermap(scaled_df.iloc[:, 1:], z_score=1,cmap=color, square=True, annot=False, figsize=(fig_width, fig_height), col_cluster=False, row_cluster=False, cbar_pos=cbar_pos, cbar_kws=cbar_kws, dendrogram_ratio=0.15)
    
    for spine in g.ax_cbar.spines:
        g.ax_cbar.spines[spine].set_linewidth(2)
    if gene_view:
        plt.setp(g.ax_heatmap.get_yticklabels(), visible=False)
    else:
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.cax.remove()
    heatmap_buffer = io.BytesIO()
    plt.savefig(heatmap_buffer, format='png', dpi=100)
    selected_data_file = gene_file.replace('media\\Result\\', '')

    # Encode the heatmap image in base64 format
    exp_file=exp_file,
    heatmap_image = heatmap_buffer.getvalue()
    heatmap_buffer.close()
    heatmap_image_base64 = base64.b64encode(heatmap_image).decode('utf-8')
    #heatmap_html = f'<div style="text-align: left;margin: 0 0 0 0;"><h4>Heatmap for {selected_data_file}</h4><img src="data:image/png;base64,{heatmap_image_base64}"/></div>'
    heatmap_html = f'<div style="text-align: left;margin: 0 0 0 0;"><img src="data:image/png;base64,{heatmap_image_base64}"/></div>'
    #heatmap_html = f'<img src="data:image/png;base64,{heatmap_image_base64}"/>'
    heatmap_download_link = f'data:image/png;base64,{heatmap_image_base64}'

    return heatmap_html, heatmap_download_link

def process_group(request):
    print("provess_group")
    context = {}
    if request.method == 'POST':
        group_info = {}
        for sample_id, group in request.POST.items():
            if sample_id.startswith('group_'):
                sample_key = sample_id.replace('group_', '')
                group_info[sample_key] = group
        sample_list, pca_html, pca_download_link = PCA_plot_custom('media/Result/Exp.xlsx',group_info)
        context = {
           'sample_list':sample_list,
            'pca_html':pca_html,
            'pca_download_link':pca_download_link
        }
    return render(request, 'pages/pca.html', context)
   
def PCA_plot(exp_file):
    print("pca_group")
    df = pd.read_excel(exp_file, index_col=0)
    groups = ', '.join(df.columns)
    sample_list = [{"sample_id": col} for col in df.columns] 
    df=df.transpose()
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(df.values)
        
    # Create a scatter plot with different colors for each group

    fig, ax = plt.subplots(figsize=(12, 10))
    for group in np.unique(groups.split(', ')):
        indices = np.where(df.index == str(group))
        ax.scatter(pca_results[indices[0], 0], pca_results[indices[0], 1], label=group, s=20)
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    frame = legend.get_frame()
    frame.set_edgecolor('1')  # 범례 테두리를 숨김
 
    x_min, x_max = pca_results[:, 0].min()-1000, pca_results[:, 0].max()+1000
    y_min, y_max = pca_results[:, 1].min()-1000, pca_results[:, 1].max()+1000
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.tight_layout()

    arrowprops = {
        'arrowstyle': '-'
        }

    texts = [ax.annotate(text=sample_id, xy=(pca_results[i, 0], pca_results[i, 1]), fontsize=10) for i, sample_id in enumerate(df.index)]
    adjust_text(texts)

    # Save the scatter plot as a PNG image
    scatter_buffer = io.BytesIO()
    plt.savefig(scatter_buffer, format='png')
    scatter_image = scatter_buffer.getvalue()
    scatter_buffer.close()

    # Encode the PNG image as base64
    scatter_image_base64 = base64.b64encode(scatter_image).decode('utf-8')

    pca_html = f'<div style="text-align: left;margin: 0 0 0 0;"><h5>PCA plot</h5><img src="data:image/png;base64,{scatter_image_base64}"/></div>'
    pca_download_link = f'data:image/png;base64,{scatter_image_base64}'

    return sample_list, pca_html, pca_download_link

def PCA_plot_custom(exp_file,group_info):
    print(group_info)
    df = pd.read_excel(exp_file, index_col=0)
    groups = ', '.join(df.columns)
    
    sample_list = [{"sample_id": col} for col in df.columns] 
    df=df.transpose()
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(df.values)
        
    # Create a scatter plot with different colors for each group
    group_colors = {}
    unique_groups = np.unique(list(group_info.values()))
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_groups)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, group in enumerate(unique_groups):
      group_samples = [sample_id for sample_id, g in group_info.items() if g == group]
      color = colors[i % len(colors)]  # 색상 순환 사용
      for sample_id in group_samples:
          if group_info[sample_id]!='':  # 그룹 정보가 비어 있지 않은 경우에만 그래프에 추가
            indices = np.where(df.index == sample_id)
            ax.scatter(pca_results[indices[0], 0], pca_results[indices[0], 1], s=30, c=[color])
      if any(group_info[sample_id] != '' for sample_id in group_samples):
        group_colors[group] = color
      print(group_colors.items())
    for group, color in group_colors.items():
        ax.scatter([], [], label=group, c=[color])

    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    frame = legend.get_frame()
    frame.set_edgecolor('1')
 
    # 공백이 아닌 group 정보를 가지고 있는 sample_ids 리스트 생성
    valid_sample_ids = [sample_id for sample_id, group in group_info.items() if group != '']
    # 해당 sample_ids에 대한 PCA 결과만 추출
    valid_pca_results = pca_results[df.index.isin(valid_sample_ids)]
    # 범위 계산
    x_min, x_max = valid_pca_results[:, 0].min() - 1000, valid_pca_results[:, 0].max() + 1000
    y_min, y_max = valid_pca_results[:, 1].min() - 1000, valid_pca_results[:, 1].max() + 1000
    # 설정
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.tight_layout()

    arrowprops = {
        'arrowstyle': '-'
        }

    texts = [ax.annotate(text=sample_id, xy=(pca_results[i, 0], pca_results[i, 1]), fontsize=10)
         for i, sample_id in enumerate(df.index) if group_info.get(sample_id, '') != '']
    adjust_text(texts)

    # Save the scatter plot as a PNG image
    scatter_buffer = io.BytesIO()
    plt.savefig(scatter_buffer, format='png')
    scatter_image = scatter_buffer.getvalue()
    scatter_buffer.close()

    # Encode the PNG image as base64
    scatter_image_base64 = base64.b64encode(scatter_image).decode('utf-8')

    pca_html = f'<div style="text-align: left;margin: 0 0 0 0;"><h3>Graph</h3><img src="data:image/png;base64,{scatter_image_base64}"/></div>'
    pca_download_link = f'data:image/png;base64,{scatter_image_base64}'

    return sample_list, pca_html, pca_download_link




# Components
@login_required(login_url='/accounts/login/')
def bc_button(request):
  context = {
    'parent': 'basic_components',
    'segment': 'button'
  }
  return render(request, "pages/components/bc_button.html", context)

@login_required(login_url='/accounts/login/')
def bc_badges(request):
  context = {
    'parent': 'basic_components',
    'segment': 'badges'
  }
  return render(request, "pages/components/bc_badges.html", context)

@login_required(login_url='/accounts/login/')
def bc_breadcrumb_pagination(request):
  context = {
    'parent': 'basic_components',
    'segment': 'breadcrumbs_&_pagination'
  }
  return render(request, "pages/components/bc_breadcrumb-pagination.html", context)

@login_required(login_url='/accounts/login/')
def bc_collapse(request):
  context = {
    'parent': 'basic_components',
    'segment': 'collapse'
  }
  return render(request, "pages/components/bc_collapse.html", context)

@login_required(login_url='/accounts/login/')
def bc_tabs(request):
  context = {
    'parent': 'basic_components',
    'segment': 'navs_&_tabs'
  }
  return render(request, "pages/components/bc_tabs.html", context)

@login_required(login_url='/accounts/login/')
def bc_typography(request):
  context = {
    'parent': 'basic_components',
    'segment': 'typography'
  }
  return render(request, "pages/components/bc_typography.html", context)

@login_required(login_url='/accounts/login/')
def icon_feather(request):
  context = {
    'parent': 'basic_components',
    'segment': 'feather_icon'
  }
  return render(request, "pages/components/icon-feather.html", context)


# Forms and Tables
@login_required(login_url='/accounts/login/')
def form_elements(request):
  context = {
    'parent': 'form_components',
    'segment': 'form_elements'
  }
  return render(request, 'pages/form_elements.html', context)

@login_required(login_url='/accounts/login/')
def basic_tables(request):
  context = {
    'parent': 'tables',
    'segment': 'basic_tables'
  }
  return render(request, 'pages/tbl_bootstrap.html', context)

# Chart and Maps
@login_required(login_url='/accounts/login/')
def morris_chart(request):
  context = {
    'parent': 'chart',
    'segment': 'morris_chart'
  }
  return render(request, 'pages/chart-morris.html', context)

@login_required(login_url='/accounts/login/')
def google_maps(request):
  context = {
    'parent': 'maps',
    'segment': 'google_maps'
  }
  return render(request, 'pages/map-google.html', context)

# Authentication
class UserRegistrationView(CreateView):
  template_name = 'accounts/auth-signup.html'
  form_class = RegistrationForm
  success_url = '/accounts/login/'

class UserLoginView(LoginView):
  template_name = 'accounts/auth-signin.html'
  form_class = LoginForm

class UserPasswordResetView(PasswordResetView):
  template_name = 'accounts/auth-reset-password.html'
  form_class = UserPasswordResetForm

class UserPasswrodResetConfirmView(PasswordResetConfirmView):
  template_name = 'accounts/auth-password-reset-confirm.html'
  form_class = UserSetPasswordForm

class UserPasswordChangeView(PasswordChangeView):
  template_name = 'accounts/auth-change-password.html'
  form_class = UserPasswordChangeForm

def logout_view(request):
  logout(request)
  return redirect('/accounts/login/')

@login_required(login_url='/accounts/login/')
def profile(request):
  context = {
    'segment': 'profile',
  }
  return render(request, 'pages/profile.html', context)

@login_required(login_url='/accounts/login/')
def sample_page(request):
  context = {
    'segment': 'sample_page',
  }
  return render(request, 'pages/sample-page.html', context)


