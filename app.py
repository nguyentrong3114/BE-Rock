from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import io, os, base64, importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Map model type to module and class
MODEL_MAP = {
    "rock_basic": ("rock_basic", "RockClustering"),
    "rock_hybrid": ("rock", "RockClusteringHybrid"),
    "rock_fast": ("rock_fast", "RockClusteringHybrid")
}

@app.route('/columns', methods=['POST'])
def get_columns():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500
    return jsonify({'columns': df.columns.tolist()})

@app.route('/cluster', methods=['POST'])
def cluster():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500

    selected_columns = request.form.getlist('columns')
    if not selected_columns:
        return jsonify({'error': 'No columns selected'}), 400
    if any(col not in df.columns for col in selected_columns):
        return jsonify({'error': f'Invalid columns selected: {selected_columns}'}), 400

    try:
        k = int(request.form.get('k', 3))
        theta = float(request.form.get('theta', 0.5))
        alpha = float(request.form.get('alpha', 0.5))
    except:
        return jsonify({'error': 'Invalid k, theta or alpha'}), 400

    model_type = request.form.get('model', 'rock_hybrid')
    if model_type not in MODEL_MAP:
        return jsonify({'error': 'Invalid model type'}), 400

    module_name, class_name = MODEL_MAP[model_type]
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)

    df_selected = df[selected_columns].copy()

    for col in df_selected.columns:
        try:
            df_selected[col] = pd.to_numeric(df_selected[col])
        except:
            pass

    numeric_cols = df_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [col for col in selected_columns if col not in numeric_cols]

    if not numeric_cols and not categorical_cols:
        return jsonify({'error': 'No valid numeric or categorical columns found'}), 400

    try:
        if model_type == 'rock_basic':
            model = ModelClass(k=k, theta=theta, metric='jaccard', sample_ratio=0.6)
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_encoded = ohe.fit_transform(df_selected)
            labels = model.fit(X_encoded)
        else:
            model = ModelClass(
                k=k,
                theta=theta,
                alpha=alpha,
                sample_ratio=0.6,
                verbose=False,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols
            )
            labels = model.fit(df_selected)

        df['cluster'] = labels.tolist()
        cluster_summary = df['cluster'].value_counts().sort_index().to_dict()

        return jsonify({
            'summary': {'total': len(df), 'clusters': len(set(labels)), 'sizes': cluster_summary},
            'data': df.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': f'Clustering failed: {str(e)}'}), 500

@app.route('/scatter', methods=['POST'])
def scatter():
    data = request.json.get('data')
    x_col = request.json.get('x_col')
    y_col = request.json.get('y_col')
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    df = pd.DataFrame(data)
    if 'cluster' not in df.columns:
        return jsonify({'error': 'Missing cluster column'}), 400

    for col in df.columns:
        if df[col].dtype == 'object' and col != 'cluster':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    if not x_col or not y_col:
        features = [c for c in df.columns if c != 'cluster']
        if len(features) < 2:
            return jsonify({'error': 'Need at least 2 feature columns'}), 400
        x_col, y_col = features[:2]

    plt.figure(figsize=(8, 6))
    for cid in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cid]
        plt.scatter(subset[x_col], subset[y_col], label=f'Cluster {cid}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.title('Scatter Plot of Clusters')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({'image': encoded})

@app.route('/chart', methods=['POST'])
def chart():
    data = request.json.get('data')
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    df = pd.DataFrame(data)
    if 'cluster' not in df.columns:
        return jsonify({'error': 'Missing cluster column'}), 400
    counts = df['cluster'].value_counts().sort_index()

    plt.figure(figsize=(6,6))
    counts.plot.pie(autopct='%1.1f%%')
    plt.title('Cluster Distribution')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({'image': encoded})
@app.route('/download', methods=['POST'])
def download_result():
    data = request.json.get('data')
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='ClusteredData')

            # Lấy workbook và worksheet
            workbook = writer.book
            worksheet = writer.sheets['ClusteredData']

            # Tính số lượng mỗi cụm
            cluster_counts = df['cluster'].value_counts().sort_index()
            chart_data = pd.DataFrame({
                'Cluster': cluster_counts.index,
                'Count': cluster_counts.values
            })

            # Ghi dữ liệu biểu đồ vào sheet mới
            chart_sheet = workbook.add_worksheet('Chart')
            chart_sheet.write_column('A2', chart_data['Cluster'])
            chart_sheet.write_column('B2', chart_data['Count'])
            chart_sheet.write('A1', 'Cluster')
            chart_sheet.write('B1', 'Count')

            # Vẽ biểu đồ tròn
            pie_chart = workbook.add_chart({'type': 'pie'})
            pie_chart.add_series({
                'name':       'Cluster Distribution',
                'categories': '=Chart!$A$2:$A${}'.format(len(cluster_counts)+1),
                'values':     '=Chart!$B$2:$B${}'.format(len(cluster_counts)+1),
                'data_labels': {'percentage': True},
            })
            pie_chart.set_title({'name': 'Cluster Distribution'})

            # Chèn biểu đồ vào sheet Chart
            chart_sheet.insert_chart('D2', pie_chart)

        output.seek(0)
        return send_file(output, as_attachment=True, download_name='result.xlsx',
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        return jsonify({'error': f'Failed to generate file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)