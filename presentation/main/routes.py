import os

from flask import Blueprint, render_template, send_from_directory, Response, abort

main = Blueprint('main', __name__)


@main.route('/')
def home():
    return render_template('main/index.html')


@main.route('/contact')
def contact():
    return render_template('main/contact.html')


@main.route('/about')
def about():
    return render_template('main/about.html')


@main.route('/drug/visualization')
def drug_visualization():
    return render_template('drug/visualization.html')


@main.route('/drug/information')
def drug_information():
    return render_template('drug/drug_information.html')


@main.route('/drug/drugs')
def drugs():
    return render_template('drug/drugs.html')


@main.route('/drug/similarity')
def smiles_similarity():
    return render_template('drug/smiles_similarity.html')


@main.route('/drug/reduction')
def reduction_smiles_similarity():
    return render_template('drug/reduction_smiles_similarity.html')


@main.route('/drug/details/<drugbank_id>')
def drug_details(drugbank_id: str):
    return render_template('drug/drug_details.html', drugbank_id=drugbank_id)


@main.route('/enzyme/drug_enzymes')
def enzymes():
    return render_template('enzyme/drug_enzymes.html')


@main.route('/target/drug_targets')
def targets():
    return render_template('target/drug_targets.html')


@main.route('/pathway/drug_pathways')
def pathways():
    return render_template('pathway/drug_pathways.html')


@main.route('/enzyme/enzyme_similarity')
def enzyme_similarity():
    return render_template('enzyme/enzyme_similarity.html')


@main.route('/enzyme/reduction_enzyme_similarity')
def enzyme_reduction():
    return render_template('enzyme/reduction_enzyme_similarity.html')


@main.route('/target/target_similarity')
def target_similarity():
    return render_template('target/target_similarity.html')


@main.route('/target/reduction_target_similarity')
def target_reduction():
    return render_template('target/reduction_target_similarity.html')


@main.route('/pathway/pathway_similarity')
def pathway_similarity():
    return render_template('pathway/pathway_similarity.html')


@main.route('/pathway/reduction_pathway_similarity')
def pathway_reduction():
    return render_template('pathway/reduction_pathway_similarity.html')


@main.route('/drug_embedding/text_embedding')
def text_embedding():
    return render_template('drug_embedding/text_embedding.html')


@main.route('/drug_embedding/reduction_embedding')
def reduction_embedding():
    return render_template('drug_embedding/reduction_embedding.html')


@main.route('/training/train')
def train():
    return render_template('training/train.html')


@main.route('/training/training_history')
def training_history():
    return render_template('training/training_history.html')


@main.route('/training/compare')
def training_compare():
    return render_template('training/compare.html')


@main.route('/training/training_history_details/<int:id>')
def training_history_details(id):
    return render_template('training/training_history_details.html', train_history_id=id)


@main.route('/training/training_history_plots/<int:id>')
def training_history_plots(id):
    return render_template('training/training_history_plots.html', train_history_id=id)


@main.route('/training/training_history_plots/<path:filename>')
def serve_training_plots(filename):
    file_path = os.path.join('', filename)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return Response(f.read(), mimetype='image/png')
    else:
        abort(404)
