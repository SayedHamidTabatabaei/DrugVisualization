function train()
{
    let data = find_body()

    showSpinner();

    fetch(`/training/train`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        if (data.status){
            console.log('Success:', data);
            hideSpinner(true);
        }
        else {
            console.log('Failed:', data.error);
            alert(data.error)
            hideSpinner(false);
        }
    })
    .catch(error => {
        console.log('Error:', error.error);
        hideSpinner(false);
    });
}

function find_body(){

    let body = {};

    try {
        body.train_model = document.getElementById('trainModelSelect').value

        if (body.train_model === '') {
            throw new Error('Train model is required');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    body.model_name = document.getElementById('modelName').value
    body.model_description = document.getElementById('modelDescription').value
    body.is_test_algorithm = document.getElementById('test-algorithm-check').checked;

    if(document.getElementById('substructure-check').checked)
    {
        body.substructure_similarity = find_select_value('substructure-similarity-select');
        body.substructure_reduction = find_select_value('substructure-reduction-select');
    }

    if(document.getElementById('target-check').checked)
    {
        body.target_similarity = find_select_value('target-similarity-select');
        body.target_reduction = find_select_value('target-reduction-select');
    }

    if(document.getElementById('enzyme-check').checked)
    {
        body.enzyme_similarity = find_select_value('enzyme-similarity-select');
        body.enzyme_reduction = find_select_value('enzyme-reduction-select');
    }

    if(document.getElementById('pathway-check').checked)
    {
        body.pathway_similarity = find_select_value('pathway-similarity-select');
        body.pathway_reduction = find_select_value('pathway-reduction-select');
    }

    if(document.getElementById('description-check').checked)
    {
        body.description_embedding = find_select_value('description-embedding-select');
        body.description_reduction = find_select_value('description-reduction-select');
    }

    if(document.getElementById('indication-check').checked)
    {
        body.indication_embedding = find_select_value('indication-embedding-select');
        body.indication_reduction = find_select_value('indication-reduction-select');
    }

    if(document.getElementById('pharmacodynamics-check').checked)
    {
        body.pharmacodynamics_embedding = find_select_value('pharmacodynamics-embedding-select');
        body.pharmacodynamics_reduction = find_select_value('pharmacodynamics-reduction-select');
    }

    if(document.getElementById('mechanism-of-action-check').checked)
    {
        body.mechanism_of_action_embedding = find_select_value('mechanism-of-action-embedding-select');
        body.mechanism_of_action_reduction = find_select_value('mechanism-of-action-reduction-select');
    }

    if(document.getElementById('toxicity-check').checked)
    {
        body.toxicity_embedding = find_select_value('toxicity-embedding-select');
        body.toxicity_reduction = find_select_value('toxicity-reduction-select');
    }

    if(document.getElementById('metabolism-check').checked)
    {
        body.metabolism_embedding = find_select_value('metabolism-embedding-select');
        body.metabolism_reduction = find_select_value('metabolism-reduction-select');
    }

    if(document.getElementById('absorption-check').checked)
    {
        body.absorption_embedding = find_select_value('absorption-embedding-select');
        body.absorption_reduction = find_select_value('absorption-reduction-select');
    }

    if(document.getElementById('half-life-check').checked)
    {
        body.half_life_embedding = find_select_value('half-life-embedding-select');
        body.half_life_reduction = find_select_value('half-life-reduction-select');
    }

    if(document.getElementById('protein-binding-check').checked)
    {
        body.protein_binding_embedding = find_select_value('protein-binding-embedding-select');
        body.protein_binding_reduction = find_select_value('protein-binding-reduction-select');
    }

    if(document.getElementById('route-of-elimination-check').checked)
    {
        body.route_of_elimination_embedding = find_select_value('route-of-elimination-embedding-select');
        body.route_of_elimination_reduction = find_select_value('route-of-elimination-reduction-select');
    }

    if(document.getElementById('volume-of-distribution-check').checked)
    {
        body.volume_of_distribution_embedding = find_select_value('volume-of-distribution-embedding-select');
        body.volume_of_distribution_reduction = find_select_value('volume-of-distribution-reduction-select');
    }

    if(document.getElementById('clearance-check').checked)
    {
        body.clearance_embedding = find_select_value('clearance-embedding-select');
        body.clearance_reduction = find_select_value('clearance-reduction-select');
    }

    if(document.getElementById('classification-description-check').checked)
    {
        body.classification_description_embedding = find_select_value('classification-description-embedding-select');
        body.classification_description_reduction = find_select_value('classification-description-reduction-select');
    }

    return body;
}

function find_select_value(dropdown){

    let value = ''

    try {
        value = document.getElementById(dropdown).value

        if (value === '') {
            throw new Error('');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    return value
}

function updateDescription() {

    const selectedModel = document.getElementById("trainModelSelect").value;

    const descriptionElement = document.getElementById("trainModelDescription");

    fetch(`/training/get_training_model_description?start=0&length=10&trainModel=${selectedModel}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            descriptionElement.textContent = data.data;
        } else {
            console.log('Error: No data found.');
        }
        hideSpinner(true);
    })
    .catch(error => {
        console.log('Error:', error)
        hideSpinner(false);
    });

}