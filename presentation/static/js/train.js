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

    try {
        body.loss_function = document.getElementById('lossFunctionSelect').value

        if (body.loss_function === '') {
            body.loss_function = null;
        }
    } catch (ex){
        return false;
    }

    body.model_name = document.getElementById('modelName').value
    body.model_description = document.getElementById('modelDescription').value
    body.is_test_algorithm = document.getElementById('test-algorithm-check').checked;
    body.class_weight = document.getElementById('class-weight-check').checked;
    body.min_sample_count = document.getElementById('min_number_samples').value;

    if(document.getElementById('substructure-check').checked)
    {
        body.substructure_similarity = find_select_value('substructure-similarity-select');
    }

    if(document.getElementById('target-check').checked)
    {
        body.target_similarity = find_select_value('target-similarity-select');
    }

    if(document.getElementById('enzyme-check').checked)
    {
        body.enzyme_similarity = find_select_value('enzyme-similarity-select');
    }

    if(document.getElementById('pathway-check').checked)
    {
        body.pathway_similarity = find_select_value('pathway-similarity-select');
    }

    if(document.getElementById('description-check').checked)
    {
        body.description_embedding = find_select_value('description-embedding-select');
    }

    if(document.getElementById('indication-check').checked)
    {
        body.indication_embedding = find_select_value('indication-embedding-select');
    }

    if(document.getElementById('pharmacodynamics-check').checked)
    {
        body.pharmacodynamics_embedding = find_select_value('pharmacodynamics-embedding-select');
    }

    if(document.getElementById('mechanism-of-action-check').checked)
    {
        body.mechanism_of_action_embedding = find_select_value('mechanism-of-action-embedding-select');
    }

    if(document.getElementById('toxicity-check').checked)
    {
        body.toxicity_embedding = find_select_value('toxicity-embedding-select');
    }

    if(document.getElementById('metabolism-check').checked)
    {
        body.metabolism_embedding = find_select_value('metabolism-embedding-select');
    }

    if(document.getElementById('absorption-check').checked)
    {
        body.absorption_embedding = find_select_value('absorption-embedding-select');
    }

    if(document.getElementById('half-life-check').checked)
    {
        body.half_life_embedding = find_select_value('half-life-embedding-select');
    }

    if(document.getElementById('protein-binding-check').checked)
    {
        body.protein_binding_embedding = find_select_value('protein-binding-embedding-select');
    }

    if(document.getElementById('route-of-elimination-check').checked)
    {
        body.route_of_elimination_embedding = find_select_value('route-of-elimination-embedding-select');
    }

    if(document.getElementById('volume-of-distribution-check').checked)
    {
        body.volume_of_distribution_embedding = find_select_value('volume-of-distribution-embedding-select');
    }

    if(document.getElementById('clearance-check').checked)
    {
        body.clearance_embedding = find_select_value('clearance-embedding-select');
    }

    if(document.getElementById('classification-description-check').checked)
    {
        body.classification_description_embedding = find_select_value('classification-description-embedding-select');
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

function updateTrainModelDescription() {

    const selectedModel = document.getElementById("trainModelSelect").value;

    const descriptionElement = document.getElementById("trainModelDescription");

    fetch(`/training/get_training_model_description?trainModel=${selectedModel}`, {
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
    })
    .catch(error => {
        console.log('Error:', error)
    });

}

function updateTrainModelImage() {

    const selectedModel = document.getElementById("trainModelSelect").value;

    const imageElement = document.getElementById("trainModelImg");

    imageElement.src = `../training/training_models/training_model_images/${selectedModel}.png`

}

function updateScenarioDescription() {

    const selectedModel = document.getElementById("scenarioSelect").value;

    const descriptionElement = document.getElementById("scenarioDescription");

    fetch(`/training/get_scenario_description?scenario=${selectedModel}`, {
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
    })
    .catch(error => {
        console.log('Error:', error)
    });

}

function updateLossFormula() {

    const selectedLoss = document.getElementById("lossFunctionSelect").value;

    const lossFormulaElement = document.getElementById("lossFormula");

    fetch(`/training/get_loss_formula?selected_loss=${selectedLoss}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            lossFormulaElement.textContent = data.data;
        } else {
            console.log('Error: No data found.');
        }
    })
    .catch(error => {
        console.log('Error:', error)
    });

}

function updateTrainDescription(){

    const sections = document.querySelectorAll('.section.column-container');
    const description_field = document.getElementById('modelDescription');

    const values = [];

    sections.forEach(section => {
        const checkedCheckboxes = section.querySelectorAll('input[type="checkbox"]:checked');

        checkedCheckboxes.forEach(checkbox => values.push(checkbox.value));
    });

    let descriptions = []

    values.forEach(value => {
        if (value === 'Substructure') {
            descriptions.push('S');
        }
        else if (value === 'Target'){
            descriptions.push('T');
        }
        else if (value === 'Enzyme'){
            descriptions.push('E');
        }
        else if (value === 'Pathway'){
            descriptions.push('P');
        }
        else if (value === 'Description'){
            descriptions.push('D');
        }
        else if (value === 'Indication'){
            descriptions.push('I');
        }
        else if (value === 'Pharmacodynamics'){
            descriptions.push('Ph');
        }
        else if (value === 'mechanism-of-action'){
            descriptions.push('Moa');
        }
        else if (value === 'Toxicity'){
            descriptions.push('Tox');
        }
        else if (value === 'Metabolism'){
            descriptions.push('M');
        }
        else if (value === 'Absorption'){
            descriptions.push('A');
        }
        else if (value === 'half-life'){
            descriptions.push('Hl');
        }
        else if (value === 'protein-binding'){
            descriptions.push('Pb');
        }
        else if (value === 'route-of-elimination'){
            descriptions.push('Roe');
        }
        else if (value === 'volume-of-distribution'){
            descriptions.push('Vod');
        }
        else if (value === 'Clearance'){
            descriptions.push('C');
        }
        else if (value === 'classification-description'){
            descriptions.push('CD');
        }
        else if (value === 'interaction-description'){
            descriptions.push('Int-D');
        }
    })

    description_field.innerText = descriptions.join(' + ');
}